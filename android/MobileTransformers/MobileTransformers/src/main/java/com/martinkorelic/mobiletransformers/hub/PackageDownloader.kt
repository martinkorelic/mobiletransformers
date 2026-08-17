package com.martinkorelic.mobiletransformers.hub

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import kotlin.coroutines.coroutineContext

/**
 * #21: the streaming download core — extracted from the WorkManager worker so it is MockWebServer-testable
 * on the JVM. For each repo-relative path: stream a GET (HTTP Range-resume from a sibling `.partial`) into
 * `destRoot/<path>`, hashing while writing, verify SHA-256 against `expectedSha[path]`, delete + retry on
 * mismatch, then publish `.partial` → final. Fails closed after `maxRetries`.
 */
object PackageDownloader {

    /**
     * Progress from *inside* a single file's transfer.
     *
     * The per-file `onProgress` below cannot describe a package whose weights are one 3 GB file: it
     * fires once, at the end. This fires as the bytes land, which is the difference between a
     * progress bar and a frozen screen.
     */
    fun interface ByteProgressListener {
        /**
         * @param path the repo-relative file being transferred.
         * @param deltaBytes bytes written since the previous call (or resumed from a `.partial` on
         *   the first call for a file, so the running total stays honest across a resume).
         * @param fileBytesTotal the file's size when the server declares one, else `null`.
         */
        fun onBytes(path: String, deltaBytes: Long, fileBytesTotal: Long?)
    }

    /** Signals which stage a given file is in, so the caller can distinguish transfer from hashing. */
    fun interface PhaseListener {
        fun onPhase(path: String, phase: DownloadProgress.Phase)
    }

    /**
     * An HTTP client sized for multi-gigabyte bodies.
     *
     * `OkHttpClient()` defaults to a **10-second read timeout**, which is a reasonable default for an
     * API call and a wrong one for streaming model weights off a CDN: one slow window mid-transfer
     * aborts a download that is minutes in, and the resulting `SocketTimeoutException` says nothing
     * about which file or how far it got. The whole-call timeout must stay 0 — capping a 4 GB transfer
     * at any wall-clock figure is a guess about the user's connection.
     */
    fun defaultClient(): OkHttpClient =
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .callTimeout(0, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()

    suspend fun download(
        client: OkHttpClient,
        files: List<String>,
        urlFor: (String) -> String,
        headers: Map<String, String>,
        expectedSha: Map<String, String>,
        destRoot: File,
        maxRetries: Int = 2,
        onProgress: (done: Int, total: Int, path: String) -> Unit = { _, _, _ -> },
        onBytes: ByteProgressListener = ByteProgressListener { _, _, _ -> },
        onPhase: PhaseListener = PhaseListener { _, _ -> },
    ): Unit =
        withContext(Dispatchers.IO) {
            files.forEachIndexed { index, path ->
                coroutineContext.ensureActive()
                val target = File(destRoot, path)
                target.parentFile?.mkdirs()
                val expected = expectedSha[path]

                var attempt = 0
                while (true) {
                    onPhase.onPhase(path, DownloadProgress.Phase.Downloading)
                    val fetched = fetchToFile(client, urlFor(path), headers, target, path, onBytes)
                    onPhase.onPhase(path, DownloadProgress.Phase.Verifying)
                    if (expected == null || fetched.sha256.equals(expected, ignoreCase = true)) break
                    target.delete()
                    if (++attempt > maxRetries) {
                        throw IOException(
                            "checksum mismatch for '$path' after ${attempt} attempt(s): " +
                                "expected $expected, got ${fetched.sha256}",
                        )
                    }
                    // A retry re-downloads the file from scratch, so the bytes already counted for it
                    // are no longer done. Retract them, or the running total drifts past the plan's
                    // size and the progress bar reports more than 100%.
                    onBytes.onBytes(path, -fetched.bytesCounted, null)
                }
                onProgress(index + 1, files.size, path)
            }
        }

    /** What one file's transfer produced: its hash, and how many bytes were reported for it. */
    private data class Fetched(val sha256: String, val bytesCounted: Long)

    private suspend fun fetchToFile(
        client: OkHttpClient,
        url: String,
        headers: Map<String, String>,
        target: File,
        path: String,
        onBytes: ByteProgressListener,
    ): Fetched {
        var counted = 0L
        val partial = File(target.parentFile, target.name + ".partial")
        val existing = if (partial.isFile) partial.length() else 0L

        val builder = Request.Builder().url(url)
        headers.forEach { (k, v) -> builder.header(k, v) }
        if (existing > 0) builder.header("Range", "bytes=$existing-")

        client.newCall(builder.build()).execute().use { resp ->
            val resumed = resp.code == 206 && existing > 0
            if (!resp.isSuccessful) throw IOException("GET $url -> HTTP ${resp.code}")
            val body = resp.body ?: throw IOException("empty body for $url")

            // `contentLength` is the length of THIS response, so on a 206 it is the remainder; the
            // file's real size is that plus what we already hold.
            val declared = body.contentLength().takeIf { it >= 0 }
            val fileTotal = declared?.let { if (resumed) it + existing else it }

            val digest = MessageDigest.getInstance("SHA-256")
            if (resumed) {
                partial.inputStream().use { feed(digest, it) }
                // Count the resumed prefix once, so a resumed download does not appear to restart
                // from zero and then overshoot its own total.
                counted += existing
                onBytes.onBytes(path, existing, fileTotal)
            } else if (partial.exists()) {
                partial.delete() // server ignored Range (200) -> restart cleanly
            }

            FileOutputStream(partial, resumed).use { out ->
                body.byteStream().use { input ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        // Per-chunk, not per-file: cancelling a 3 GB transfer used to be observed
                        // only at the next file boundary, i.e. not at all for a single-weight
                        // package. The `.partial` is left in place, so Range-resume picks it up.
                        coroutineContext.ensureActive()
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        digest.update(buf, 0, n)
                        counted += n
                        onBytes.onBytes(path, n.toLong(), fileTotal)
                    }
                }
            }
            val hex = digest.digest().joinToString("") { "%02x".format(it) }
            if (target.exists()) target.delete()
            if (!partial.renameTo(target)) {
                partial.copyTo(target, overwrite = true)
                partial.delete()
            }
            return Fetched(hex, counted)
        }
    }

    private fun feed(digest: MessageDigest, input: java.io.InputStream) {
        val buf = ByteArray(1 shl 16)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            digest.update(buf, 0, n)
        }
    }
}

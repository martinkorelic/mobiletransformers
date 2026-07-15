package com.martinkorelic.mobiletransformers.hub

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
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

    suspend fun download(
        client: OkHttpClient,
        files: List<String>,
        urlFor: (String) -> String,
        headers: Map<String, String>,
        expectedSha: Map<String, String>,
        destRoot: File,
        maxRetries: Int = 2,
        onProgress: (done: Int, total: Int, path: String) -> Unit = { _, _, _ -> },
    ): Unit =
        withContext(Dispatchers.IO) {
            files.forEachIndexed { index, path ->
                coroutineContext.ensureActive()
                val target = File(destRoot, path)
                target.parentFile?.mkdirs()
                val expected = expectedSha[path]

                var attempt = 0
                while (true) {
                    val actual = fetchToFile(client, urlFor(path), headers, target)
                    if (expected == null || actual.equals(expected, ignoreCase = true)) break
                    target.delete()
                    if (++attempt > maxRetries) {
                        throw IOException(
                            "checksum mismatch for '$path' after ${attempt} attempt(s): " +
                                "expected $expected, got $actual",
                        )
                    }
                }
                onProgress(index + 1, files.size, path)
            }
        }

    private fun fetchToFile(
        client: OkHttpClient,
        url: String,
        headers: Map<String, String>,
        target: File,
    ): String {
        val partial = File(target.parentFile, target.name + ".partial")
        val existing = if (partial.isFile) partial.length() else 0L

        val builder = Request.Builder().url(url)
        headers.forEach { (k, v) -> builder.header(k, v) }
        if (existing > 0) builder.header("Range", "bytes=$existing-")

        client.newCall(builder.build()).execute().use { resp ->
            val resumed = resp.code == 206 && existing > 0
            if (!resp.isSuccessful) throw IOException("GET $url -> HTTP ${resp.code}")
            val body = resp.body ?: throw IOException("empty body for $url")

            val digest = MessageDigest.getInstance("SHA-256")
            if (resumed) {
                partial.inputStream().use { feed(digest, it) }
            } else if (partial.exists()) {
                partial.delete() // server ignored Range (200) -> restart cleanly
            }

            FileOutputStream(partial, resumed).use { out ->
                body.byteStream().use { input ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        digest.update(buf, 0, n)
                    }
                }
            }
            val hex = digest.digest().joinToString("") { "%02x".format(it) }
            if (target.exists()) target.delete()
            if (!partial.renameTo(target)) {
                partial.copyTo(target, overwrite = true)
                partial.delete()
            }
            return hex
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

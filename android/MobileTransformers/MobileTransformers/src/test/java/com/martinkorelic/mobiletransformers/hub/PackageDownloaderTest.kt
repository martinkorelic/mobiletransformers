package com.martinkorelic.mobiletransformers.hub

import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.security.MessageDigest
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** #21: the streaming download core over MockWebServer — sha256 verify + retry-on-mismatch. */
class PackageDownloaderTest {

    private lateinit var server: MockWebServer
    private val dest: File = Files.createTempDirectory("dl").toFile()
    private val client = OkHttpClient()

    @Before
    fun start() {
        server = MockWebServer()
        server.start()
    }

    @After
    fun stop() {
        server.shutdown()
        dest.deleteRecursively()
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun urlFor(path: String): String = server.url("/$path").toString()

    @Test
    fun downloadsAndVerifiesFiles() = runBlocking {
        val body = "hello model".toByteArray()
        server.enqueue(MockResponse().setBody(String(body)))
        PackageDownloader.download(
            client = client,
            files = listOf("variants/cpu-int4/inference/model.onnx"),
            urlFor = ::urlFor,
            headers = emptyMap(),
            expectedSha = mapOf("variants/cpu-int4/inference/model.onnx" to sha256(body)),
            destRoot = dest,
        )
        val f = File(dest, "variants/cpu-int4/inference/model.onnx")
        assertTrue(f.isFile)
        assertEquals("hello model", f.readText())
    }

    @Test
    fun retriesOnChecksumMismatchThenSucceeds() = runBlocking {
        val good = "correct".toByteArray()
        server.enqueue(MockResponse().setBody("corrupted")) // first: wrong bytes
        server.enqueue(MockResponse().setBody(String(good))) // retry: good
        PackageDownloader.download(
            client = client,
            files = listOf("f.bin"),
            urlFor = ::urlFor,
            headers = emptyMap(),
            expectedSha = mapOf("f.bin" to sha256(good)),
            destRoot = dest,
            maxRetries = 2,
        )
        assertEquals("correct", File(dest, "f.bin").readText())
    }

    @Test
    fun failsClosedAfterPersistentMismatch() {
        repeat(4) { server.enqueue(MockResponse().setBody("still wrong")) }
        val ex = assertThrows(IOException::class.java) {
            runBlocking {
                PackageDownloader.download(
                    client = client,
                    files = listOf("f.bin"),
                    urlFor = ::urlFor,
                    headers = emptyMap(),
                    expectedSha = mapOf("f.bin" to sha256("expected".toByteArray())),
                    destRoot = dest,
                    maxRetries = 2,
                )
            }
        }
        assertTrue(ex.message!!.contains("checksum mismatch"))
    }

    @Test
    fun noExpectedShaSkipsVerification() = runBlocking {
        server.enqueue(MockResponse().setBody("{}"))
        PackageDownloader.download(
            client = client,
            files = listOf("mobiletransformers_manifest.json"),
            urlFor = ::urlFor,
            headers = emptyMap(),
            expectedSha = emptyMap(),
            destRoot = dest,
        )
        assertEquals("{}", File(dest, "mobiletransformers_manifest.json").readText())
    }

    // --- byte-level progress ---------------------------------------------------

    /**
     * Progress must arrive *during* a file, not only when it ends.
     *
     * The per-file callback fires once per file. A real package's weights are one or two files of
     * 1–4 GB, so that signal renders as "0 / 2 files" for the whole download — indistinguishable from
     * a stalled connection, which is exactly the report this exists to prevent. A body larger than
     * the 64 KB read buffer must therefore produce more than one byte callback.
     */
    @Test
    fun reportsBytesWhileAFileIsStillTransferring() = runBlocking {
        val body = ByteArray(300_000) { (it % 251).toByte() }
        server.enqueue(MockResponse().setBody(okio.Buffer().write(body)))

        val deltas = mutableListOf<Long>()
        var declaredTotal: Long? = null
        PackageDownloader.download(
            client = client,
            files = listOf("big.bin"),
            urlFor = ::urlFor,
            headers = emptyMap(),
            expectedSha = mapOf("big.bin" to sha256(body)),
            destRoot = dest,
            onBytes = { _, delta, total ->
                deltas += delta
                if (total != null) declaredTotal = total
            },
        )

        assertTrue("expected several in-flight updates, got ${deltas.size}", deltas.size > 1)
        assertEquals(body.size.toLong(), deltas.sum())
        assertEquals(body.size.toLong(), declaredTotal)
    }

    /**
     * A checksum retry re-downloads from scratch, so the bytes already counted must be retracted.
     * Without that the running total exceeds the plan's size and the progress bar passes 100%.
     */
    @Test
    fun aChecksumRetryRetractsTheBytesItAlreadyCounted() = runBlocking {
        val good = "correct".toByteArray()
        server.enqueue(MockResponse().setBody("corrupted"))
        server.enqueue(MockResponse().setBody(String(good)))

        var net = 0L
        PackageDownloader.download(
            client = client,
            files = listOf("f.bin"),
            urlFor = ::urlFor,
            headers = emptyMap(),
            expectedSha = mapOf("f.bin" to sha256(good)),
            destRoot = dest,
            maxRetries = 2,
            onBytes = { _, delta, _ -> net += delta },
        )
        assertEquals("the failed attempt's bytes were never retracted", good.size.toLong(), net)
    }

    /** Downloading and verifying are separately visible: hashing a multi-GB file is its own wait. */
    @Test
    fun reportsTheVerifyPhaseSeparatelyFromTheTransfer() = runBlocking {
        val body = "hello".toByteArray()
        server.enqueue(MockResponse().setBody(String(body)))

        val phases = mutableListOf<DownloadProgress.Phase>()
        PackageDownloader.download(
            client = client,
            files = listOf("f.bin"),
            urlFor = ::urlFor,
            headers = emptyMap(),
            expectedSha = mapOf("f.bin" to sha256(body)),
            destRoot = dest,
            onPhase = { _, phase -> phases += phase },
        )
        assertEquals(
            listOf(DownloadProgress.Phase.Downloading, DownloadProgress.Phase.Verifying),
            phases,
        )
    }

    /**
     * Cancellation must be observed inside the read loop, and must leave the partial bytes behind.
     *
     * `ensureActive()` used to be called only at each *file* boundary, so a single-weight package
     * could not be cancelled at all — the check next ran after the 3 GB file it was meant to
     * interrupt. The surviving `.partial` is what makes a cancelled pull resumable rather than wasted.
     */
    @Test
    fun cancellingMidFileStopsPromptlyAndKeepsTheResumablePartial() = runBlocking {
        val total = 2_000_000
        // Throttled so the transfer is still in flight when the cancel lands: at 16 KB per 50 ms the
        // whole body needs ~6 s, and the cancel arrives after ~0.3 s.
        server.enqueue(
            MockResponse()
                .setBody(okio.Buffer().write(ByteArray(total)))
                .throttleBody(16_384, 50, java.util.concurrent.TimeUnit.MILLISECONDS),
        )

        var counted = 0L
        val job = launch {
            PackageDownloader.download(
                client = client,
                files = listOf("big.bin"),
                urlFor = ::urlFor,
                headers = emptyMap(),
                expectedSha = emptyMap(),
                destRoot = dest,
                onBytes = { _, delta, _ -> counted += delta },
            )
        }
        delay(300)
        job.cancelAndJoin()

        // Prompt: the loop broke while bytes were still arriving, not after the body finished.
        assertTrue("nothing had transferred yet — the test cannot prove promptness", counted > 0)
        assertTrue("the whole body arrived, so nothing was actually interrupted", counted < total)

        // Resumable: the partial survives, and the final name was never published.
        val partial = File(dest, "big.bin.partial")
        assertTrue("no .partial left behind — a cancelled pull cannot resume", partial.isFile)
        assertTrue(partial.length() in 1 until total.toLong())
        assertTrue("the file was published despite cancellation", !File(dest, "big.bin").exists())
    }

    /**
     * The default client must not carry OkHttp's 10-second read timeout: one slow window mid-transfer
     * would abort a download that is minutes in, and the whole-call timeout must stay uncapped
     * because no wall-clock figure is a safe bound on a 4 GB body over an unknown connection.
     */
    @Test
    fun theDefaultClientIsConfiguredForMultiGigabyteBodies() {
        val c = PackageDownloader.defaultClient()
        assertTrue("read timeout is too short for streaming weights", c.readTimeoutMillis >= 60_000)
        assertEquals("a whole-call timeout would cap large downloads", 0, c.callTimeoutMillis)
        assertTrue(c.retryOnConnectionFailure)
    }
}

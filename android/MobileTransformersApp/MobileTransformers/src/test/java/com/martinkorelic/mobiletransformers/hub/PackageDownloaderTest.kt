package com.martinkorelic.mobiletransformers.hub

import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.security.MessageDigest
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
}

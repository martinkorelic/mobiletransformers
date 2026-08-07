package com.martinkorelic.mobiletransformers.internal.runtime

import com.martinkorelic.mobiletransformers.MissingArtifactException
import com.martinkorelic.mobiletransformers.packages.ChecksumVerifier
import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #23: the fail-closed, map-driven merged-weight load precondition (JVM, no device/JNI). Proves the
 * contract #9 writes into `inference/` — `weight_handoff_map.json` + flat `<name>.bin` (+ map `sha256`
 * or sibling `.bin.sha256`) — is validated before a session is ever created, and that a present-but-
 * broken map throws [MissingArtifactException] naming the offending tensor rather than downgrading.
 */
class HandoffPreconditionTest {

    private val dir: File = Files.createTempDirectory("handoff-precondition").toFile()

    @After
    fun cleanup() {
        dir.deleteRecursively()
    }

    private fun writeBin(name: String, bytes: ByteArray): File =
        File(dir, name).apply { writeBytes(bytes) }

    private fun writeMap(vararg entries: String) {
        File(dir, "weight_handoff_map.json").writeText(
            """{"schemaVersion":"1.0","minReaderVersion":"1.0","handoffMode":"external_initializer",""" +
                """"entries":[${entries.joinToString(",")}]}""",
        )
    }

    private fun entry(layer: String, role: String, bin: String, sha: String? = null): String {
        val shaField = sha?.let { ""","sha256":{"$role":"$it"}""" } ?: ""
        return """{"trainingBaseLayerName":"$layer","dtype":"float16","shape":[2,2],""" +
            """"inferenceInitializerNames":{"$role":"$layer.MatMul.$role"},""" +
            """"externalDataLocation":{"$role":"$bin"}$shaField}"""
    }

    @Test
    fun absentMapIsNotReadyAndDoesNotThrow() {
        assertFalse(HandoffPrecondition.loadMergedWeightsReady(dir))
    }

    @Test
    fun emptyEntriesIsNotReady() {
        File(dir, "weight_handoff_map.json")
            .writeText("""{"schemaVersion":"1.0","minReaderVersion":"1.0","entries":[]}""")
        assertFalse(HandoffPrecondition.loadMergedWeightsReady(dir))
    }

    @Test
    fun validMapWithSidecarChecksumIsReady() {
        val bin = writeBin("w0.bin", byteArrayOf(1, 2, 3, 4))
        File(dir, "w0.bin.sha256").writeText(ChecksumVerifier.sha256(bin) + "\n")
        writeMap(entry("layer0", "weight", "w0.bin"))
        assertTrue(HandoffPrecondition.loadMergedWeightsReady(dir))
    }

    @Test
    fun validMapWithMapChecksumIsReady() {
        val bin = writeBin("w0.bin", byteArrayOf(9, 8, 7))
        writeMap(entry("layer0", "weight", "w0.bin", sha = ChecksumVerifier.sha256(bin)))
        assertTrue(HandoffPrecondition.loadMergedWeightsReady(dir))
    }

    /**
     * #9/#23 regression: post-merge load. The device merger rewrites `<name>.bin` and its sidecar but
     * never touches the map, so after an on-device merge the map still carries the exporter's pre-merge
     * base digest. Preferring the map here made a *correct* merge throw — blocking #9's load smoke and
     * #19's train→merge→generate. The sidecar is the live digest and must win.
     */
    @Test
    fun sidecarWinsOverStaleMapChecksum() {
        val bin = writeBin("w0.bin", byteArrayOf(4, 5, 6, 7)) // "post-merge" bytes
        File(dir, "w0.bin.sha256").writeText(ChecksumVerifier.sha256(bin) + "\n")
        writeMap(entry("layer0", "weight", "w0.bin", sha = "11".repeat(32))) // stale shipped digest
        assertTrue(HandoffPrecondition.loadMergedWeightsReady(dir))
    }

    /** The inverse: a stale sidecar is still fail-closed — precedence must not weaken the gate. */
    @Test
    fun staleSidecarStillThrowsEvenWhenMapMatches() {
        val bin = writeBin("w0.bin", byteArrayOf(4, 5, 6, 7))
        File(dir, "w0.bin.sha256").writeText("22".repeat(32) + "\n")
        writeMap(entry("layer0", "weight", "w0.bin", sha = ChecksumVerifier.sha256(bin)))
        val ex = assertThrows(MissingArtifactException::class.java) {
            HandoffPrecondition.loadMergedWeightsReady(dir)
        }
        assertTrue(ex.message!!.contains("checksum mismatch"))
    }

    @Test
    fun missingBinThrowsNamingTensor() {
        writeMap(entry("layer0", "weight", "gone.bin", sha = "deadbeef"))
        val ex = assertThrows(MissingArtifactException::class.java) {
            HandoffPrecondition.loadMergedWeightsReady(dir)
        }
        assertTrue(ex.message!!.contains("layer0"))
        assertTrue(ex.message!!.contains("gone.bin"))
    }

    @Test
    fun checksumMismatchThrows() {
        writeBin("w0.bin", byteArrayOf(1, 2, 3))
        writeMap(entry("layer0", "weight", "w0.bin", sha = "00".repeat(32)))
        val ex = assertThrows(MissingArtifactException::class.java) {
            HandoffPrecondition.loadMergedWeightsReady(dir)
        }
        assertTrue(ex.message!!.contains("checksum mismatch"))
    }

    @Test
    fun missingChecksumSourceThrows() {
        writeBin("w0.bin", byteArrayOf(1))
        writeMap(entry("layer0", "weight", "w0.bin")) // neither map sha256 nor sidecar
        val ex = assertThrows(MissingArtifactException::class.java) {
            HandoffPrecondition.loadMergedWeightsReady(dir)
        }
        assertTrue(ex.message!!.contains("no checksum"))
    }

    @Test
    fun incompatibleMajorSchemaThrows() {
        File(dir, "weight_handoff_map.json")
            .writeText("""{"schemaVersion":"2.0","minReaderVersion":"2.0","entries":[]}""")
        assertThrows(MissingArtifactException::class.java) {
            HandoffPrecondition.loadMergedWeightsReady(dir)
        }
    }

    @Test
    fun invalidJsonThrows() {
        File(dir, "weight_handoff_map.json").writeText("{ not json")
        assertThrows(MissingArtifactException::class.java) {
            HandoffPrecondition.loadMergedWeightsReady(dir)
        }
    }

    @Test
    fun presenceQuerySkipsChecksumsAndDoesNotThrow() {
        writeBin("w0.bin", byteArrayOf(1)) // present but no checksum anywhere
        writeMap(entry("layer0", "weight", "w0.bin"))
        // The full gate would throw ("no checksum"); the cheap capability query only checks existence.
        assertTrue(HandoffPrecondition.mergedWeightsPresent(dir))
    }

    @Test
    fun presenceQueryFalseWhenBinMissing() {
        writeMap(entry("layer0", "weight", "gone.bin"))
        assertFalse(HandoffPrecondition.mergedWeightsPresent(dir))
    }
}

package com.martinkorelic.mobiletransformers.packages

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * Which fine-tuning technique a package carries, and what precision its graph actually is.
 *
 * The exporter has written both since the training stage existed and the device read neither, so an
 * app could not tell a MARS package from a LoRA one — which matters most for MARS, this project's own
 * contribution: someone running the fine-tuning demo could not see which method they were watching.
 *
 * They live in two different files, and that is the part worth pinning: `peftMethods` is a
 * **manifest** field, while `inferenceGraphPrecision` sits in the `inference/optimum_config.json`
 * side-car beside the graph it describes. Reading either from the wrong place returns silently empty.
 */
class PeftMetadataTest {

    @get:Rule
    val temp = TemporaryFolder()

    // --- peftMethods: a manifest field ---------------------------------------------------------

    @Test
    fun `the manifest's declared peft methods are parsed`() {
        val manifest = MobileTransformersManifest.parse(
            """{"schemaVersion":"1.0","peftMethods":["mars"]}""",
        )
        assertEquals(listOf("mars"), manifest.peftMethods)
    }

    @Test
    fun `a package declaring several methods keeps all of them`() {
        val manifest = MobileTransformersManifest.parse(
            """{"schemaVersion":"1.0","peftMethods":["lora","lora-xs"]}""",
        )
        assertEquals(listOf("lora", "lora-xs"), manifest.peftMethods)
    }

    @Test
    fun `an older package that predates the field reads as empty, not as an error`() {
        // Absence means "not declared", never "no PEFT" — and must never fail a load.
        val manifest = MobileTransformersManifest.parse("""{"schemaVersion":"1.0"}""")
        assertTrue(manifest.peftMethods.isEmpty())
    }

    // --- inferenceGraphPrecision: an optimum_config.json field ---------------------------------

    @Test
    fun `the measured graph precision is read from the inference side-car`() {
        val dir = temp.newFolder("inference")
        java.io.File(dir, PackageTask.FILENAME).writeText(
            """{"task":"text-generation-with-past","modelType":"llama","inferenceGraphPrecision":"fp32"}""",
        )
        assertEquals("fp32", PackageTask.read(dir).inferenceGraphPrecision)
    }

    @Test
    fun `a package that never measured its precision reports null rather than guessing`() {
        val dir = temp.newFolder("inference")
        java.io.File(dir, PackageTask.FILENAME).writeText("""{"task":"text-generation"}""")
        assertNull(PackageTask.read(dir).inferenceGraphPrecision)
    }

    @Test
    fun `the variant id is not the precision - cpu-int4 legitimately ships fp32`() {
        // The asymmetry this field exists to expose: the inference export does not quantize, so the
        // directory name says int4 while the graph is fp32. A UI reading the variant id would lie.
        val dir = temp.newFolder("inference")
        java.io.File(dir, PackageTask.FILENAME).writeText(
            """{"task":"text-generation-with-past","quantization":"int4","inferenceGraphPrecision":"fp32"}""",
        )
        assertEquals("fp32", PackageTask.read(dir).inferenceGraphPrecision)
    }

    @Test
    fun `a missing side-car does not throw`() {
        assertNull(PackageTask.read(temp.newFolder("empty")).inferenceGraphPrecision)
    }
}

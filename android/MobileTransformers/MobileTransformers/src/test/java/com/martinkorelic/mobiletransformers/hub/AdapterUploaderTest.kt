package com.martinkorelic.mobiletransformers.hub

import com.martinkorelic.mobiletransformers.MobileTransformersException
import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** #22: adapter package build + Mode-1/Mode-2 gate + privacy-gated card (pure; upload is the device leg). */
class AdapterUploaderTest {

    private val cacheDir: File = Files.createTempDirectory("adapter").toFile()

    @After
    fun cleanup() {
        cacheDir.deleteRecursively()
    }

    private fun seedCache(repoId: String, peftMethod: String, rank: Int?, alpha: Int?) {
        val sanitized = repoId.replace("/", "__")
        val train = File(cacheDir, "$sanitized/train").apply { mkdirs() }
        val cfg = buildString {
            append("{\"peft_method\":\"$peftMethod\"")
            if (rank != null) append(",\"rank\":$rank")
            if (alpha != null) append(",\"alpha\":$alpha")
            append(",\"peft_target\":[\"q_proj\"],\"trainable_parameter_count\":123}")
        }
        File(train, "training_config.json").writeText(cfg)
        File(train, "weight_handoff_map.json").writeText(
            """{"schemaVersion":"1.0","minReaderVersion":"1.0","entries":[
               {"trainingBaseLayerName":"L","inferenceInitializerNames":{"weight":"model.x.MatMul.weight"},
                "externalDataLocation":{"weight":"model.x.MatMul.weight.bin"}}]}""",
        )
    }

    @Test
    fun buildsMetadataFromCache() {
        seedCache("org/model", "lora", 8, 16)
        val meta = AdapterPackageBuilder.build(cacheDir, "org/model")
        assertEquals("lora", meta.peftMethod)
        assertEquals(8, meta.rank)
        assertEquals(16, meta.alpha)
        assertEquals(listOf("model.x.MatMul.weight"), meta.tensorNames)
    }

    @Test
    fun loraWithFactorsIsMode1Peft() {
        seedCache("org/model", "lora", 8, 16)
        val meta = AdapterPackageBuilder.build(cacheDir, "org/model")
        assertEquals(AdapterMode.PEFT, AdapterModeGate.decide(meta))
    }

    @Test
    fun marsIsMode2Native() {
        seedCache("org/mars", "mars", 8, 8)
        val meta = AdapterPackageBuilder.build(cacheDir, "org/mars")
        assertEquals(AdapterMode.NATIVE, AdapterModeGate.decide(meta))
    }

    @Test
    fun loraWithoutRankFallsToNative() {
        seedCache("org/x", "lora", null, null)
        val meta = AdapterPackageBuilder.build(cacheDir, "org/x")
        assertEquals(AdapterMode.NATIVE, AdapterModeGate.decide(meta))
    }

    @Test
    fun cardCarriesPrivacyWarningAndLicense() {
        seedCache("org/model", "lora", 8, 16)
        val meta = AdapterPackageBuilder.build(cacheDir, "org/model")
        val card = AdapterCard.render(meta, AdapterMode.PEFT, baseModelLicense = "Apache-2.0")
        assertTrue(card.contains("Privacy warning"))
        assertTrue(card.contains("## Licenses"))
        assertTrue(card.contains("Apache-2.0"))
        AdapterCard.assertRequiredSections(card) // no throw
    }

    @Test
    fun cardMissingPrivacyWarningFailsClosed() {
        assertThrows(MobileTransformersException::class.java) {
            AdapterCard.assertRequiredSections("## Licenses\n- Base model weights: x")
        }
    }

    @Test
    fun uploadDisabledByDefault() {
        assertFalse(AdapterUploader.uploadEnabled())
    }
}

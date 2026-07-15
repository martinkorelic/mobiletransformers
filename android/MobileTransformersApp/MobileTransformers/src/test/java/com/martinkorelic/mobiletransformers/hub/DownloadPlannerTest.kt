package com.martinkorelic.mobiletransformers.hub

import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** #21: downloadPlan glob expansion against fileSizes (Kotlin mirror of the Python allow-patterns). */
class DownloadPlannerTest {

    private val manifestJson = """
        {
          "schemaVersion": "1.0", "minReaderVersion": "1.0", "defaultVariant": "cpu-int4",
          "variants": [{"id":"cpu-int4","features":["core","inference","train","rag","genai"]}],
          "fileSizes": {
            "mobiletransformers_manifest.json": 1,
            "shared/tokenizer/tokenizer.json": 1,
            "shared/tokenizer/vocab.json": 1,
            "variants/cpu-int4/inference/model.onnx": 1,
            "variants/cpu-int4/inference/genai_config.json": 1,
            "variants/cpu-int4/train/training_model.onnx": 1,
            "variants/cpu-int4/embedding/rag_config.json": 1,
            "variants/cpu-int4/checksums.json": 1
          },
          "downloadPlan": {
            "cpu-int4": {
              "core": ["mobiletransformers_manifest.json", "shared/tokenizer/**"],
              "checksums": ["variants/cpu-int4/checksums.json"],
              "inference": ["variants/cpu-int4/inference/**"],
              "train": ["variants/cpu-int4/train/**"],
              "rag": ["variants/cpu-int4/embedding/**"],
              "genai": ["variants/cpu-int4/inference/genai_config.json"]
            }
          }
        }
    """.trimIndent()

    private val manifest = MobileTransformersManifest.parse(manifestJson)

    @Test
    fun inferenceOnlyPlanExpandsGlobsAndExcludesTrainRag() {
        val files = DownloadPlanner.planFiles(manifest, "cpu-int4", features = setOf("inference"), genai = false)
        assertTrue(files.contains("variants/cpu-int4/inference/model.onnx"))
        assertTrue(files.contains("shared/tokenizer/tokenizer.json"))
        assertTrue(files.contains("shared/tokenizer/vocab.json"))
        assertTrue(files.contains("variants/cpu-int4/checksums.json"))
        assertTrue(files.contains("mobiletransformers_manifest.json"))
        assertFalse(files.contains("variants/cpu-int4/train/training_model.onnx"))
        assertFalse(files.contains("variants/cpu-int4/embedding/rag_config.json"))
        // (genai_config.json lives under inference/, so the inference glob legitimately includes it.)
    }

    @Test
    fun trainAndRagAndGenaiIncludedWhenRequested() {
        val files = DownloadPlanner.planFiles(
            manifest, "cpu-int4", features = setOf("train", "rag"), genai = true,
        )
        assertTrue(files.contains("variants/cpu-int4/train/training_model.onnx"))
        assertTrue(files.contains("variants/cpu-int4/embedding/rag_config.json"))
        assertTrue(files.contains("variants/cpu-int4/inference/genai_config.json"))
    }

    @Test
    fun groupsForAlwaysIncludesCoreChecksumsInference() {
        assertEquals(setOf("core", "checksums", "inference"), DownloadPlanner.groupsFor(emptySet(), false))
    }
}

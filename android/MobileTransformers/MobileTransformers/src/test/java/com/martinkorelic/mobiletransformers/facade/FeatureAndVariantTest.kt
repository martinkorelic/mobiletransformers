package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.packages.NoCompatibleVariantException
import com.martinkorelic.mobiletransformers.packages.VariantSelector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** #17: engine-selector feature semantics + manifest variant selection over the existing #13 classes. */
class FeatureAndVariantTest {

    @Test
    fun genAiAndManualInferenceAreEngineSelectors() {
        assertTrue(ModelFeature.GenAI.isEngineSelector)
        assertTrue(ModelFeature.ManualInference.isEngineSelector)
        // genuine feature groups are NOT engine selectors (never trigger a second download)
        assertTrue(!ModelFeature.Inference.isEngineSelector)
        assertTrue(!ModelFeature.Training.isEngineSelector)
        assertTrue(!ModelFeature.Rag.isEngineSelector)
    }

    private fun manifest() =
        MobileTransformersManifest(
            baseModelId = "org/base",
            defaultVariant = "cpu-int4",
            variants =
                listOf(
                    MobileTransformersManifest.Variant(
                        id = "cpu-int4",
                        executionProvider = "cpu",
                        quantization = "int4",
                        supportedEngines = listOf("native"),
                        abi = listOf("arm64-v8a"),
                        features = listOf("inference", "training"),
                        recommendedDeviceMemoryMb = 2048,
                    ),
                    MobileTransformersManifest.Variant(
                        id = "cpu-qint8",
                        executionProvider = "cpu",
                        quantization = "QInt8",
                        supportedEngines = listOf("native", "genai"),
                        abi = listOf("arm64-v8a"),
                        features = listOf("inference"),
                        recommendedDeviceMemoryMb = 4096,
                    ),
                ),
        )

    @Test
    fun selectsDefaultVariantForArm64() {
        val v = VariantSelector.select(manifest(), abis = listOf("arm64-v8a"))
        assertEquals("cpu-int4", v.id) // smallest recommended memory + defaultVariant tie-break
    }

    @Test
    fun selectsGenaiCapableVariantWhenEngineRequested() {
        val v =
            VariantSelector.select(
                manifest(),
                abis = listOf("arm64-v8a"),
                requestedEngine = "genai",
            )
        assertEquals("cpu-qint8", v.id) // only this variant supports the genai engine
    }

    @Test
    fun rejectsWhenNoAbiMatches() {
        assertThrows(NoCompatibleVariantException::class.java) {
            VariantSelector.select(manifest(), abis = listOf("x86"))
        }
    }
}

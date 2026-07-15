package com.martinkorelic.mobiletransformers.runtime

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * #11: engine selection/fallback + the data-driven EP registry (F3) — all pure, no device/JNI (the GenAI
 * availability probe is injected via [GenAiSupport.probe]).
 */
class RuntimeSelectionTest {

    private val both = setOf("native", "genai")

    @After
    fun restoreProbe() {
        // Reset to a benign default so other test classes aren't affected (production sets the JNI probe).
        GenAiSupport.probe = { false }
    }

    @Test
    fun genaiSelectedOnlyWhenRequestedSupportedAndAvailable() {
        assertEquals(
            InferenceEngine.GENAI,
            ModelRuntimeFactory.selectEngine(InferenceEngine.GENAI, both, InferenceEngine.NATIVE, true),
        )
    }

    @Test
    fun fallsBackToNativeWhenGenaiUnavailable() {
        assertEquals(
            InferenceEngine.NATIVE,
            ModelRuntimeFactory.selectEngine(InferenceEngine.GENAI, both, InferenceEngine.NATIVE, false),
        )
    }

    @Test
    fun fallsBackToNativeWhenVariantDoesNotSupportGenai() {
        assertEquals(
            InferenceEngine.NATIVE,
            ModelRuntimeFactory.selectEngine(InferenceEngine.GENAI, setOf("native"), InferenceEngine.NATIVE, true),
        )
    }

    @Test
    fun nativeRequestAlwaysNative() {
        assertEquals(
            InferenceEngine.NATIVE,
            ModelRuntimeFactory.selectEngine(InferenceEngine.NATIVE, both, InferenceEngine.NATIVE, true),
        )
    }

    @Test
    fun nullRequestUsesDefaultEngine() {
        assertEquals(
            InferenceEngine.GENAI,
            ModelRuntimeFactory.selectEngine(null, both, InferenceEngine.GENAI, true),
        )
        assertEquals(
            InferenceEngine.NATIVE,
            ModelRuntimeFactory.selectEngine(null, both, InferenceEngine.NATIVE, true),
        )
    }

    @Test
    fun epRegistryResolvesOrderedProvidersPerEngine() {
        GenAiSupport.probe = { true }
        assertEquals(
            listOf("cpu", "xnnpack", "nnapi"),
            EngineRegistry.providersFor(InferenceEngine.NATIVE),
        )
        assertEquals(listOf("genai"), EngineRegistry.providersFor(InferenceEngine.GENAI))
    }

    @Test
    fun epRegistryDropsGenaiWhenUnavailable() {
        GenAiSupport.probe = { false }
        assertEquals(emptyList<String>(), EngineRegistry.providersFor(InferenceEngine.GENAI))
        // Native providers are always available (the floor).
        assertEquals(listOf("cpu", "xnnpack", "nnapi"), EngineRegistry.providersFor(InferenceEngine.NATIVE))
    }
}

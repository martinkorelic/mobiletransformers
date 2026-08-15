package com.martinkorelic.mobiletransformers.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What a picker may offer must equal what the loader will accept.
 *
 * ### The defect this pins
 *
 * `RuntimeCapabilities.availableEngines` was assembled in the facade from two conditions — the
 * package ships `inference/genai_config.json`, and the native GenAI probe succeeds — while
 * [ModelRuntimeFactory.selectEngine] applies a third: the manifest variant's `supportedEngines`.
 *
 * FunctionGemma is the package where those diverge. Its `inference/` stage carries a
 * `genai_config.json`, and the probe returns true on every device, but its manifest declares
 * `supportedEngines: ["native"]` because Gemma-3 inference export goes through optimum rather than
 * the vendored GenAI builder. So the facade reported GenAI as available, the app's picker offered
 * it, the user selected it, and the load failed with
 *
 *     Generation session was never created: Engine "GENAI" is unavailable —
 *     explicitly requested but not selectable: supportedEngines=[native]
 *
 * Both halves were behaving correctly in isolation. The bug was that there were two halves.
 */
class EngineSelectionTest {

    private val nativeOnly = setOf("native")
    private val both = setOf("native", "genai")

    @Test
    fun aNativeOnlyManifestDoesNotOfferGenAiEvenWithAGenAiConfigPresent() {
        // Exactly the FunctionGemma package.
        val available = ModelRuntimeFactory.enginesAvailableFor(
            declaredEngines = nativeOnly,
            genaiConfigPresent = true,
            genaiAvailable = true,
        )

        assertEquals(setOf(InferenceEngine.NATIVE), available)
    }

    @Test
    fun aGenAiCapableManifestOffersItWhenTheConfigAndProbeAgree() {
        val available = ModelRuntimeFactory.enginesAvailableFor(
            declaredEngines = both,
            genaiConfigPresent = true,
            genaiAvailable = true,
        )

        assertTrue(InferenceEngine.GENAI in available)
    }

    @Test
    fun nativeIsAlwaysOffered() {
        for (declared in listOf(nativeOnly, both, emptySet(), null)) {
            for (config in listOf(true, false)) {
                for (probe in listOf(true, false)) {
                    val available =
                        ModelRuntimeFactory.enginesAvailableFor(declared, config, probe)
                    assertTrue(
                        "Native is #11's guaranteed floor (declared=$declared)",
                        InferenceEngine.NATIVE in available,
                    )
                }
            }
        }
    }

    @Test
    fun aPackageDeclaringNothingStaysPermissive() {
        // An unknown declaration must not become a narrower one, or upgrading the SDK breaks
        // packages that work today. Same rule as ModelRuntimeFactory.create's default argument.
        val available = ModelRuntimeFactory.enginesAvailableFor(
            declaredEngines = null,
            genaiConfigPresent = true,
            genaiAvailable = true,
        )

        assertTrue(InferenceEngine.GENAI in available)
    }

    @Test
    fun aMissingGenAiConfigWithdrawsTheOffer() {
        val available = ModelRuntimeFactory.enginesAvailableFor(
            declaredEngines = both,
            genaiConfigPresent = false,
            genaiAvailable = true,
        )

        assertFalse(InferenceEngine.GENAI in available)
    }

    @Test
    fun aFailedNativeProbeWithdrawsTheOffer() {
        val available = ModelRuntimeFactory.enginesAvailableFor(
            declaredEngines = both,
            genaiConfigPresent = true,
            genaiAvailable = false,
        )

        assertFalse(InferenceEngine.GENAI in available)
    }

    /**
     * The property, over every combination: offering an engine and then refusing it is the bug, and
     * so is refusing one that was never offered. Exhaustive because there are only 16 states, and
     * because a case-by-case test is what let the manifest condition go missing in the first place.
     */
    @Test
    fun whatIsOfferedIsExactlyWhatTheFactoryWouldSelect() {
        for (declared in listOf(nativeOnly, both, emptySet<String>(), null)) {
            for (configPresent in listOf(true, false)) {
                for (probe in listOf(true, false)) {
                    val offered = ModelRuntimeFactory.enginesAvailableFor(declared, configPresent, probe)

                    val selected = ModelRuntimeFactory.selectEngine(
                        requested = InferenceEngine.GENAI,
                        supportedEngines = declared ?: both,
                        defaultEngine = InferenceEngine.NATIVE,
                        genaiAvailable = probe,
                    )
                    // The facade additionally requires the side-car, which selectEngine never sees.
                    val loadWouldSucceed = configPresent && selected == InferenceEngine.GENAI

                    assertEquals(
                        "declared=$declared config=$configPresent probe=$probe: the picker and the " +
                            "loader disagree about GenAI",
                        loadWouldSucceed,
                        InferenceEngine.GENAI in offered,
                    )
                }
            }
        }
    }
}

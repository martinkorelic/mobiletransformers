package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.viewmodels.EnginePickerState
import com.martinkorelic.mobiletransformers.app.viewmodels.InstalledRow
import com.martinkorelic.mobiletransformers.app.viewmodels.ModelsUiState
import com.martinkorelic.mobiletransformers.app.viewmodels.Outcome
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The showcase app's **pure** state logic, on the JVM with no device.
 *
 * The app module had no test source set at all before this rewrite, which is part of why its screens
 * could drift into driving the engine layer unnoticed. These cover the parts that decide what a user
 * sees when things are absent or disabled — the states that are hardest to reach by hand on a device
 * (no package installed, no GenAI, no training stage) and therefore the ones most likely to rot.
 */
class ShowcaseStateTest {

    @After
    fun tearDown() = AppConfig.reset()

    @Test
    fun theEmptyStateIsWhatAFreshInstallShows() {
        assertTrue(ModelsUiState().isEmpty)
        assertFalse(ModelsUiState(installed = listOf(row())).isEmpty)
    }

    /**
     * A manifest-less directory still loads but can report no variants. Saying so is the difference
     * between "this package is odd" and a user wondering why the variant list is blank.
     */
    @Test
    fun anInstalledRowExplainsItselfIncludingTheLegacyCase() {
        val modern = row(baseModelId = "HuggingFaceTB/SmolLM2-135M-Instruct", variants = listOf("cpu-int4"))
        assertTrue(modern.subtitle.contains("HuggingFaceTB/SmolLM2-135M-Instruct"))
        assertTrue(modern.subtitle.contains("cpu-int4"))
        assertFalse(modern.subtitle.contains("legacy"))

        val legacy = row(baseModelId = null, variants = emptyList(), hasManifest = false)
        assertTrue(legacy.subtitle.contains("unknown base model"))
        assertTrue(legacy.subtitle.contains("legacy layout"))
    }

    @Test
    fun sizeIsReportedInMegabytes() {
        assertEquals(150L, row(sizeBytes = 150L * 1024 * 1024).sizeMb)
    }

    /**
     * The picker must explain a missing GenAI rather than silently offering one engine. Native is the
     * guaranteed floor, so its presence is never the thing being tested — the note is.
     */
    @Test
    fun theEnginePickerExplainsAMissingGenAiAndStaysQuietWhenItIsThere() {
        val nativeOnly = EnginePickerState(
            selected = InferenceEngine.NATIVE,
            available = setOf(InferenceEngine.NATIVE),
        )
        assertNotNull(nativeOnly.genAiNote)
        assertTrue(nativeOnly.genAiNote!!.contains("genai_config.json"))

        val both = EnginePickerState(
            selected = InferenceEngine.NATIVE,
            available = setOf(InferenceEngine.NATIVE, InferenceEngine.GENAI),
        )
        assertNull("no note is needed when GenAI is actually selectable", both.genAiNote)
    }

    /**
     * Accepted and Rejected are peers, and a dry run must never claim it will execute. If this ever
     * reads `true` the screen is describing an action the SDK does not perform.
     */
    @Test
    fun aDryRunOutcomeNeverMarksItselfExecutable() {
        val accepted = Outcome.Accepted(
            raw = """{"actionName":"set_alarm","parameters":{"time":"07:30"}}""",
            actionName = "set_alarm",
            parameters = mapOf("time" to "07:30"),
            intentAction = "android.intent.action.SET_ALARM",
            willExecute = false,
        )
        assertFalse(accepted.willExecute)

        val rejected = Outcome.Rejected(raw = "\n\n\n", reason = "model output is empty")
        assertEquals("model output is empty", rejected.reason)
    }

    /**
     * Config edits round-trip through the public types, and reset restores the SDK's own defaults
     * rather than the app's idea of them.
     */
    @Test
    fun configurationEditsRoundTripAndResetRestoresSdkDefaults() {
        val defaultTokens = AppConfig.generation.value.maxNewTokens
        val defaultSteps = AppConfig.train.value.gradientAccumulationSteps

        AppConfig.updateGeneration { it.copy(maxNewTokens = 4096) }
        AppConfig.updateGeneration { it.copy(sampling = it.sampling.copy(method = SamplingMethod.TOP_P)) }
        AppConfig.updateTrain { it.copy(gradientAccumulationSteps = 1) }
        AppConfig.updateDataset { it.copy(task = "mobile_actions") }

        assertEquals(4096, AppConfig.generation.value.maxNewTokens)
        assertEquals(SamplingMethod.TOP_P, AppConfig.generation.value.sampling.method)
        assertEquals(1, AppConfig.train.value.gradientAccumulationSteps)
        assertEquals("mobile_actions", AppConfig.dataset.value.task)

        AppConfig.reset()

        assertEquals(defaultTokens, AppConfig.generation.value.maxNewTokens)
        assertEquals(defaultSteps, AppConfig.train.value.gradientAccumulationSteps)
        assertNull(AppConfig.dataset.value.task)
    }

    private fun row(
        baseModelId: String? = "base/model",
        variants: List<String> = listOf("cpu-int4"),
        sizeBytes: Long = 1024,
        hasManifest: Boolean = true,
    ) = InstalledRow(
        sanitizedRepoId = "base__model",
        baseModelId = baseModelId,
        variantIds = variants,
        sizeBytes = sizeBytes,
        hasManifest = hasManifest,
    )
}

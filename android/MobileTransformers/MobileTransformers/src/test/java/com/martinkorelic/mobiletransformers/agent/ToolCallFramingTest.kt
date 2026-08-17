package com.martinkorelic.mobiletransformers.agent

import com.martinkorelic.mobiletransformers.GenerateCallback
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.ORTGenerationConfig
import com.martinkorelic.mobiletransformers.RetrieveCallback
import com.martinkorelic.mobiletransformers.TrainCallback
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.federated.FederatedConfig
import com.martinkorelic.mobiletransformers.federated.FederatedRoundResult
import com.martinkorelic.mobiletransformers.federated.LocalRoundTraining
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.PromptStrategy
import com.martinkorelic.mobiletransformers.runtime.ClassificationResult
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.GroundedResult
import com.martinkorelic.mobiletransformers.runtime.IngestResult
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.PushResult
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import com.martinkorelic.mobiletransformers.training.TrainingJob
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * A tool-call prompt must be framed exactly **once**.
 *
 * [ToolPromptBuilder] writes a complete `<start_of_turn>…` turn structure of its own, deliberately:
 * FunctionGemma's shipped chat template is 13 KB of `namespace()`, `dictsort` and macros Pebble cannot
 * evaluate, so relying on the tokenizer to supply the framing was never an option for the one family
 * that most needs it.
 *
 * That made a *second* framing impossible for a while, and the invariant went unenforced as a result:
 * `ORTTokenizerNative` read the chat template only from `tokenizer_config.json`, the exporter has only
 * ever written it to a sibling `chat_template.jinja`, so `chatTemplate` was null for every package and
 * the generator never wrapped anything. Teaching the tokenizer to read that sibling file (see
 * `ChatTemplateResolutionTest`) turns the dormant hazard live for any package whose template Pebble
 * *can* render — SmolLM2's, for one. [toolCallPromptsAreNotWrappedTwice] pins the guard; it fails if
 * `generateToolCall` stops suppressing the template.
 */
class ToolCallFramingTest {

    /** Records the [GenerationConfig] each generate() was handed, which is the thing under test. */
    private class RecordingSession(override val capabilities: RuntimeCapabilities) : ModelSession {
        val configs = mutableListOf<GenerationConfig>()
        val prompts = mutableListOf<String>()

        override suspend fun generate(
            prompt: String,
            config: GenerationConfig,
            callback: GenerateCallback?,
        ): GenerationResult {
            prompts += prompt
            configs += config
            // Not a tool call — generateToolCall then returns NoCall, which is fine: this test is
            // about what went IN, not what came back.
            return GenerationResult(text = "no call here", tokenCount = 3)
        }

        override suspend fun applyPeft(peft: PeftConfig) = Unit
        override suspend fun train(dataset: DatasetConfig, config: TrainConfig, callback: TrainCallback?) =
            TrainingResult(finalStep = 0, merged = false)
        override fun trainingJob(): TrainingJob = throw UnsupportedOperationException()
        override suspend fun merge() = MergeResult(merged = false)
        override suspend fun retrieve(query: String, config: RagConfig, callback: RetrieveCallback?) =
            RetrievalResult()
        override suspend fun ingest(path: String, config: RagConfig, progress: IngestionProgress?) =
            IngestResult(0)
        override suspend fun classify(text: String, device: DeviceConfig, topK: Int) =
            ClassificationResult()
        override suspend fun generateWithRag(
            query: String,
            rag: RagConfig,
            generation: GenerationConfig,
            promptStrategy: PromptStrategy,
            callback: com.martinkorelic.mobiletransformers.GenerateCallback?,
            retrieveCallback: com.martinkorelic.mobiletransformers.RetrieveCallback?,
        ) = GroundedResult("")
        override suspend fun pushAdapter(hubConfig: HubConfig, repoId: String) = PushResult(repoId)
        override suspend fun federatedRound(
            config: FederatedConfig,
            globalRecord: ByteArray?,
            roundNumber: Int,
            localTraining: LocalRoundTraining,
            metrics: Map<String, Double>,
            train: Boolean,
        ) = FederatedRoundResult(round = roundNumber, importedTensors = 0, update = ByteArray(0), trainedLocally = false)
        override fun close() = Unit
    }

    private fun caps() =
        RuntimeCapabilities(
            engine = InferenceEngine.NATIVE,
            supportsTraining = false,
            supportsMerge = false,
            supportsRag = false,
            supportsEmbedding = false,
        )

    private fun validator() =
        FunctionCallValidator(
            listOf(
                ActionSpec(
                    actionName = "set_alarm",
                    parameters = mapOf("time" to "string"),
                    allowedIntent = "android.intent.action.SET_ALARM",
                    validationRules = mapOf("time" to "HH:mm"),
                    privacyClass = "harmless-demo",
                ),
            ),
        )

    /** The guard: having framed the turns itself, the facade must suppress the tokenizer's template. */
    @Test
    fun toolCallPromptsAreNotWrappedTwice() = runBlocking {
        val session = RecordingSession(caps())
        val model = MobileTransformerModel(session, caps(), "test/repo")

        model.generateToolCall("wake me at 07:30", validator(), parser = ToolCallParser.FunctionGemma)

        assertEquals(1, session.configs.size)
        assertFalse(
            "generateToolCall framed the prompt itself, so the chat template must be suppressed",
            session.configs.single().applyChatTemplate,
        )
        assertTrue(
            "sanity: the prompt really was framed by ToolPromptBuilder",
            session.prompts.single().contains("<start_of_turn>"),
        )
    }

    /**
     * The distinction the blanket version got wrong. The JSON dialect emits **no** turn markers — just
     * declarations and the instruction — so the chat template is what supplies the framing there.
     * Suppressing it would strip framing rather than de-duplicate it, leaving a JSON-dialect model on
     * a template-carrying package worse off than before the template was ever read.
     */
    @Test
    fun jsonDialectToolCallsKeepTheTemplateBecauseTheyAreNotSelfFramed() = runBlocking {
        val session = RecordingSession(caps())
        val model = MobileTransformerModel(session, caps(), "test/repo")

        model.generateToolCall("wake me at 07:30", validator(), parser = ToolCallParser.Json)

        assertTrue(
            "the JSON branch frames no turns, so the template must still apply",
            session.configs.single().applyChatTemplate,
        )
        assertFalse(session.prompts.single().contains("<start_of_turn>"))
    }

    /** The predicate the facade keys on must agree with what the builder actually emits. */
    @Test
    fun framesOwnTurnsAgreesWithTheRenderedPrompt() {
        val allowlist = validator().allowlist
        for (parser in listOf(ToolCallParser.FunctionGemma, ToolCallParser.Json)) {
            val rendered = ToolPromptBuilder.prompt(allowlist, parser, "hi")
            assertEquals(
                "framesOwnTurns disagrees with the rendered prompt for $parser",
                ToolPromptBuilder.framesOwnTurns(parser),
                rendered.contains("<start_of_turn>"),
            )
        }
    }

    /**
     * `declareTools = false` means the caller supplied the whole prompt and wants the session's normal
     * behaviour, template included. Suppressing it there would silently strip framing the caller was
     * relying on.
     */
    @Test
    fun anUnframedToolCallLeavesTheTemplateAlone() = runBlocking {
        val session = RecordingSession(caps())
        val model = MobileTransformerModel(session, caps(), "test/repo")

        model.generateToolCall("wake me at 07:30", validator(), declareTools = false)

        assertTrue(session.configs.single().applyChatTemplate)
        assertEquals("wake me at 07:30", session.prompts.single())
    }

    /** Plain chat keeps the template. This is the whole point of reading `chat_template.jinja`. */
    @Test
    fun plainGenerationKeepsTheTemplate() = runBlocking {
        val session = RecordingSession(caps())
        val model = MobileTransformerModel(session, caps(), "test/repo")

        model.generate("hello")

        assertTrue(session.configs.single().applyChatTemplate)
    }

    /** The flag has to survive the public→internal mapping, or the guard never reaches the generator. */
    @Test
    fun theFlagSurvivesTheConfigMapping() {
        assertFalse(GenerationConfig(applyChatTemplate = false).toOrt().applyChatTemplate)
        assertTrue(GenerationConfig().toOrt().applyChatTemplate)
    }

    /** `overrideConfig` merges field-by-field; an explicit false must not be lost to the default. */
    @Test
    fun overrideConfigCarriesAnExplicitSuppression() {
        val base = ORTGenerationConfig()
        assertFalse(base.overrideConfig(ORTGenerationConfig(applyChatTemplate = false)).applyChatTemplate)
        // And an override that says nothing leaves the base alone.
        assertFalse(
            ORTGenerationConfig(applyChatTemplate = false).overrideConfig(ORTGenerationConfig()).applyChatTemplate,
        )
    }
}

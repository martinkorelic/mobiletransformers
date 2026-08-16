package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.agent.FunctionCallValidator
import com.martinkorelic.mobiletransformers.agent.RejectedCallException
import com.martinkorelic.mobiletransformers.agent.ToolCallResult
import com.martinkorelic.mobiletransformers.agent.ToolCallParser
import com.martinkorelic.mobiletransformers.agent.ToolPromptBuilder
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
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.PromptAssembler
import com.martinkorelic.mobiletransformers.rag.PromptStrategy
import com.martinkorelic.mobiletransformers.runtime.ClassificationResult
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.GroundedResult
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.IngestResult
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.PushResult
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import com.martinkorelic.mobiletransformers.training.TrainingJob

/**
 * The stable public model handle (#17, extended by #19). Every method delegates to the [ModelSession]; no
 * engine logic and no `ORT*`/`*Native`/`Job`/repository type appears in this class's surface. Obtained
 * from [MobileTransformers.fromPretrained].
 */
class MobileTransformerModel internal constructor(
    private val session: ModelSession,
    val capabilities: RuntimeCapabilities,
    val repoId: String,
) {
    /** The engine resolved for this handle (Native floor or GenAI). */
    val engine: InferenceEngine get() = capabilities.engine

    /** Features actually installed in this package. */
    val installedFeatures: Set<ModelFeature> get() = capabilities.availableFeatures

    /** #19: select/validate the PEFT method against the installed package (no native call, no download). */
    suspend fun applyPeft(peft: PeftConfig) = session.applyPeft(peft)

    /**
     * One-shot training: suspends until the run finishes and returns the result.
     *
     * For status/event flows, cooperative cancellation or resume, use [trainingJob].
     */
    suspend fun train(
        dataset: DatasetConfig,
        config: TrainConfig = TrainConfig(),
        callback: TrainCallback? = null,
    ): TrainingResult = session.train(dataset, config, callback)

    /**
     * The lifecycle-shaped training handle for this model (#18): `status`/`events` flows, cooperative
     * `cancel(saveCheckpoint)`, and `checkpoint()`/`canResume`.
     *
     * The whole `training/` package was unreachable before this accessor existed — in particular there
     * was no way to cancel a run from the public API.
     */
    fun trainingJob(): TrainingJob = session.trainingJob()

    suspend fun merge(): MergeResult = session.merge()

    suspend fun generate(
        prompt: String,
        config: GenerationConfig = GenerationConfig(),
        callback: GenerateCallback? = null,
    ): GenerationResult = session.generate(prompt, config, callback)

    suspend fun retrieve(
        query: String,
        config: RagConfig = RagConfig(),
        callback: RetrieveCallback? = null,
    ): RetrievalResult = session.retrieve(query, config, callback)

    /** #26: ingest a `.txt`/`.md`/`.jsonl` file into the RAG vector store. */
    suspend fun ingest(
        path: String,
        config: RagConfig = RagConfig(),
        progress: IngestionProgress? = null,
    ): IngestResult = session.ingest(path, config, progress)

    /**
     * #33: classify [text] with a sequence-classification package.
     *
     * The other half of encoder support. Fine-tuning a BERT-family classifier on device worked end to
     * end and the result could never be *asked* anything — this handle offered generate, retrieve,
     * ingest and train, every one of which assumes a decoder.
     *
     * Check [RuntimeCapabilities.supportsClassification] first. Asking a decoder to classify throws
     * rather than reading generation logits as class scores, which would be a confident wrong answer
     * rather than an error.
     *
     * ```kotlin
     * if (model.capabilities.supportsClassification) {
     *     val result = model.classify("this sentence is grammatical")
     *     show(result.best?.label, result.best?.score)
     * }
     * ```
     *
     * @param device where to run it; defaults to the same CPU floor as everything else.
     * @param topK how many labels to return in [ClassificationResult.top]. The full ranking is always
     *   in `scores`.
     */
    suspend fun classify(
        text: String,
        device: DeviceConfig = DeviceConfig(),
        topK: Int = 5,
    ): ClassificationResult = session.classify(text, device, topK)

    /** #27: grounded generation — retrieve → assemble prompt → generate. `result.prompt` is inspectable. */
    suspend fun generateWithRag(
        query: String,
        rag: RagConfig = RagConfig(),
        generation: GenerationConfig = GenerationConfig(),
        promptStrategy: PromptStrategy = PromptAssembler.DEFAULT,
    ): GroundedResult = session.generateWithRag(query, rag, generation, promptStrategy)

    /** #19 surface; throws `NotImplementedFeatureException` until the #22 adapter push-back lands. */
    suspend fun pushAdapter(hubConfig: HubConfig, repoId: String): PushResult =
        session.pushAdapter(hubConfig, repoId)

    /**
     * #37: generate a **validated tool call** for `instruction`, or a first-class refusal.
     *
     * This is the seam the tool-call feature was missing: the validator and the intent binder existed
     * and were tested, but nothing routed generated text into them, so raw output and the boundary that
     * judges it never met outside of unit tests.
     *
     * The safety contract holds by construction. [ToolCallResult.Accepted] is the only carrier of a
     * [com.martinkorelic.mobiletransformers.agent.ValidatedCall], only [FunctionCallValidator] can build
     * one, and `IntentBinder` accepts nothing else — so there is no path from model output to an Android
     * intent that skips the allowlist, and the reachable set of intents is fixed when `validator` is
     * constructed.
     *
     * Build `validator` from the action schema written beside the training set, so the boundary enforced
     * here is the same artifact the model was trained toward:
     *
     * ```kotlin
     * val validator = FunctionCallValidator.fromSchema(File(packageDir, "action_schema.json"))
     * when (val result = model.generateToolCall("wake me at 07:30", validator)) {
     *     is ToolCallResult.Accepted -> show(result.dryRun())   // willExecute = false
     *     is ToolCallResult.Rejected -> show("I can't do that: ${result.reason}")
     *     is ToolCallResult.NoCall   -> show(result.raw)        // it answered in words
     * }
     * ```
     *
     * **[ToolCallResult.NoCall] is why this is usable as the only chat entry point.** Declare the
     * allowlist on every turn and let the outcome say what happened: prose comes back as prose, a
     * call comes back validated. That removes the "am I asking for a tool call right now?" switch a
     * caller would otherwise have to put in front of the user, who has no way to know the answer
     * before seeing the reply.
     *
     * @param parser how to read a call out of the model's text. Defaults to the one suited to this
     *   package's base model — **FunctionGemma does not emit JSON**, so a fixed JSON reader rejected
     *   every well-formed call it made. A parser chooses *which candidate* to check and nothing else;
     *   every allowlist and rule check still runs, so it cannot admit an undeclared action.
     * @param declareTools prepend the allowlist as a tool declaration the model can read. Without it
     *   the model is asked to call one of a set of functions it was never shown, which only a model
     *   fine-tuned on this exact allowlist can do. Turn it off when the caller has already framed the
     *   prompt itself.
     */
    suspend fun generateToolCall(
        instruction: String,
        validator: FunctionCallValidator,
        config: GenerationConfig = GenerationConfig(),
        parser: ToolCallParser = ToolCallParser.forDialect(capabilities.toolCalling.dialect),
        declareTools: Boolean = true,
        callback: GenerateCallback? = null,
    ): ToolCallResult {
        val prompt = if (declareTools) {
            // The whole turn, not declarations glued to an instruction: FunctionGemma emits its call
            // grammar inside a model turn, and nothing else on the device supplies that framing.
            ToolPromptBuilder.prompt(validator.allowlist, parser, instruction)
        } else {
            instruction
        }
        // Suppress the tokenizer's chat template only when the builder above emitted turns of its own
        // (the FunctionGemma dialect), or the two framings nest. The JSON dialect emits no turn
        // markers, so there the template is still what supplies them and must be left alone.
        // Harmless before the tokenizer learned to read `chat_template.jinja` — no package had a
        // template to apply — and a live distinction now that they do.
        val framedHere = declareTools && ToolPromptBuilder.framesOwnTurns(parser)
        val raw = session.generate(
            prompt,
            if (framedHere) config.copy(applyChatTemplate = false) else config,
            callback,
).text
        val call = parser.parse(raw)
            // Not a refusal: the model said something that is not a call. The raw text is the answer,
            // and reporting it as "rejected" both misleads the user and hides parser mismatches —
            // see ToolCallResult.NoCall.
            ?: return ToolCallResult.NoCall(raw = raw)
        return try {
            ToolCallResult.Accepted(raw = raw, call = validator.validate(call))
        } catch (e: RejectedCallException) {
            // Refusal is the expected answer for untrusted output, so it is a value, not a throw.
            ToolCallResult.Rejected(raw = raw, reason = e.message ?: "rejected")
        }
    }

    /**
     * #35/#36: run **one** federated round on this device — import the cohort's global adapter, train
     * locally under [localTraining]'s bounds, and export this device's update.
     *
     * Nothing is uploaded here: the round returns bytes and accepts bytes, so the transport (HTTPS to
     * `federated serve`, or `adb` in a device test) stays the caller's choice. `config` is checked
     * first and fails closed naming the missing protection — consent, TLS, auth, or the default-off
     * `BuildConfig.FEDERATION_ENABLED` — before any tensor is read.
     *
     * ```kotlin
     * val result = model.federatedRound(
     *     config = FederatedConfig(gatewayUrl = "https://…", clientAuthToken = token,
     *                              consent = FederatedConsent.GRANTED),
     *     globalRecord = previousAggregate,   // null for round 0
     *     roundNumber = 1,
     *     localTraining = { round -> model.train(dataset, TrainConfig(maxSteps = 20)) },
     * )
     * upload(result.update)                   // result.payloadBytes is the #36 DoD measurement
     * ```
     */
    suspend fun federatedRound(
        config: FederatedConfig,
        globalRecord: ByteArray?,
        roundNumber: Int,
        localTraining: LocalRoundTraining,
        metrics: Map<String, Double> = emptyMap(),
        train: Boolean = true,
    ): FederatedRoundResult =
        session.federatedRound(config, globalRecord, roundNumber, localTraining, metrics, train)

    fun close() = session.close()
}

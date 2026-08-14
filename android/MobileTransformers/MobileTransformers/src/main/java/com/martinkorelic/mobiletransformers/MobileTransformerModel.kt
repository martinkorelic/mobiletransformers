package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.agent.FunctionCallValidator
import com.martinkorelic.mobiletransformers.agent.RejectedCallException
import com.martinkorelic.mobiletransformers.agent.ToolCallResult
import com.martinkorelic.mobiletransformers.agent.extractFirstJsonObject
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.PromptAssembler
import com.martinkorelic.mobiletransformers.rag.PromptStrategy
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
     * }
     * ```
     *
     * @param extractJson pull the first balanced JSON object out of surrounding prose before validating.
     *   Chooses *which substring* to validate and nothing else — every allowlist and rule check still
     *   runs — so it cannot admit an action the app did not declare. Turn it off to require that the
     *   model emit bare JSON and nothing else.
     */
    suspend fun generateToolCall(
        instruction: String,
        validator: FunctionCallValidator,
        config: GenerationConfig = GenerationConfig(),
        extractJson: Boolean = true,
        callback: GenerateCallback? = null,
    ): ToolCallResult {
        val raw = session.generate(instruction, config, callback).text
        val candidate = if (extractJson) extractFirstJsonObject(raw) else raw
        return try {
            ToolCallResult.Accepted(raw = raw, call = validator.validate(candidate))
        } catch (e: RejectedCallException) {
            // Refusal is the expected answer for untrusted output, so it is a value, not a throw.
            ToolCallResult.Rejected(raw = raw, reason = e.message ?: "rejected")
        }
    }

    fun close() = session.close()
}

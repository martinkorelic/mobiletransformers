package com.martinkorelic.mobiletransformers.runtime

import com.martinkorelic.mobiletransformers.GenerateCallback
import com.martinkorelic.mobiletransformers.RetrieveCallback
import com.martinkorelic.mobiletransformers.TrainCallback
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.PromptStrategy
import com.martinkorelic.mobiletransformers.training.TrainingJob

/**
 * The internal whole-model contract the facade delegates to (#17, extended by #19). This is
 * **`ModelSession`**, NOT #11's engine-level `ModelRuntime` (`load/generate/release`) — this is the
 * facade-level `applyPeft/train/merge/generate/retrieve` surface. `RepositoryBackedModelSession` is the
 * only implementation; it never talks to a concrete engine, delegating generation to whichever engine
 * #11's `ModelRuntimeFactory` selected.
 */
interface ModelSession {
    val capabilities: RuntimeCapabilities

    /** #19: select/validate the PEFT method against what the installed package supports. No native call. */
    suspend fun applyPeft(peft: PeftConfig)

    suspend fun train(dataset: DatasetConfig, config: TrainConfig, callback: TrainCallback? = null): TrainingResult

    /**
     * The lifecycle-shaped training handle for this model (#18): `status`/`events` flows, cooperative
     * `cancel`, and `checkpoint`/`canResume`.
     *
     * [train] remains the one-shot convenience. Without this accessor the whole `training/` package
     * (`TrainingJob`, `TrainingJobManager`, `TrainingStatus`, `TrainingEvent`, `TrainingEventAdapter`)
     * was unreachable from the public API — it had zero non-test callers — and there was no way to
     * cancel a run at all.
     */
    fun trainingJob(): TrainingJob

    suspend fun merge(): MergeResult

    suspend fun generate(prompt: String, config: GenerationConfig, callback: GenerateCallback? = null): GenerationResult

    suspend fun retrieve(query: String, config: RagConfig, callback: RetrieveCallback? = null): RetrievalResult

    /** #26: ingest a `.txt`/`.md`/`.jsonl` file into the RAG vector store (chunk → embed → store). */
    suspend fun ingest(path: String, config: RagConfig, progress: IngestionProgress? = null): IngestResult

    /** #27: retrieve → assemble prompt → generate; the assembled prompt is returned for inspection. */
    suspend fun generateWithRag(
        query: String,
        rag: RagConfig,
        generation: GenerationConfig,
        promptStrategy: PromptStrategy,
    ): GroundedResult

    /** #19 surface; throws `NotImplementedFeatureException` until the #22 adapter push-back lands. */
    suspend fun pushAdapter(hubConfig: HubConfig, repoId: String): PushResult

    fun close()
}

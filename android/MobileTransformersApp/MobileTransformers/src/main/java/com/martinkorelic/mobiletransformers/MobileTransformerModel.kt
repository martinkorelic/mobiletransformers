package com.martinkorelic.mobiletransformers

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

    suspend fun train(
        dataset: DatasetConfig,
        config: TrainConfig = TrainConfig(),
        callback: TrainCallback? = null,
    ): TrainingResult = session.train(dataset, config, callback)

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

    fun close() = session.close()
}

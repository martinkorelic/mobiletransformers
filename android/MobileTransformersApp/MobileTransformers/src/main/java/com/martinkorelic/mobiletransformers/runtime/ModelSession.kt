package com.martinkorelic.mobiletransformers.runtime

import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig

/**
 * The internal whole-model contract the facade delegates to (#17). This is **`ModelSession`**, NOT #11's
 * engine-level `ModelRuntime` (`load/generate/release`) — this is the facade-level `train/merge/generate/
 * retrieve` surface. `RepositoryBackedModelSession` is the only implementation this pass; it never talks to
 * a concrete engine, delegating generation to whichever engine #11's `ModelRuntimeFactory` selected.
 */
interface ModelSession {
    val capabilities: RuntimeCapabilities

    suspend fun train(dataset: DatasetConfig, config: TrainConfig): TrainingResult

    suspend fun merge(): MergeResult

    suspend fun generate(prompt: String, config: GenerationConfig): GenerationResult

    suspend fun retrieve(query: String, config: RagConfig): RetrievalResult

    fun close()
}

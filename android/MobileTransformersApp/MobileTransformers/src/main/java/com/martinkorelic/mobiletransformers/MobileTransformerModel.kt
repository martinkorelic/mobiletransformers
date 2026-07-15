package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult

/**
 * The stable public model handle (#17). Every method delegates to the [ModelSession]; no engine logic and
 * no `ORT*`/`*Native`/`Job` type appears in this class's surface. Obtained from
 * [MobileTransformers.fromPretrained].
 */
class MobileTransformerModel internal constructor(
    private val session: ModelSession,
    val capabilities: RuntimeCapabilities,
) {
    suspend fun train(dataset: DatasetConfig, config: TrainConfig = TrainConfig()): TrainingResult =
        session.train(dataset, config)

    suspend fun merge(): MergeResult = session.merge()

    suspend fun generate(prompt: String, config: GenerationConfig = GenerationConfig()): GenerationResult =
        session.generate(prompt, config)

    suspend fun retrieve(query: String, config: RagConfig = RagConfig()): RetrievalResult =
        session.retrieve(query, config)

    fun close() = session.close()
}

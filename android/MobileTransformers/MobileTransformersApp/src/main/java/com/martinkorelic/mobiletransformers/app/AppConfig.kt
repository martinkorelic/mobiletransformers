package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * The knobs the Configuration screen edits and every other screen consumes.
 *
 * The old app kept ~45 fields spread over `ORTGenerationConfig`, `ORTTrainingConfig`, `ORTRagConfig`,
 * `SamplingOptions`, `DeviceOptions` and `SchedulerConfig`. This holds the **public** equivalents and
 * nothing else, which is the whole point of the rewrite: if a setting the old app could express is not
 * reachable through `GenerationConfig`/`TrainConfig`/`RagConfig`/`DatasetConfig`, that is a facade gap
 * to record against #17/#19 — not a reason to reach for an `ORT*` type.
 */
object AppConfig {
    private val _generation = MutableStateFlow(GenerationConfig())
    val generation: StateFlow<GenerationConfig> = _generation.asStateFlow()

    private val _train = MutableStateFlow(TrainConfig())
    val train: StateFlow<TrainConfig> = _train.asStateFlow()

    private val _rag = MutableStateFlow(RagConfig())
    val rag: StateFlow<RagConfig> = _rag.asStateFlow()

    private val _dataset = MutableStateFlow(DatasetConfig())
    val dataset: StateFlow<DatasetConfig> = _dataset.asStateFlow()

    fun updateGeneration(block: (GenerationConfig) -> GenerationConfig) {
        _generation.value = block(_generation.value)
    }

    fun updateTrain(block: (TrainConfig) -> TrainConfig) {
        _train.value = block(_train.value)
    }

    fun updateRag(block: (RagConfig) -> RagConfig) {
        _rag.value = block(_rag.value)
    }

    fun updateDataset(block: (DatasetConfig) -> DatasetConfig) {
        _dataset.value = block(_dataset.value)
    }

    /** Restore every section to the SDK's own defaults. */
    fun reset() {
        _generation.value = GenerationConfig()
        _train.value = TrainConfig()
        _rag.value = RagConfig()
        _dataset.value = DatasetConfig()
    }
}

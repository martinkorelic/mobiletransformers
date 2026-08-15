package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
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

    /**
     * Execution-provider and memory settings, applied to every config that carries them.
     *
     * `DeviceConfig` is a field on `GenerationConfig`, `TrainConfig` **and** `RagConfig`, and no
     * screen edited any of the three — so the execution provider, core profile and memory profile
     * were part of the public surface and completely unreachable from the app that exists to
     * demonstrate it. Held once and fanned out on write, because "run inference on XNNPACK but train
     * on CPU" is not a distinction a showcase should invite by accident; a caller that genuinely
     * wants it sets the field per config, which the SDK still allows.
     */
    private val _device = MutableStateFlow(DeviceConfig())
    val device: StateFlow<DeviceConfig> = _device.asStateFlow()

    /**
     * The PEFT method to apply before the next training run.
     *
     * `MobileTransformerModel.applyPeft` validates a selection against what the installed package
     * supports, and nothing in the app ever called it — so the one API that reports a PEFT mismatch
     * before a run rather than during it had no worked example.
     */
    private val _peft = MutableStateFlow<PeftConfig>(PeftConfig.Lora())
    val peft: StateFlow<PeftConfig> = _peft.asStateFlow()

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

    /** Set the device options and fan them out to every config that carries a [DeviceConfig]. */
    fun updateDevice(block: (DeviceConfig) -> DeviceConfig) {
        val next = block(_device.value)
        _device.value = next
        _generation.value = _generation.value.copy(device = next)
        _train.value = _train.value.copy(device = next)
        _rag.value = _rag.value.copy(device = next)
    }

    fun updatePeft(value: PeftConfig) {
        _peft.value = value
    }

    /** Restore every section to the SDK's own defaults. */
    fun reset() {
        _generation.value = GenerationConfig()
        _train.value = TrainConfig()
        _rag.value = RagConfig()
        _dataset.value = DatasetConfig()
        _device.value = DeviceConfig()
        _peft.value = PeftConfig.Lora()
    }
}

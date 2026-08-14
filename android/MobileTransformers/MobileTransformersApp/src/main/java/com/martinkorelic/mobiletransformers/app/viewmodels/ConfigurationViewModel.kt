package com.martinkorelic.mobiletransformers.app.viewmodels

import androidx.lifecycle.ViewModel
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.constants.SearchType
import kotlinx.coroutines.flow.StateFlow

/**
 * The ~45 knobs, expressed through the **public** config types only.
 *
 * The old Configuration screen was 1,091 lines editing `ORTGenerationConfig`, `ORTTrainingConfig`,
 * `ORTRagConfig`, `SamplingOptions`, `DeviceOptions` and `SchedulerConfig` directly. Everything it
 * could express is reachable here through `GenerationConfig`/`TrainConfig`/`RagConfig`/`DatasetConfig`
 * — which is the check this screen exists to perform. A knob that turned out to be unreachable would
 * be a facade gap to record against #17/#19, not a licence to import an `ORT*` type.
 */
class ConfigurationViewModel : ViewModel() {

    val generation: StateFlow<GenerationConfig> = AppConfig.generation
    val train: StateFlow<TrainConfig> = AppConfig.train
    val rag: StateFlow<RagConfig> = AppConfig.rag
    val dataset: StateFlow<DatasetConfig> = AppConfig.dataset

    // --- generation ---------------------------------------------------------------------------
    fun setMaxNewTokens(v: Int) = AppConfig.updateGeneration { it.copy(maxNewTokens = v.coerceAtLeast(1)) }

    fun setSamplingMethod(v: SamplingMethod) =
        AppConfig.updateGeneration { it.copy(sampling = it.sampling.copy(method = v)) }

    fun setTemperature(v: Float) =
        AppConfig.updateGeneration { it.copy(sampling = it.sampling.copy(temperature = v)) }

    fun setTopK(v: Int) = AppConfig.updateGeneration { it.copy(sampling = it.sampling.copy(topK = v)) }

    fun setTopP(v: Float) = AppConfig.updateGeneration { it.copy(sampling = it.sampling.copy(topP = v)) }

    fun setSeed(v: Int) = AppConfig.updateGeneration { it.copy(sampling = it.sampling.copy(seed = v)) }

    fun setSystemPrompt(v: String) =
        AppConfig.updateGeneration { it.copy(systemPrompt = v.ifBlank { null }) }

    fun setLoadMerged(v: Boolean) = AppConfig.updateGeneration { it.copy(loadMerged = v) }

    // --- training -----------------------------------------------------------------------------
    fun setEpochs(v: Int) = AppConfig.updateTrain { it.copy(epochs = v.coerceAtLeast(1)) }

    fun setBatchSize(v: Int) = AppConfig.updateTrain { it.copy(batchSize = v.coerceAtLeast(1)) }

    /**
     * `maxSteps` is an **upper bound**, not a target: training also stops at the end of the epoch, so
     * `rows / batchSize` wins when it is smaller. Measured the hard way on 2026-08-14 — a run asking
     * for 120 steps took 54 because the dataset held 108 rows.
     */
    fun setMaxSteps(v: Int?) = AppConfig.updateTrain { it.copy(maxSteps = v) }

    /**
     * The default is 4, and `optimizerStep` fires on `globalStep % gradAccumSteps == 0` — so a short
     * bounded run at the default can complete, report success on every callback, and apply **no
     * update at all**. Worth surfacing rather than burying.
     */
    fun setGradientAccumulationSteps(v: Int) =
        AppConfig.updateTrain { it.copy(gradientAccumulationSteps = v.coerceAtLeast(1)) }

    fun setLearningRate(v: Float) = AppConfig.updateTrain { it.copy(learningRate = v) }

    fun setScheduler(v: SchedulerType) = AppConfig.updateTrain { it.copy(scheduler = v) }

    fun setWarmupSteps(v: Int) = AppConfig.updateTrain { it.copy(warmupSteps = v.coerceAtLeast(0)) }

    fun setMergeAtEnd(v: Boolean) = AppConfig.updateTrain { it.copy(mergeAtEnd = v) }

    fun setResumeFromState(v: Boolean) = AppConfig.updateTrain { it.copy(resumeFromState = v) }

    // --- rag ----------------------------------------------------------------------------------
    fun setTopKRag(v: Int) = AppConfig.updateRag { it.copy(topK = v.coerceAtLeast(1)) }

    fun setSearchType(v: SearchType) = AppConfig.updateRag { it.copy(searchType = v) }

    fun setMinScore(v: Double) = AppConfig.updateRag { it.copy(minScore = v) }

    fun setChunkSize(v: Int) = AppConfig.updateRag { it.copy(chunkSize = v.coerceAtLeast(1)) }

    fun setChunkOverlap(v: Int) = AppConfig.updateRag { it.copy(chunkOverlap = v.coerceAtLeast(0)) }

    // --- dataset ------------------------------------------------------------------------------
    fun setTrainFile(v: String) = AppConfig.updateDataset { it.copy(trainFile = v) }

    fun setTask(v: String) = AppConfig.updateDataset { it.copy(task = v.ifBlank { null }) }

    fun setMaxSequenceLength(v: Int) =
        AppConfig.updateDataset { it.copy(maxSequenceLength = v.coerceAtLeast(1)) }

    fun setMaxDatasetLength(v: Int) =
        AppConfig.updateDataset { it.copy(maxDatasetLength = v.coerceAtLeast(1)) }

    fun reset() = AppConfig.reset()
}

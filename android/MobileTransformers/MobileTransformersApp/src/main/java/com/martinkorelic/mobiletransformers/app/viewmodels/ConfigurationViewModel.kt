package com.martinkorelic.mobiletransformers.app.viewmodels

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.Tasks
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.AppSnackbar
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.CoreConfigId
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.constants.MemoryConfigId
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.constants.SearchType
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

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
    val device: StateFlow<DeviceConfig> = AppConfig.device
    val peft: StateFlow<PeftConfig> = AppConfig.peft

    val modelState: StateFlow<ModelState> = ModelHolder.state

    /** The task names the trainer actually dispatches on — not a list retyped into the UI. */
    val taskOptions: List<Tasks.Task> = Tasks.TASKS

    /**
     * The `.jsonl` files present in the loaded package's `train/` stage.
     *
     * `DatasetConfig.trainFile` names a file the trainer opens at
     * `<cacheDir>/<repoId>/train/<trainFile>.jsonl`, and the field was free text — so the only way to
     * discover a wrong name was a failed run. Listing what is actually there turns it into a choice,
     * and shows an empty list when the answer is "you have not installed a dataset yet", which is the
     * real diagnosis in the common case.
     */
    fun availableTrainFiles(context: Context): List<String> {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return emptyList()
        val trainDir = PackagePaths.forCache(
            context.filesDir,
            PackageFormat.sanitizeRepoId(model.repoId),
        ).train
        if (!trainDir.isDirectory) return emptyList()
        return trainDir.listFiles()
            .orEmpty()
            .filter { it.isFile && it.name.endsWith(".jsonl") }
            .map { it.name.removeSuffix(".jsonl") }
            .sorted()
    }

    // --- device -------------------------------------------------------------------------------
    fun setExecutionProvider(v: ExecutionProvider) = AppConfig.updateDevice { it.copy(executionProvider = v) }

    fun setCoreConfig(v: CoreConfigId) = AppConfig.updateDevice { it.copy(coreConfigId = v) }

    fun setMemoryConfig(v: MemoryConfigId) = AppConfig.updateDevice { it.copy(memoryConfigId = v) }

    fun setProfiling(v: Boolean) = AppConfig.updateDevice { it.copy(enableProfiling = v) }

    // --- peft ---------------------------------------------------------------------------------

    /**
     * Select a PEFT method and validate it against the installed package.
     *
     * `applyPeft` is the SDK's own check that the requested method matches what the package was
     * exported with — PEFT topology is baked in at export time, so a mismatch is a fact about the
     * package, not something the device can fix. Reporting it here, at selection, is the difference
     * between a clear refusal and a training run that fails for a reason recorded in a config file.
     */
    fun setPeft(v: PeftConfig) {
        AppConfig.updatePeft(v)
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        viewModelScope.launch {
            runCatching { model.applyPeft(v) }
                .onSuccess { AppSnackbar.success("PEFT set to ${v.label}") }
                .onFailure { AppSnackbar.error(it.message ?: "this package does not support ${v.label}") }
        }
    }

    fun setPeftRank(rank: Int) = setPeft(AppConfig.peft.value.withRank(rank))

    fun setPeftAlpha(alpha: Int) = setPeft(AppConfig.peft.value.withAlpha(alpha))

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

/**
 * The PEFT methods as a pickable list.
 *
 * `PeftConfig` is a sealed class with per-variant fields, which is right for the API and awkward for
 * a picker. This flattens it to the choice a user actually makes — which method — while preserving
 * the current rank and alpha across a switch, so changing method does not silently reset them.
 */
val peftOptions: List<String> = listOf("lora", "mars-opt0", "mars-opt1", "mars-quantized")

val PeftConfig.label: String
    get() = when (this) {
        is PeftConfig.Lora -> "lora"
        is PeftConfig.MarsOpt0 -> "mars-opt0"
        is PeftConfig.MarsOpt1 -> "mars-opt1"
        is PeftConfig.MarsQuantized -> "mars-quantized"
    }

/** Build the named method, carrying [rank]/[alpha] over so a method switch is not also a reset. */
fun peftOf(label: String, rank: Int, alpha: Int): PeftConfig = when (label) {
    "mars-opt0" -> PeftConfig.MarsOpt0(rank = rank, alpha = alpha)
    "mars-opt1" -> PeftConfig.MarsOpt1(rank = rank, alpha = alpha)
    "mars-quantized" -> PeftConfig.MarsQuantized(rank = rank, alpha = alpha)
    else -> PeftConfig.Lora(rank = rank, alpha = alpha)
}

fun PeftConfig.withRank(rank: Int): PeftConfig = peftOf(label, rank, alpha)

fun PeftConfig.withAlpha(alpha: Int): PeftConfig = peftOf(label, rank, alpha)

/** What each method costs and requires, shown under the picker. */
fun peftDescription(label: String): String = when (label) {
    "lora" -> "low-rank adapters on the attention projections; the default and the widest support"
    "mars-opt0" -> "MARS, fully trainable, no quantization"
    "mars-opt1" -> "MARS, partially trainable (frozen + fused down-proj), no quantization"
    "mars-quantized" -> "MARS with 8- or 4-bit weights; smallest memory, narrowest package support"
    else -> ""
}

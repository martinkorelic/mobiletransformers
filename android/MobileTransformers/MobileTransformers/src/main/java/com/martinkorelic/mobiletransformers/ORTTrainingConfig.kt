package com.martinkorelic.mobiletransformers

// Scheduler configuration classes
sealed class SchedulerConfig {
    data class Linear(
        val learningRate: Float = 1e-4f,
        val startFactor: Float = 1.0f,
        val endFactor: Float = 0.333f
    ) : SchedulerConfig()

    data class Cosine(
        val learningRate: Float = 1e-4f,
        val minLearningRate: Float = 0f,
        val warmupSteps: Int = 10
    ) : SchedulerConfig()
}

data class DatasetOptions(
    val trainFile: String = "arc_e",
    val datasetBatchSize: Int? = 64,
    val maxSequenceLength: Int? = 512,
    val maxDatasetLength: Int? = 256,
    val removeLongSamples: Boolean = false,
    val datasetSplit: Boolean = false,
    val datasetShuffle: Boolean = false,
    val testRatio: Float = 0.0f
)

data class DeviceOptions (
    val enableProfiling: Boolean = false,
    val coreConfigId: String = "opt1",
    val memoryConfigId: String = "high_perf",
    val executionProvider : String = "cpu"
)

data class ORTTrainingConfig(
    val repoName : String = "model",
    val onnxName : String = ".onnx",
    val taskName : String = "none",

    val batchSize: Int = 4,
    val numTrainEpochs: Int = 1,
    val maxSteps: Int? = 10,
    val saveSteps: Int = 100,
    val gradAccumSteps: Int = 4,

    val mergeWeightsAtEnd : Boolean = true,
    val saveModelAtEnd : Boolean = true,

    /**
     * Keep the native training session OPEN when the run finishes (#36).
     *
     * `startTraining` releases the session on every exit path — with the checkpoint saved when
     * [saveModelAtEnd], without it otherwise. That is why every post-training step in this library
     * (the merge, the checkpoint cadence) happens *inside* `startTraining`: afterwards there is no
     * session left to act on. A federated round has to read the checkpoint it just trained, and
     * re-opening a session to do so would reload ~176 MB to look at the ~2 MB of adapter factors it
     * had in memory a moment earlier.
     *
     * Default `false`, so every existing caller keeps the release-at-end behaviour it was written
     * against. When true, the caller owns the session and MUST call [destroySession] itself.
     */
    val keepSessionAtEnd : Boolean = false,
    val loadFromState : Boolean = true,
    val profileMetrics : Boolean = false,

    val datasetOptions: DatasetOptions = DatasetOptions(),

    val schedulerType: String = "linear", // Options: "linear", "cosine"
    val schedulerConfig: SchedulerConfig = SchedulerConfig.Linear(),

    val deviceOptions: DeviceOptions = DeviceOptions(),

    var customPreprocess: TaskPreprocessor? = null
) {

    val learningRate: Float
        get() = when (schedulerConfig) {
            is SchedulerConfig.Linear -> schedulerConfig.learningRate
            is SchedulerConfig.Cosine -> schedulerConfig.learningRate
        }

    val minLearningRate: Float
        get() = when (schedulerConfig) {
            is SchedulerConfig.Linear -> 0f
            is SchedulerConfig.Cosine -> schedulerConfig.minLearningRate
        }

    val warmupSteps: Int
        get() = when (schedulerConfig) {
            is SchedulerConfig.Linear -> 0
            is SchedulerConfig.Cosine -> schedulerConfig.warmupSteps
        }

    // TODO: Fix override config
    fun overrideConfig(override: ORTTrainingConfig?): ORTTrainingConfig {
        if (override == null) return this

        return this.copy(
            repoName = override.repoName,
            onnxName = override.onnxName,
            taskName = if (override.taskName != "none") override.taskName else this.taskName,
            batchSize = override.batchSize,
            numTrainEpochs = override.numTrainEpochs,
            maxSteps = override.maxSteps,
            saveSteps = override.saveSteps,
            gradAccumSteps = override.gradAccumSteps,
            schedulerType = override.schedulerType,
            schedulerConfig = override.schedulerConfig,
            mergeWeightsAtEnd = override.mergeWeightsAtEnd,
            loadFromState = override.loadFromState,
            profileMetrics = override.profileMetrics,
            deviceOptions = override.deviceOptions,
            datasetOptions = override.datasetOptions,
            customPreprocess = override.customPreprocess ?: this.customPreprocess
        )
    }
}

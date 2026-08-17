package com.martinkorelic.mobiletransformers

/**
 * Public training progress (#19), mapped 1:1 from the internal `TrainingProgress`.
 */
data class TrainProgress(
    val currentStep: Int,
    val currentEpoch: Int,
    val totalLoss: Float = 0f,
    val epochLoss: Float = 0f,
    val stepLoss: Float = 0f,
    val learningRate: Float = 0f,
    val stepDurationMs: Long = 0L,
    val epochDurationMs: Long = 0L,
    val totalDurationMs: Long = 0L,
    val isCompleted: Boolean = false,
)

/**
 * Public training lifecycle callback for [MobileTransformerModel.train] (#19). Mirrors the internal
 * `TrainingCallback` so app code never imports repository/`ORT*` types. All methods are optional.
 */
interface TrainCallback {
    fun onModelLoadStart() {}

    fun onModelLoadEnd() {}

    fun onDataLoadEnd(totalSteps: Int, stepsPerEpoch: Int) {}

    fun onStepEnd(progress: TrainProgress) {}

    fun onEpochEnd(progress: TrainProgress) {}

    fun onMergeStart(progress: TrainProgress) {}

    fun onMergeEnd(progress: TrainProgress) {}

    fun onCompletion(progress: TrainProgress) {}

    fun onError(error: Throwable) {}
}

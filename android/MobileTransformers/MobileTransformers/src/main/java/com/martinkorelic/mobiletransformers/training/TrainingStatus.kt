package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.runtime.TrainingResult

/**
 * Public, lifecycle-shaped training status (#18). Maps the `LLMState`/`TrainingCallback` surface without
 * leaking `ORTTrainerNative` or the native handle. [TrainingResult] is the SAME type #17's
 * `ModelSession.train` returns — #18 enriches it with `checkpoint`/`summary` (see `runtime/Results.kt`).
 */
sealed interface TrainingStatus {
    data object Idle : TrainingStatus

    data object Preparing : TrainingStatus // LLMState.ReadyTrain bring-up

    data class Running(val progress: TrainingProgress) : TrainingStatus // LLMState.Training

    data object Merging : TrainingStatus // onMergeStart..onMergeEnd

    data object Saving : TrainingStatus // LLMState.SavingModel

    data class Completed(val result: TrainingResult) : TrainingStatus

    data class Cancelled(val checkpoint: CheckpointInfo?) : TrainingStatus

    data class Failed(val error: Throwable) : TrainingStatus
}

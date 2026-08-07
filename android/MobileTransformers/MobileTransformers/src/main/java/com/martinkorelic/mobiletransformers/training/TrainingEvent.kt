package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.ORTTrainerNative
import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.runtime.TrainingResult

/**
 * Structured training event stream (#18), one-to-one with the existing `TrainingCallback` surface. Emitted
 * on a `SharedFlow<TrainingEvent>` by [TrainingEventAdapter].
 */
sealed interface TrainingEvent {
    data class DataLoaded(val totalSteps: Int, val stepsPerEpoch: Int) : TrainingEvent

    data class Step(val progress: TrainingProgress) : TrainingEvent // onStepEnd

    data class OptimizerStep(val progress: TrainingProgress) : TrainingEvent

    data class Epoch(val progress: TrainingProgress) : TrainingEvent // onEpochEnd

    data class Metric(val m: ORTTrainerNative.TrainingStepMetrics) : TrainingEvent

    data object MergeStarted : TrainingEvent

    data object MergeFinished : TrainingEvent

    data class Saved(val progress: TrainingProgress) : TrainingEvent

    data class Done(val result: TrainingResult) : TrainingEvent

    data class Error(val t: Throwable) : TrainingEvent
}

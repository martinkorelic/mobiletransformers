package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.ORTTrainerNative
import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.repository.TrainingCallback
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Translates the existing `TrainingCallback` surface (#18) into the public [TrainingStatus] state and the
 * [TrainingEvent] stream, one-to-one. Pure of any native handle — it only reacts to callbacks, so it is
 * unit-testable by driving a scripted callback sequence. The `checkpointSupplier`/`summarySupplier` seams let
 * [TrainingJob] inject the real `training_state.json`/`training_logs.json` reads (defaults return null).
 */
class TrainingEventAdapter(
    private val checkpointSupplier: () -> CheckpointInfo? = { null },
    // The PUBLIC summary type (#17) — the ORT-side value is converted via `toPublic()`.
    private val summarySupplier: () -> com.martinkorelic.mobiletransformers.runtime.TrainingSummary? = { null },
) : TrainingCallback {

    private val _status = MutableStateFlow<TrainingStatus>(TrainingStatus.Idle)
    val status: StateFlow<TrainingStatus> = _status.asStateFlow()

    private val _events = MutableSharedFlow<TrainingEvent>(replay = 0, extraBufferCapacity = 128)
    val events: SharedFlow<TrainingEvent> = _events.asSharedFlow()

    private var merged = false

    private fun emit(event: TrainingEvent) {
        _events.tryEmit(event)
    }

    override fun onModelLoadStart() {
        _status.value = TrainingStatus.Preparing
    }

    override fun onDataLoadEnd(totalSteps: Int, stepsPerEpoch: Int) {
        emit(TrainingEvent.DataLoaded(totalSteps, stepsPerEpoch))
    }

    override fun onStepEnd(trainingProgress: TrainingProgress) {
        _status.value = TrainingStatus.Running(trainingProgress)
        emit(TrainingEvent.Step(trainingProgress))
    }

    override fun onOptimizerStep(trainingProgress: TrainingProgress) {
        _status.value = TrainingStatus.Running(trainingProgress)
        emit(TrainingEvent.OptimizerStep(trainingProgress))
    }

    override fun onEpochEnd(trainingProgress: TrainingProgress) {
        _status.value = TrainingStatus.Running(trainingProgress)
        emit(TrainingEvent.Epoch(trainingProgress))
    }

    override fun onMergeStart(trainingProgress: TrainingProgress) {
        _status.value = TrainingStatus.Merging
        emit(TrainingEvent.MergeStarted)
    }

    override fun onMergeEnd(trainingProgress: TrainingProgress) {
        merged = true
        emit(TrainingEvent.MergeFinished)
    }

    override fun onSaveModelStart(trainingProgress: TrainingProgress) {
        _status.value = TrainingStatus.Saving
    }

    override fun onSaveModelEnd(trainingProgress: TrainingProgress) {
        emit(TrainingEvent.Saved(trainingProgress))
    }

    override fun onCompletion(trainingProgress: TrainingProgress) {
        val result =
            TrainingResult(
                finalStep = trainingProgress.currentStep,
                finalEpoch = trainingProgress.currentEpoch,
                finalLoss = trainingProgress.totalLoss,
                totalDurationMs = trainingProgress.totalDurationMs,
                merged = merged,
                checkpoint = checkpointSupplier(),
                summary = summarySupplier(),
            )
        _status.value = TrainingStatus.Completed(result)
        emit(TrainingEvent.Done(result))
    }

    override fun onError(error: Throwable) {
        _status.value = TrainingStatus.Failed(error)
        emit(TrainingEvent.Error(error))
    }

    /** Called by [TrainingJob.cancel] after the loop breaks and the checkpoint (if any) is persisted. */
    fun markCancelled(checkpoint: CheckpointInfo?) {
        _status.value = TrainingStatus.Cancelled(checkpoint)
    }
}

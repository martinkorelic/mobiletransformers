package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.TaskPreprocessor
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.repository.TrainingRepository
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Public, lifecycle-shaped training handle (#18). Wraps `LLMRepository`/`ORTTrainerNative` without leaking
 * them or the native handle. `start` drives the existing prepare→train path with a [TrainingEventAdapter] as
 * the callback; `cancel` sets the cooperative `ORTTrainerNative.cancelRequested` flag so the loop breaks and
 * the existing `saveModel`+`saveTrainingState` path can persist a checkpoint; `checkpoint`/`canResume` are
 * read-only projections of the unchanged `training_state.json` (see [CheckpointInfo]).
 */
class TrainingJob internal constructor(
    private val repo: LLMRepository,
    @Suppress("unused") private val repoId: String,
) {
    private val training = TrainingRepository(repo)

    private val adapter =
        TrainingEventAdapter(
            checkpointSupplier = { checkpoint() },
            // Real run summary when trainingConfig.profileMetrics collected one; null otherwise.
            summarySupplier = { repo.ortTrainerNative?.lastSummary?.toPublic() },
        )

    val status: StateFlow<TrainingStatus> = adapter.status
    val events: SharedFlow<TrainingEvent> = adapter.events

    /** Start (or resume, when [canResume]) training. Suspends until the run completes or fails. */
    suspend fun start(args: ORTTrainingConfig? = null, preprocess: TaskPreprocessor? = null) {
        repo.ortTrainerNative?.cancelRequested = false
        training.performTraining(args, adapter, preprocess)
    }

    /**
     * Cooperatively cancel: set the native cancel flag, let the loop break, then persist via the existing
     * save path if [saveCheckpoint]. Emits [TrainingStatus.Cancelled] with the persisted checkpoint.
     */
    suspend fun cancel(saveCheckpoint: Boolean = true) {
        repo.ortTrainerNative?.cancelRequested = true
        training.endTraining(saveCheckpoint)
        adapter.markCancelled(if (saveCheckpoint) checkpoint() else null)
    }

    /** Read the current `training_state.json` projection, or null if the trainer isn't initialized. */
    fun checkpoint(): CheckpointInfo? {
        val trainer = repo.ortTrainerNative ?: return null
        return CheckpointInfo.read(trainer.checkpointPath, "${trainer.checkpointPath.removeSuffix("checkpoint")}training_state.json")
    }

    /** True when a state file exists and `loadFromState` would restore it. */
    val canResume: Boolean
        get() = checkpoint()?.exists == true
}

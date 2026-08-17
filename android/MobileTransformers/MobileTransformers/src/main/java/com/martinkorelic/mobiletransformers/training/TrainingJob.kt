package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.TaskPreprocessor
import com.martinkorelic.mobiletransformers.Tasks
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.internal.config.toOrt
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
     * Start (or resume) training from the **public** config types.
     *
     * #17/#19 gap found building the showcase app's Train screen. `MobileTransformerModel.trainingJob()`
     * is the only way to reach `status`/`events`/`cancel`/`checkpoint`, but the sole way to *start* the
     * job it returns took an `ORTTrainingConfig` — an engine-layer type the public API otherwise never
     * mentions and a facade-only app is forbidden to import. So the lifecycle-shaped API was reachable
     * and unusable at the same time: an app could either have progress flows and cancellation (by
     * reaching around the facade) or stay on the facade and use the one-shot
     * [com.martinkorelic.mobiletransformers.MobileTransformerModel.train], never both.
     *
     * Maps through the same `ConfigMappers` the one-shot path uses, including the "caller supplies the
     * data, so the caller names its preprocessor" rule — so the two entry points cannot drift into
     * training differently from the same configs.
     */
    suspend fun start(dataset: DatasetConfig, config: TrainConfig = TrainConfig()) {
        val ortConfig = config.toOrt(repo.trainingConfig).copy(
            datasetOptions = dataset.toOrt(),
            // Checked here, in the caller's frame. Left to the trainer's constructor this throws
            // inside LLMRepository's own coroutine scope, where no caller `catch` can see it and the
            // process dies instead. See Tasks.resolve.
            taskName = Tasks.resolve(dataset.task, repo.trainingConfig.taskName),
        )
        // `DatasetConfig` names a registered task rather than carrying a lambda, so the preprocessor
        // is resolved by name inside the training path — the same rule scheduled training relies on
        // (a lambda cannot survive a worker rebuilt after process death).
        start(ortConfig, null)
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

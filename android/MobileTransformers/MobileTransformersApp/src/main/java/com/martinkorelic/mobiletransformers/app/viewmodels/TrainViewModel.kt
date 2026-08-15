package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.AppSnackbar
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.SampleData
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.scheduler.ScheduledChunk
import com.martinkorelic.mobiletransformers.scheduler.TrainingScheduleConfig
import com.martinkorelic.mobiletransformers.scheduler.TrainingScheduler
import com.martinkorelic.mobiletransformers.training.TrainingEvent
import com.martinkorelic.mobiletransformers.training.TrainingStatus
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * #18/#19/#34 — training driven through `trainingJob()`, plus #34's charging-cycle scheduler.
 *
 * Uses the **lifecycle** handle rather than the one-shot `train()`, because status/events/cancel/resume
 * are the half an app actually needs. That was only reachable by importing `ORTTrainingConfig` until
 * `TrainingJob.start(DatasetConfig, TrainConfig)` was added — recorded against #17/#19.
 */
class TrainViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(TrainUiState())
    val ui: StateFlow<TrainUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    private val _scheduledRuns = MutableStateFlow<List<ScheduledRun>>(emptyList())

    /**
     * The scheduled queue for the loaded model.
     *
     * `TrainingScheduler.observe` has existed since #34 and had **no caller**, so the only feedback a
     * user got from scheduling was a UUID in a text field. "Queued and waiting for the charger",
     * "running chunk 3" and "finished an hour ago" were indistinguishable — for a feature whose whole
     * point is that it runs when you are not watching.
     */
    val scheduledRuns: StateFlow<List<ScheduledRun>> = _scheduledRuns.asStateFlow()

    private var observeJob: Job? = null

    init {
        // Re-subscribe when the model changes: the unique work name is per repo id.
        viewModelScope.launch {
            ModelHolder.state.collect { state ->
                observeJob?.cancel()
                val repoId = (state as? ModelState.Loaded)?.model?.repoId
                if (repoId == null) {
                    _scheduledRuns.value = emptyList()
                    return@collect
                }
                observeJob = launch {
                    TrainingScheduler.observeChunks(getApplication(), repoId).collect { chunks ->
                        _scheduledRuns.value = chunks.map { it.toScheduledRun() }
                    }
                }
            }
        }
    }

    /** Cancel the queued run; the chunk in flight checkpoints on its way out. */
    fun cancelSchedule() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        runCatching { TrainingScheduler.cancel(getApplication(), loaded.model.repoId) }
            .onSuccess {
                _ui.value = _ui.value.copy(scheduled = null)
                AppSnackbar.info("Scheduled run cancelled")
            }
            .onFailure { AppSnackbar.error(it.message ?: "could not cancel the scheduled run") }
    }

    fun start() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        if (!loaded.model.capabilities.supportsTraining) {
            _ui.value = _ui.value.copy(
                error = "this package has no train/ stage — re-export with TRAIN=1 or pull a " +
                    "train-capable variant",
            )
            return
        }
        viewModelScope.launch {
            val job = loaded.model.trainingJob()
            _ui.value = _ui.value.copy(
                running = true,
                error = null,
                events = emptyList(),
                points = emptyList(),
            )
            AppSnackbar.info("Training started")

            // Observe before starting: a run short enough to finish first would otherwise report nothing.
            val statusJob = launch {
                job.status.collect { s -> _ui.value = _ui.value.copy(status = s.describe()) }
            }
            val eventJob = launch {
                job.events.collect { e ->
                    val ui = _ui.value
                    _ui.value = ui.copy(
                        events = (ui.events + e.describe()).takeLast(EVENT_LOG_LIMIT),
                        // Every Step event already carried loss, learning rate and step duration;
                        // rendering it with `toString()` threw all of it away.
                        points = e.point()?.let { ui.points + it } ?: ui.points,
                    )
                }
            }
            try {
                job.start(dataset = AppConfig.dataset.value, config = AppConfig.train.value)
                _ui.value = _ui.value.copy(canResume = job.canResume)
                val last = _ui.value.points.lastOrNull()
                AppSnackbar.success(
                    last?.let { "Training finished — loss %.4f at step %d".format(it.loss, it.step) }
                        ?: "Training finished",
                )
            } catch (e: Throwable) {
                val reason = e.message ?: e::class.java.simpleName
                _ui.value = _ui.value.copy(error = reason)
                AppSnackbar.error(reason)
            } finally {
                statusJob.cancel()
                eventJob.cancel()
                _ui.value = _ui.value.copy(running = false)
            }
        }
    }

    /**
     * Copy the bundled tool-call training set into the installed package and point [AppConfig] at it.
     *
     * Without this the Start button could not work on a freshly pulled package: the trainer reads
     * `<cacheDir>/<repoId>/train/<trainFile>.jsonl`, model packages deliberately ship no training
     * data, and the app had no way to create one. That made the whole Train screen unreachable for
     * anyone who had not `adb push`ed a dataset by hand.
     *
     * The cache root mirrors what `ModelHolder` lets `fromPretrained` default to (`context.filesDir`);
     * `sanitizeRepoId` is the public mapping from repo id to cache directory name.
     */
    fun installSampleDataset() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        val app = getApplication<Application>()
        runCatching {
            SampleData.installTrainingSet(
                context = app,
                // The cache root ModelHolder lets `fromPretrained` default to.
                cacheDir = app.filesDir,
                sanitizedRepoId = PackageFormat.sanitizeRepoId(loaded.model.repoId),
            )
        }
            .onSuccess { installed ->
                if (installed == null) {
                    _ui.value = _ui.value.copy(
                        error = "this package has no train/ stage, so there is nowhere to put a " +
                            "training set — pull one with the Training feature requested",
                    )
                } else {
                    AppConfig.updateDataset {
                        it.copy(trainFile = SampleData.TRAIN_FILE, task = SampleData.TRAIN_TASK)
                    }
                    _ui.value = _ui.value.copy(
                        error = null,
                        datasetNote = "installed ${installed.name} (${installed.length()} bytes) and " +
                            "set trainFile=${SampleData.TRAIN_FILE}, task=${SampleData.TRAIN_TASK}",
                    )
                }
            }
            .onFailure { _ui.value = _ui.value.copy(error = it.message ?: "could not install sample data") }
    }

    fun cancel() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        viewModelScope.launch {
            // Cooperative: the native loop breaks at the next step boundary and the checkpoint is
            // persisted, so cancelling is resumable rather than destructive.
            runCatching { loaded.model.trainingJob().cancel(saveCheckpoint = true) }
                .onFailure { _ui.value = _ui.value.copy(error = it.message) }
        }
    }

    /** #34: hand the same public configs to WorkManager instead of running now. */
    fun schedule() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        if (!loaded.model.capabilities.supportsScheduledTraining) {
            _ui.value = _ui.value.copy(error = "scheduled training needs a train-capable package")
            return
        }
        runCatching {
            TrainingScheduler.schedule(
                context = getApplication(),
                repoId = loaded.model.repoId,
                dataset = AppConfig.dataset.value,
                training = AppConfig.train.value,
                config = TrainingScheduleConfig(),
            )
        }.onSuccess {
            _ui.value = _ui.value.copy(
                scheduled = "queued as $it — chunks run only while charging and idle, and each " +
                    "chunk re-evaluates its constraints",
            )
            AppSnackbar.success("Scheduled — it will start when the device is charging and idle")
        }.onFailure {
            _ui.value = _ui.value.copy(error = it.message)
            AppSnackbar.error(it.message ?: "could not schedule training")
        }
    }

    fun merge() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        viewModelScope.launch {
            runCatching { loaded.model.merge() }
                .onSuccess {
                    _ui.value = _ui.value.copy(status = "merged=${it.merged}")
                    AppSnackbar.success(
                        if (it.merged) {
                            "Merged — generation now uses the fine-tuned weights"
                        } else {
                            "Nothing to merge: no adapter has been trained yet"
                        },
                    )
                }
                .onFailure {
                    _ui.value = _ui.value.copy(error = it.message)
                    AppSnackbar.error(it.message ?: "merge failed")
                }
        }
    }
}

/** Keep the log bounded: a long run emits thousands of steps and this is a phone. */
private const val EVENT_LOG_LIMIT = 300

data class TrainUiState(
    val running: Boolean = false,
    val status: String = "idle",
    val events: List<String> = emptyList(),
    /** The series behind the charts — one entry per reported step. */
    val points: List<StepPoint> = emptyList(),
    val canResume: Boolean = false,
    val scheduled: String? = null,
    val datasetNote: String? = null,
    val error: String? = null,
)

/** One scheduled chunk, as the queue panel renders it. */
data class ScheduledRun(val stateLabel: String, val detail: String)

/** Put the SDK's chunk state into the words the panel shows. */
private fun ScheduledChunk.toScheduledRun(): ScheduledRun {
    val label = when (state) {
        ScheduledChunk.State.WaitingForConstraints -> "Waiting for charging + idle"
        ScheduledChunk.State.Running -> "Running" + (chunk?.let { " chunk $it" } ?: "")
        ScheduledChunk.State.Finished -> "Chunk finished"
        ScheduledChunk.State.Failed -> "Failed"
        ScheduledChunk.State.Blocked -> "Blocked by an earlier chunk"
        ScheduledChunk.State.Cancelled -> "Cancelled"
    }
    val detail = buildString {
        globalStep?.let { append("globalStep $it") }
        if (stalled) {
            if (isNotEmpty()) append(" · ")
            append("advanced no steps, so no further chunk was queued")
        }
        error?.let {
            if (isNotEmpty()) append(" · ")
            append(it)
        }
        if (isEmpty()) {
            append(
                "Chunks run only while the device is charging and idle, and each one re-checks that " +
                    "before starting.",
            )
        }
    }
    return ScheduledRun(label, detail)
}

/** One reported training step, as the charts consume it. */
data class StepPoint(
    val step: Int,
    val loss: Float,
    val learningRate: Float,
    val stepDurationMs: Long,
)

/**
 * A human-readable status line rather than a Kotlin class name.
 *
 * `this::class.java.simpleName` rendered `Running` for a run and gave no indication of *where* it
 * was, even though `TrainingStatus.Running` carries the progress that answers exactly that.
 */
private fun TrainingStatus.describe(): String = when (this) {
    is TrainingStatus.Idle -> "idle"
    is TrainingStatus.Preparing -> "preparing — loading the training graph"
    is TrainingStatus.Running ->
        "step ${progress.currentStep} · epoch ${progress.currentEpoch} · loss %.4f".format(progress.stepLoss)
    is TrainingStatus.Merging -> "merging the adapter into the inference graph"
    is TrainingStatus.Saving -> "saving checkpoint"
    is TrainingStatus.Completed ->
        "completed — %d steps, final loss %.4f".format(result.finalStep, result.finalLoss)
    is TrainingStatus.Cancelled ->
        "cancelled" + (checkpoint?.let { " — checkpoint at step ${it.currentGlobalStep}" } ?: "")
    is TrainingStatus.Failed -> "failed: ${error.message ?: error::class.java.simpleName}"
}

/**
 * One aligned line per event.
 *
 * This used to be `toString()` on the event, which printed a whole Kotlin data class — including the
 * nested `TrainingProgress` with all ten of its fields — for every step. The log was technically
 * complete and practically unreadable, and it is the chart's table-view twin, so it has to be the
 * place a value can actually be read.
 */
private fun TrainingEvent.describe(): String = when (this) {
    is TrainingEvent.DataLoaded -> "data loaded · $totalSteps steps · $stepsPerEpoch per epoch"
    is TrainingEvent.Step ->
        "step %-5d loss %-9.4f lr %-10.2e %d ms".format(
            progress.currentStep, progress.stepLoss, progress.learningRate, progress.stepDurationMs,
        )
    is TrainingEvent.OptimizerStep -> "optimizer step at ${progress.currentStep}"
    is TrainingEvent.Epoch ->
        "epoch %d ended · epoch loss %.4f · %d ms".format(
            progress.currentEpoch, progress.epochLoss, progress.epochDurationMs,
        )
    is TrainingEvent.Metric -> "metric: $m"
    is TrainingEvent.MergeStarted -> "merge started"
    is TrainingEvent.MergeFinished -> "merge finished"
    is TrainingEvent.Saved -> "checkpoint saved at step ${progress.currentStep}"
    is TrainingEvent.Done ->
        "done · %d steps · final loss %.4f · %d ms".format(
            result.finalStep, result.finalLoss, result.totalDurationMs,
        )
    is TrainingEvent.Error -> "error: ${t.message ?: t::class.java.simpleName}"
}

/** The chart point a step event carries, or null for the events that are not steps. */
private fun TrainingEvent.point(): StepPoint? = when (this) {
    is TrainingEvent.Step -> StepPoint(
        step = progress.currentStep,
        loss = progress.stepLoss,
        learningRate = progress.learningRate,
        stepDurationMs = progress.stepDurationMs,
    )
    else -> null
}

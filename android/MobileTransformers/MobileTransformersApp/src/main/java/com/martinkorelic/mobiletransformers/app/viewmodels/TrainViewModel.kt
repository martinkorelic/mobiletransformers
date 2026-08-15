package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.SampleData
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.scheduler.TrainingScheduleConfig
import com.martinkorelic.mobiletransformers.scheduler.TrainingScheduler
import com.martinkorelic.mobiletransformers.training.TrainingEvent
import com.martinkorelic.mobiletransformers.training.TrainingStatus
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
            _ui.value = _ui.value.copy(running = true, error = null, events = emptyList())

            // Observe before starting: a run short enough to finish first would otherwise report nothing.
            val statusJob = launch {
                job.status.collect { s -> _ui.value = _ui.value.copy(status = s.describe()) }
            }
            val eventJob = launch {
                job.events.collect { e ->
                    _ui.value = _ui.value.copy(events = (_ui.value.events + e.describe()).takeLast(200))
                }
            }
            try {
                job.start(dataset = AppConfig.dataset.value, config = AppConfig.train.value)
                _ui.value = _ui.value.copy(canResume = job.canResume)
            } catch (e: Throwable) {
                _ui.value = _ui.value.copy(error = e.message ?: e::class.java.simpleName)
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
        }.onFailure { _ui.value = _ui.value.copy(error = it.message) }
    }

    fun merge() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        viewModelScope.launch {
            runCatching { loaded.model.merge() }
                .onSuccess { _ui.value = _ui.value.copy(status = "merged=${it.merged}") }
                .onFailure { _ui.value = _ui.value.copy(error = it.message) }
        }
    }
}

data class TrainUiState(
    val running: Boolean = false,
    val status: String = "idle",
    val events: List<String> = emptyList(),
    val canResume: Boolean = false,
    val scheduled: String? = null,
    val datasetNote: String? = null,
    val error: String? = null,
)

private fun TrainingStatus.describe(): String = this::class.java.simpleName

private fun TrainingEvent.describe(): String = toString()

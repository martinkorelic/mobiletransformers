# Training Lifecycle & Checkpoint Contracts

**Priority (global #):** 17  |  **Prerequisites:** `00_code_plans/05_android_facade_foundation.md` (#16)  |  **Blocks:** `02_code_plans/01_hf_style_kotlin_facade.md` (#18)

---

## Purpose

`ORTTrainerNative` (`ORTTrainerNative.kt:19`) already implements a full on-device training loop, checkpoint/resume, end-of-training merge, and a rich callback stream. `LLMRepository` wraps it loosely with coroutine `Job`s and an `LLMState` enum (`LLMRepository.kt:30-38`, `runTraining`/`saveTraining` at `:467/:484`). What is missing is a **stable, public, lifecycle-shaped API** the facade (#16/#18) can expose without leaking ORT internals or the native handle, plus an explicit **checkpoint/resume contract** that surfaces the existing `training_state.json` metadata **without changing its native format**, and a structured **metrics/event stream**. This plan adds a `TrainingJob` abstraction and the contracts; it deliberately stops short of WorkManager scheduling (it only lays the foundation).

Existing facts to wrap (do not reimplement):
- **Loop & callbacks:** `startTraining(callback: TrainingCallback?)` (`ORTTrainerNative.kt:171`) drives epochs/steps and fires `TrainingCallback` (`LLMRepository.kt:49-65`): `onDataLoadEnd(totalSteps, stepsPerEpoch)`, `onStepEnd`, `onOptimizerStep`, `onEpochEnd`, `onMergeStart/End`, `onSaveModelStart/End`, `onCompletion`, `onError`.
- **Progress payload:** `TrainingProgress` (`ORTProgress.kt:5-16`): `currentStep`, `currentEpoch`, `totalLoss`, `epochLoss`, `stepLoss`, `learningRate`, `stepDurationMs`, `epochDurationMs`, `totalDurationMs`, `isCompleted`.
- **Metrics:** `TrainingStepMetrics` (`ORTTrainerNative.kt:47-54`: step, epoch, loss, learningRate, stepDurationMs, memoryUsageMB) gated by `trainingConfig.profileMetrics`; `TrainingSummary` (`ORTTrainerNative.kt:56-65`) written to `training_logs.json`.
- **Checkpoint files:** under `train/`: `checkpoint` dir (`ORTTrainerNative.kt:29`), `training_model.onnx`/`eval_model.onnx`/`optimizer_model.onnx`, and `training_state.json`.
- **Checkpoint state format (DO NOT CHANGE):** `TrainingState` (`ORTTrainerNative.kt:13-17`) = `{ schedulerState: SchedulerState, currentGlobalStep: Int, currentEpoch: Int }`, where `SchedulerState` (`ORTScheduler.kt:13-19`) = `{ totalSteps, warmupSteps, minLr, initialLr, currentStep }`. Saved by `saveTrainingState` (`ORTTrainerNative.kt:112`, Gson pretty-print) every `saveSteps` (`ORTTrainerNative.kt:314-317`), loaded by `loadTrainingState` (`ORTTrainerNative.kt:85`) iff `trainingConfig.loadFromState` (`ORTTrainerNative.kt:81`). Native ORT `checkpoint` (the binary weight/optimizer state) is persisted via `saveModel`/`releaseTrainingSession(saveCheckpoint)` JNI (`ORTTrainerNative.kt:315/584/593`).
- **Lifecycle entry/exit:** `LLMRepository.prepareTraining` (`LLMRepository.kt:436`) → `ReadyTrain`; `runTraining` (`:467`) → `Training`; `saveTraining(saveModel)` (`:484`) → `destroySession(saveModel)` → `NotInitialized`.

---

## Touched / new files

**Android Kotlin (new), package `com.martinkorelic.ortmobile.training`:**
- `TrainingJob.kt` — public handle: `start`, `cancel`, `status`, progress flow, checkpoint info, final result.
- `TrainingStatus.kt` — sealed status + `TrainingResult`.
- `TrainingEvent.kt` — structured event stream (sealed) mapping the `TrainingCallback` surface.
- `CheckpointInfo.kt` — read-only projection of `training_state.json` (no format change).
- `TrainingJobManager.kt` — creates/owns `TrainingJob`s per `sanitizedRepoId`; foundation hook for WorkManager (not wired here).

**Android Kotlin (light edit):**
- `LLMRepository.kt` — expose the underlying `Job` from `runTraining`/`saveTraining` (already returns `Job?`) and a way to read `ortTrainerNative.trainingState`. No loop logic changes.
- `ORTTrainerNative.kt` — add a read-only getter for current `TrainingState` and a cooperative cancel flag checked in the step loop (`while` at `:217`, inner `for` at `:241`). Do **not** change `TrainingState`/`SchedulerState` shapes or `saveTrainingState` JSON.

---

## Data contract — public lifecycle API

```kotlin
sealed interface TrainingStatus {
    data object Idle : TrainingStatus
    data object Preparing : TrainingStatus            // maps LLMState.ReadyTrain bring-up
    data class Running(val progress: TrainingProgress) : TrainingStatus  // LLMState.Training
    data object Merging : TrainingStatus              // onMergeStart..onMergeEnd
    data object Saving : TrainingStatus               // LLMState.SavingModel
    data class Completed(val result: TrainingResult) : TrainingStatus
    data class Cancelled(val checkpoint: CheckpointInfo?) : TrainingStatus
    data class Failed(val error: Throwable) : TrainingStatus
}

data class TrainingResult(
    val finalStep: Int, val finalEpoch: Int,
    val finalLoss: Float, val totalDurationMs: Long,
    val merged: Boolean, val checkpoint: CheckpointInfo?,
    val summary: ORTTrainerNative.TrainingSummary?   // from training_logs.json when profileMetrics
)

// Read-only projection of train/training_state.json — format unchanged.
data class CheckpointInfo(
    val currentGlobalStep: Int,                       // TrainingState.currentGlobalStep
    val currentEpoch: Int,                            // TrainingState.currentEpoch
    val schedulerStep: Int,                           // SchedulerState.currentStep
    val totalSteps: Int,                              // SchedulerState.totalSteps
    val checkpointDirPath: String,                    // ORTTrainerNative.checkpointPath
    val stateJsonPath: String,                        // train/training_state.json
    val exists: Boolean
)
```

**Event stream** (`SharedFlow<TrainingEvent>`), one-to-one with `TrainingCallback`:
```kotlin
sealed interface TrainingEvent {
    data class DataLoaded(val totalSteps: Int, val stepsPerEpoch: Int) : TrainingEvent
    data class Step(val progress: TrainingProgress) : TrainingEvent        // onStepEnd
    data class OptimizerStep(val progress: TrainingProgress) : TrainingEvent
    data class Epoch(val progress: TrainingProgress) : TrainingEvent       // onEpochEnd
    data class Metric(val m: ORTTrainerNative.TrainingStepMetrics) : TrainingEvent  // loss, memoryUsageMB, step, duration
    data object MergeStarted : TrainingEvent
    data object MergeFinished : TrainingEvent
    data class Saved(val progress: TrainingProgress) : TrainingEvent
    data class Done(val result: TrainingResult) : TrainingEvent
    data class Error(val t: Throwable) : TrainingEvent
}
```

`TrainingJob` surface:
```kotlin
class TrainingJob internal constructor(repo: LLMRepository, repoId: String) {
    val status: StateFlow<TrainingStatus>
    val events: SharedFlow<TrainingEvent>
    suspend fun start(args: ORTTrainingConfig? = null, preprocess: TaskPreprocessor? = null)
    suspend fun cancel(saveCheckpoint: Boolean = true)   // cooperative; persists training_state.json + native checkpoint
    fun checkpoint(): CheckpointInfo?                     // read current state file
    val canResume: Boolean                                // training_state.json exists && loadFromState path valid
}
```

---

## Implementation steps

1. **Adapter callback.** Implement an internal `TrainingCallback` (`LLMRepository.kt:49`) that translates each method into a `TrainingEvent` on the `events` flow and updates `status`:
   - `onDataLoadEnd` → `DataLoaded` + `Preparing→Running`.
   - `onStepEnd`/`onOptimizerStep`/`onEpochEnd` → corresponding events; `Running(progress)`.
   - `onStepEnd` also emits `Metric` when the step produced a `TrainingStepMetrics` (profileMetrics on).
   - `onMergeStart/End` → `Merging` then back; `onSaveModelStart/End` → `Saving`.
   - `onCompletion` → build `TrainingResult` (read `training_logs.json` `TrainingSummary` if present), `Completed`, `Done`.
   - `onError` → `Failed` + `Error`.
2. **Wire start.** `start()` calls `LLMRepository.prepareTraining(args, preprocess)` then `runTraining()` (`LLMRepository.kt:436/467`), passing the adapter as `trainingCallback`. Keep everything on the existing coroutine scope/`Job`.
3. **Cooperative cancel.** Add an `@Volatile var cancelRequested` in `ORTTrainerNative`, checked at the top of the step `for` loop (`ORTTrainerNative.kt:241`) and the epoch `while` (`:217`). On set, break out cleanly so the existing `saveModel`+`saveTrainingState` path (`:314-317`) can run if `saveCheckpoint`, then `destroySession(saveCheckpoint)`. `TrainingJob.cancel(saveCheckpoint)` sets the flag and joins the `Job`; emits `Cancelled(checkpoint())`.
4. **Checkpoint projection.** `CheckpointInfo` reads `train/training_state.json` with Gson into the existing `TrainingState` (`ORTTrainerNative.kt:13`) and projects fields — **no new fields, no rename**. `checkpointDirPath` = `ORTTrainerNative.checkpointPath` (`:29`). `canResume` = state file exists AND `trainingConfig.loadFromState` would load it (`:81`).
5. **Resume.** Resume is already native: set `trainingConfig.loadFromState = true`; `ORTTrainerNative.init` loads state (`:82`), `startTraining` restores scheduler + `globalStep`/`epoch` and skips already-done batches (`:196-209`, mid-epoch skip at `:249-253`). `TrainingJob.start` on a job with `canResume` simply starts with `loadFromState=true`; the lifecycle reports resumed `currentStep` via the first `DataLoaded`/`Step` events.
6. **Final status & merge.** Merge at end is governed by `trainingConfig.mergeWeightsAtEnd` → `mergeExportSessionWeights()` (`ORTTrainerNative.kt:369-387`), which writes per-tensor external initializers into `inference/` per the handoff map (#7/#8). `TrainingResult.merged` reflects whether this ran.
7. **Manager + WorkManager seam.** `TrainingJobManager.getOrCreate(repoId)` owns one `TrainingJob` per package. Expose a `TrainingJobSpec` (repoId + `ORTTrainingConfig` snapshot) that a future WorkManager `Worker` can reconstruct — **define the spec, do not add the WorkManager dependency or Worker here.**

---

## Interactions

- **#16 facade foundation:** `TrainingJob`/`TrainingStatus`/`TrainingEvent` are the public training surface the SDK facade exposes; the facade hides `LLMRepository`/`ORTTrainerNative`/native handle.
- **#18 HF-style facade:** `MobileTransformers.train(...)` returns a `TrainingJob`; `merge`/`generate` consume the merged `inference/` output.
- **#7/#8 handoff + merger:** end-of-training merge produces the inference external initializers; this plan reports merge progress but owns no merge logic.
- **#12 cache bridge:** training reads/writes within `<cacheDir>/<sanitizedRepoId>/train/`; merge output lands in `inference/` so the same package is immediately inferenceable.

---

## Tests & smokes

- **Status mapping (unit, fake `TrainingCallback`):** drive the adapter with a scripted callback sequence (dataLoad→steps→optimizer→epoch→merge→save→completion) and assert the `status` transitions and `events` order exactly.
- **Cancel:** start a fake long loop, call `cancel(saveCheckpoint=true)`; assert the loop breaks, `saveTrainingState` was invoked, `Cancelled(checkpoint)` emitted, and `checkpoint().currentGlobalStep` reflects the persisted step. Repeat `saveCheckpoint=false` → no state write.
- **Checkpoint round-trip:** write a `training_state.json` fixture with known `currentGlobalStep=120, currentEpoch=2, schedulerState.currentStep=120`; assert `CheckpointInfo` projects them and `canResume==true`; assert the on-disk JSON is **unchanged** after reading (format preserved).
- **Resume:** train N steps with `saveSteps` small so a checkpoint lands; restart with `loadFromState=true`; assert first emitted `Step.currentStep` ≥ saved step and no double-counting (mid-epoch skip honored).
- **Result/summary:** with `profileMetrics=true`, assert `TrainingResult.summary` is populated from `training_logs.json` and `Metric` events carry `memoryUsageMB`.
- **Instrumentation smoke:** tiny dataset + tiny LoRA package; run `TrainingJob.start` to completion with `mergeWeightsAtEnd=true`; assert `inference/` gains per-tensor `.bin` files and `Completed.merged==true`, then a generation step succeeds via `LLMRepository`.

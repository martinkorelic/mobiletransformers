# Charging-Cycle Training Scheduler (WorkManager)

**Priority #34 | Prerequisites: #18 (`00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md`), #17 (`00_code_plans/05_android_facade_foundation.md`) | Blocks: #36 (`04_code_plans/04`, federated round scheduling)**

> Tier-3 systems capability, not a new ML algorithm. The contribution is robust on-device execution under mobile constraints with measured energy/thermal traces. Must not block v1.0.

## Purpose

Run fine-tuning during favorable device states (charging, idle, battery-not-low) via Android `WorkManager`, in bounded chunks that checkpoint often and resume cleanly across interrupted charge cycles. Framed as "train safely when constraints allow, checkpoint, resume" — not invisible background training. Built entirely on the existing `TrainingState`/`saveSteps`/`loadFromState` resume primitives; the scheduler lives **outside** `ORTTrainerNative`.

## Touched / new files

Kotlin:
- NEW `android/.../scheduler/TrainingWorker.kt` — `CoroutineWorker` (foreground) that runs one bounded training chunk.
- NEW `android/.../scheduler/TrainingScheduler.kt` — enqueues `WorkManager` work with constraints; exposes paused/resumed/interrupted/completed callbacks.
- NEW `android/.../scheduler/TrainingScheduleConfig.kt` — the config below.
- `android/.../repository/TrainingRepository.kt` — `performTraining` (`:9-22`) is reused per chunk; the scheduler calls it, does not reimplement it.
- `android/.../ORTTrainerNative.kt` — unchanged; resume uses `TrainingState` (`:13-17`), `saveTrainingState`/`loadTrainingState` (`:85-150`), `saveSteps` checkpoint (`:314-317`). Per-chunk bound via `maxSteps` (`startTraining` already supports `maxSteps`, `:186-189`).
- `android/.../ORTScheduler.kt` — **blocking gap to close (Linear scheduler only).** `LinearLRScheduler.loadFromState`/`stateDict` are `TODO("Not yet implemented")` (`:156-161`), so a **linear**-schedule run is not persisted across a chunk boundary — a resumed chunk would restart the LR schedule, corrupting multi-chunk training. (`CosineLRScheduler` already implements both, `:77`/`:111`, and `ORTTrainerNative` already saves/restores `schedulerState` via `training_state.json` — `ORTTrainerNative.kt:112-139/204` — so the cosine path works today.) Implement the Linear pair mirroring the Cosine overrides; note `SchedulerState`'s current fields (`totalSteps`/`warmupSteps`/`minLr`/`initialLr`/`currentStep`) are cosine-shaped — map Linear's `baseLr`/`startFactor`/`endFactor`/`totalIters` onto them (or extend the state additively) **without breaking the existing `training_state.json` format** (#18's do-not-change rule). The chunked-execution `loadFromState(training_state.json)` step below depends on this.
- `AndroidManifest.xml` (library + sample) — declare the foreground service type + permissions.

## Data contracts / interfaces

### `TrainingScheduleConfig`

```kotlin
data class TrainingScheduleConfig(
    val requiresCharging: Boolean = true,
    val requiresDeviceIdle: Boolean = false,
    val requiresBatteryNotLow: Boolean = true,
    val maxRuntimeMinutes: Int = 30,        // bound per chunk
    val checkpointEverySteps: Int = 50,     // maps to ORTTrainingConfig.saveSteps
    val notificationTitle: String = "Training",
    val notificationChannelId: String = "mt_training",
)
```

### Chunked execution contract

```
each TrainingWorker run:
    acquire single model/session lock (no race with foreground train/merge/generate)
    check thermal + battery + storage before starting
    loadFromState(training_state.json)         # resume globalStep/epoch/scheduler
    run performTraining(bounded by maxSteps/maxRuntimeMinutes)
    saveTrainingState() + saveModel(checkpoint)  # always checkpoint before exit
    release native resources + lock
    if more steps remain: enqueue next chunk (constraints re-evaluated by WorkManager)
```

Foreground service with a mandatory persistent notification (Android 14+ requires a declared service type + permission). Doze defers work while unplugged/idle; charging exits Doze and lets pending chunks run.

## Implementation steps

1. Add `TrainingScheduleConfig` + map `checkpointEverySteps → ORTTrainingConfig.saveSteps`. **Implement `LinearLRScheduler.stateDict()`/`loadFromState()` (`:156-161`) first** — without them, a linear-schedule cross-chunk resume restarts the schedule (Cosine already works). The chunk's `saveTrainingState()` must include the scheduler `stateDict()` and `loadFromState(...)` must restore it.
2. `TrainingWorker` (foreground `CoroutineWorker`): notification with progress + cancel action; bounded chunk; always checkpoint on exit (success/stop/cancel).
3. `TrainingScheduler`: build `Constraints` (charging/idle/battery); enqueue unique work; chain next chunk; expose lifecycle callbacks.
4. Single model/session lock shared with foreground operations (the session lock from #18).
5. Separate network from training: model/package download uses network constraints; training runs from local files only.
6. Decide + declare Android 14+ foreground service type + manifest permissions in the **library**, not by accident in the sample app.
7. Log thermal/battery/energy per chunk in the `docs/mobile_evaluation.md` style.

## Interactions

- **#18 (lifecycle/checkpoint contracts)**: `TrainingJob`, progress events, checkpoint metadata, cooperative cancellation, session lock come from there.
- **`TrainingRepository`/`ORTTrainerNative`**: reused for local work; resume via existing state files.
- **#36 (federated Android)**: reuses this scheduler for round scheduling.

## References

- Long-running foreground workers (`setForeground`): https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running
- Defining constraints (`Constraints.Builder`): https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started/define-work
- `CoroutineWorker` reference: https://developer.android.com/reference/androidx/work/CoroutineWorker
- `dataSync` FGS type (API 34+): https://developer.android.com/develop/background-work/services/fgs/service-types
- FGS 6h/24h time cap (API 35+): https://developer.android.com/develop/background-work/services/fgs/timeout
- Thermal-status listener API: https://developer.android.com/reference/android/os/PowerManager.OnThermalStatusChangedListener

## Worked example

A charging-constrained, foreground, self-chaining chunk worker that checkpoints and pauses on thermal pressure:

```kotlin
// TrainingScheduler.kt — constraints (only run when charging + battery OK)
val constraints = Constraints.Builder()
    .setRequiresCharging(true)
    .setRequiresBatteryNotLow(true)
    .build()
WorkManager.getInstance(ctx).enqueueUniqueWork(
    "mt_training", ExistingWorkPolicy.KEEP,
    OneTimeWorkRequestBuilder<TrainingWorker>().setConstraints(constraints).build(),
)
```

```kotlin
// TrainingWorker.kt — one bounded chunk, foreground dataSync, checkpoint, chain next
class TrainingWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result {
        setForeground(getForegroundInfo())   // FOREGROUND_SERVICE_TYPE_DATA_SYNC (API 34+)
        val pm = applicationContext.getSystemService(PowerManager::class.java)
        if (pm.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) {
            return Result.retry()             // pause; WorkManager re-runs when cooler
        }
        scheduler.loadFromState("training_state.json")        // resume globalStep + LR schedule
        repository.performTraining(maxSteps = config.maxStepsPerChunk)   // bounded chunk
        scheduler.saveTrainingState()                          // always checkpoint before exit
        return if (stepsRemain()) { enqueueNextChunk(); Result.success() } else Result.success()
    }

    override suspend fun getForegroundInfo() = ForegroundInfo(
        NOTIF_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
    )
}
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- JVM config-mapping test (`TrainingScheduleConfigTest.kt`): `checkpointEverySteps → ORTTrainingConfig.saveSteps`; `maxRuntimeMinutes`/`requiresCharging`/`requiresBatteryNotLow` map onto the `Constraints` builder.
- **Scheduler-state resume test** (`ORTSchedulerTest.kt`): for BOTH `LinearLRScheduler` (the `:156-161` `TODO` being closed) and `CosineLRScheduler` (regression), serialize `stateDict()`, restore via `loadFromState`, and assert the next LR equals the uninterrupted-run LR at the same step (no schedule restart).
- Session-lock unit test (`SessionLockTest.kt`): the scheduled worker's lock acquisition is mutually exclusive with a foreground train/merge/generate holder.
- Plus the module **compiles** (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `globalStep` resume parse test: feed a fixture `training_state.json`, call `loadFromState`, assert the restored `globalStep`/`epoch` equal the fixture values (no device needed).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Android instrumentation: worker creation + foreground `dataSync` notification shown.
- Constraint/Doze behavior: unplugged-idle defers; charging resumes; cancel checkpoints cleanly.
- Full resume test: interrupt after a checkpoint, restart, confirm `globalStep` advances from `training_state.json` and the LR continues uninterrupted.
- Energy/thermal metrics export for a short scheduled run (`docs/mobile_evaluation.md` style).

**Workflow (end-to-end)** — *(CHECKPOINT #34, device/manual)* schedule a charging-constrained job that runs multiple bounded chunks, checkpoints, survives Doze, and resumes the LR schedule correctly: enqueue with `setRequiresCharging(true).setRequiresBatteryNotLow(true)`, let WorkManager run chunk 1 as a `dataSync` foreground worker, unplug to force Doze (assert pending chunks defer), re-plug to resume, and assert across the chunk boundary that `globalStep` advanced and the resumed LR matches the uninterrupted-run LR at the same step. Capture thermal + energy traces per chunk.

**Definition of done** — `LinearLRScheduler.stateDict()`/`loadFromState()` are implemented and Cosine still passes (the LR schedule survives a chunk boundary for both scheduler types); a charging/idle/battery-constrained `WorkManager` job runs bounded chunks as a `dataSync` foreground service with a persistent notification, checkpoints on every exit (success/stop/cancel), chains the next chunk, and pauses on `THERMAL_STATUS_SEVERE`; resume advances `globalStep` from `training_state.json` without restarting the LR schedule; the scheduled worker cannot race foreground operations (shared session lock from #18); and a short scheduled run exports thermal/energy traces.

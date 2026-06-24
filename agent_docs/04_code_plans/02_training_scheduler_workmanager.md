# Charging-Cycle Training Scheduler (WorkManager)

**Priority #33 | Prerequisites: #17 (`00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md`), #16 (`00_code_plans/05_android_facade_foundation.md`) | Blocks: #35 (`04_code_plans/04`, federated round scheduling)**

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
- `android/.../ORTScheduler.kt` — **blocking gap to close.** The LR-scheduler `loadFromState`/`stateDict` are `TODO("Not yet implemented")` (`:156-161`), so the learning-rate schedule is **not** persisted across a chunk boundary — a resumed chunk would restart the LR schedule, corrupting multi-chunk training. Implement `stateDict()` (serialize `SchedulerState`: current step, base/last LR, warmup/decay progress) and `loadFromState()` to restore it, mirroring the already-implemented overrides at `:77,:111`. The chunked-execution `loadFromState(training_state.json)` step below depends on this.
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

1. Add `TrainingScheduleConfig` + map `checkpointEverySteps → ORTTrainingConfig.saveSteps`. **Implement `ORTScheduler.stateDict()`/`loadFromState()` (`:156-161`) first** — without persisted LR-scheduler state, cross-chunk resume restarts the schedule. The chunk's `saveTrainingState()` must include the scheduler `stateDict()` and `loadFromState(...)` must restore it.
2. `TrainingWorker` (foreground `CoroutineWorker`): notification with progress + cancel action; bounded chunk; always checkpoint on exit (success/stop/cancel).
3. `TrainingScheduler`: build `Constraints` (charging/idle/battery); enqueue unique work; chain next chunk; expose lifecycle callbacks.
4. Single model/session lock shared with foreground operations (the session lock from #17).
5. Separate network from training: model/package download uses network constraints; training runs from local files only.
6. Decide + declare Android 14+ foreground service type + manifest permissions in the **library**, not by accident in the sample app.
7. Log thermal/battery/energy per chunk in the `docs/mobile_evaluation.md` style.

## Interactions

- **#17 (lifecycle/checkpoint contracts)**: `TrainingJob`, progress events, checkpoint metadata, cooperative cancellation, session lock come from there.
- **`TrainingRepository`/`ORTTrainerNative`**: reused for local work; resume via existing state files.
- **#35 (federated Android)**: reuses this scheduler for round scheduling.

## Tests & smokes

- JVM config-mapping test (`checkpointEverySteps → saveSteps`).
- Android instrumentation: worker creation + foreground notification shown.
- Constraint/Doze behavior: unplugged-idle defers; charging resumes; cancel checkpoints cleanly.
- Resume test: interrupt after a checkpoint, restart, confirm `globalStep` advances from `training_state.json`.
- **Scheduler-state resume test**: interrupt mid-schedule, restore via `ORTScheduler.loadFromState`, assert the next LR equals the uninterrupted-run LR at the same step (no schedule restart).
- Session-lock test: scheduled worker cannot race foreground train/merge/generate.
- Energy/thermal metrics export for a short scheduled run.

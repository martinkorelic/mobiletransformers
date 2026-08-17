package com.martinkorelic.mobiletransformers.scheduler

import androidx.work.Constraints
import com.martinkorelic.mobiletransformers.ORTTrainingConfig

/**
 * #34: when a scheduled training chunk is allowed to run, and how big it is.
 *
 * The framing this plan insists on is "train safely when constraints allow, checkpoint, resume" —
 * **not** invisible background training. Every field here is either a device-state precondition or a
 * bound on how much work one chunk may do before it must checkpoint.
 *
 * This is pure data with two pure mappings ([toConstraints], [applyTo]) so both can be asserted on a
 * host; the worker that consumes them is the device leg.
 */
data class TrainingScheduleConfig(
    /** Only run while plugged in. The whole point of the feature. */
    val requiresCharging: Boolean = true,
    /** Also wait for device idle. Off by default: it makes a short demo run untestable. */
    val requiresDeviceIdle: Boolean = false,
    /** Never run the battery down past the system's "low" threshold. */
    val requiresBatteryNotLow: Boolean = true,
    /** Wall-clock bound on one chunk. */
    val maxRuntimeMinutes: Int = 30,
    /** Step bound on one chunk. Whichever bound is hit first ends the chunk — and it checkpoints. */
    val maxStepsPerChunk: Int = 50,
    /** Maps onto [ORTTrainingConfig.saveSteps]: how often the native loop persists mid-chunk. */
    val checkpointEverySteps: Int = 50,
    /** Total steps the whole job should reach across all chunks; null = run the configured epochs out. */
    val totalSteps: Int? = null,
    /**
     * Earliest the **first** chunk may start, in minutes from `schedule()`. 0 = as soon as the
     * constraints allow.
     *
     * ### What Android does and does not let you promise
     *
     * WorkManager's `setInitialDelay` is the only start-time control there is, and it is a **floor,
     * not an appointment**: the system batches deferrable work, and Doze can hold it well past the
     * delay. So "start at 02:00" is expressible as "not before 02:00" and nothing stronger. An exact
     * wall-clock start would need `AlarmManager.setExactAndAllowWhileIdle` plus the
     * `SCHEDULE_EXACT_ALARM` permission, which is the wrong trade for a multi-hour training job —
     * exact alarms exist for alarm clocks and calendar reminders, and Google Play restricts them to
     * that. The constraints ([requiresCharging] and friends) are the real gate anyway; a delay only
     * moves the earliest moment they are consulted.
     *
     * Applied to the first chunk only. Chunk N+1 re-enqueues immediately and waits on constraints —
     * re-delaying each chunk would stretch a run by the delay on every boundary.
     */
    val initialDelayMinutes: Long = 0L,
    val notificationTitle: String = "Training",
    val notificationChannelId: String = "mt_training",
) {
    init {
        require(maxRuntimeMinutes > 0) { "maxRuntimeMinutes must be positive, was $maxRuntimeMinutes" }
        require(maxStepsPerChunk > 0) { "maxStepsPerChunk must be positive, was $maxStepsPerChunk" }
        require(checkpointEverySteps > 0) {
            "checkpointEverySteps must be positive, was $checkpointEverySteps"
        }
        require(initialDelayMinutes >= 0) {
            "initialDelayMinutes cannot be negative, was $initialDelayMinutes"
        }
    }

    /** The WorkManager preconditions. Storage-not-low is unconditional: a checkpoint needs room. */
    fun toConstraints(): Constraints =
        Constraints.Builder()
            .setRequiresCharging(requiresCharging)
            .setRequiresDeviceIdle(requiresDeviceIdle)
            .setRequiresBatteryNotLow(requiresBatteryNotLow)
            .setRequiresStorageNotLow(true)
            .build()

    /**
     * Bound one chunk of an existing training config.
     *
     * `loadFromState = true` is what makes chunk N+1 continue chunk N rather than restart it: it
     * restores `globalStep`/`epoch` **and** the LR scheduler state from `training_state.json`.
     * `saveModelAtEnd` guarantees the chunk boundary is itself a checkpoint. Merging is deliberately
     * NOT done per chunk — merge is an end-of-job act, and doing it every chunk would rewrite every
     * trainable tensor on disk each time.
     */
    fun applyTo(base: ORTTrainingConfig, resumedGlobalStep: Int = 0): ORTTrainingConfig =
        base.copy(
            // `maxSteps` is a CUMULATIVE target, not a per-chunk budget: `ORTTrainerNative` computes
            // `totalSteps = maxSteps ?: epochs*stepsPerEpoch` and loops `while (globalStep < totalSteps)`
            // AFTER restoring `globalStep` from `training_state.json`. Passing the chunk size directly
            // therefore made every chunk after the first a no-op — chunk 2 restored globalStep=2, read
            // "Training for 2 steps", found `2 < 2` false and exited having trained nothing, while
            // still reporting success. Caught on device by `ScheduledTrainingDeviceTest`; the host
            // tests could not see it because they exercise the LR arithmetic, not the loop bound.
            maxSteps = resumedGlobalStep + maxStepsPerChunk,
            saveSteps = checkpointEverySteps,
            loadFromState = true,
            saveModelAtEnd = true,
            mergeWeightsAtEnd = false,
        )
}

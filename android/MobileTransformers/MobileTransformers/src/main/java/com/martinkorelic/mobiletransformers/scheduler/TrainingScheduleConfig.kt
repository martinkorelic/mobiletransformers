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
    val notificationTitle: String = "Training",
    val notificationChannelId: String = "mt_training",
) {
    init {
        require(maxRuntimeMinutes > 0) { "maxRuntimeMinutes must be positive, was $maxRuntimeMinutes" }
        require(maxStepsPerChunk > 0) { "maxStepsPerChunk must be positive, was $maxStepsPerChunk" }
        require(checkpointEverySteps > 0) {
            "checkpointEverySteps must be positive, was $checkpointEverySteps"
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

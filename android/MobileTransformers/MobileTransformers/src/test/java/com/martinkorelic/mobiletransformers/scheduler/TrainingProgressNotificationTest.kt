package com.martinkorelic.mobiletransformers.scheduler

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The fraction behind the ongoing training notification.
 *
 * `TrainingWorker.foregroundInfo` has always accepted a `progress` and called `setProgress(100, …)`
 * with it — but its only call site passed `null`, once, before the first optimizer step. The
 * mechanism worked and nothing fed it, so a multi-hour run showed a static "Training chunk 1" that
 * gave no sign the device was still working. `reportProgress` now drives it from the step stream, and
 * this pins the arithmetic it drives it with.
 *
 * The subtle part is that both numbers are **cumulative across chunks**: `currentStep` comes from
 * `globalStep` restored out of `training_state.json`, and `totalSteps` is a whole-run target. Treating
 * either as chunk-local would restart the bar at 0% on every resume.
 */
class TrainingProgressNotificationTest {

    @Test
    fun reportsTheWholeRunFractionNotTheChunks() {
        // Chunk 3 resuming at step 150 of a 200-step run is 75% done, not 0%.
        assertEquals(75, TrainingWorker.percentComplete(currentStep = 150, totalSteps = 200))
    }

    @Test
    fun spansTheRange() {
        assertEquals(0, TrainingWorker.percentComplete(0, 200))
        assertEquals(50, TrainingWorker.percentComplete(100, 200))
        assertEquals(100, TrainingWorker.percentComplete(200, 200))
    }

    /**
     * A chunk may legitimately overshoot: `applyTo` sets `maxSteps = resumedGlobalStep +
     * maxStepsPerChunk`, which the final chunk can carry past the target. An uncoerced value would
     * hand `setProgress` a figure over 100.
     */
    @Test
    fun overshootIsClampedRatherThanReportedAbove100() {
        assertEquals(100, TrainingWorker.percentComplete(205, 200))
    }

    /** No declared target means no honest fraction; the notification stays indeterminate. */
    @Test
    fun anAbsentOrZeroTargetYieldsNoFraction() {
        assertNull(TrainingWorker.percentComplete(10, null))
        assertNull(TrainingWorker.percentComplete(10, 0))
        assertNull(TrainingWorker.percentComplete(10, -5))
    }

    @Test
    fun aNegativeStepYieldsNoFraction() {
        assertNull(TrainingWorker.percentComplete(-1, 200))
    }

    /**
     * The throttle only suppresses repeats if the value is stable within a percent band — integer
     * division gives that, but only if it does not overflow first. A long-running job at a large step
     * count multiplied by 100 in Int space would wrap negative.
     */
    @Test
    fun aLargeStepCountDoesNotOverflow() {
        assertEquals(50, TrainingWorker.percentComplete(50_000_000, 100_000_000))
    }
}

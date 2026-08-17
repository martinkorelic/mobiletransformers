package com.martinkorelic.mobiletransformers

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #34/#18: learning-rate schedule state round-trip (JVM, no device/JNI).
 *
 * [ORTTrainerNative.saveTrainingState] calls `scheduler.stateDict()` on EVERY checkpoint write, and
 * [LinearLRScheduler] — the default schedule — implemented it as `TODO("Not yet implemented")`.
 * `NotImplementedError` is an `Error`, so it was not caught by the `catch (e: Exception)` guarding the
 * save: any run with `maxSteps > saveSteps` on the default schedule died at its first checkpoint.
 */
class ORTSchedulerTest {

    private val eps = 1e-5f

    // --- LinearLRScheduler ------------------------------------------------------------------------

    @Test
    fun linearStateDictDoesNotThrowAndCarriesThePosition() {
        val s = LinearLRScheduler(baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 5)
        repeat(3) { s.step() }

        val state = s.stateDict()
        assertEquals(5, state.totalSteps)
        assertEquals(0, state.warmupSteps) // linear has no warmup phase
        assertEquals(1e-3f, state.initialLr, eps) // baseLr * startFactor
        assertEquals(1e-3f / 3f, state.minLr, eps) // baseLr * endFactor
        assertEquals(3, state.currentStep)
    }

    @Test
    fun linearResumesMidScheduleWithoutRestartingIt() {
        val a = LinearLRScheduler(baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 10)
        repeat(4) { a.step() }
        val lrAtInterrupt = a.getLR()

        val b = LinearLRScheduler(baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 10)
        b.loadFromState(a.stateDict())

        // Resuming restores the LR the interrupted run was last on, not the start of the schedule.
        assertEquals(lrAtInterrupt, b.getLR(), eps)
        assertTrue("resumed LR must have decayed below baseLr", b.getLR() < 1e-3f)

        // ...and the schedules stay in lockstep from there on.
        repeat(5) { assertEquals(a.step(), b.step(), eps) }
    }

    @Test
    fun linearLoadFromStepZeroIsTheScheduleStart() {
        val s = LinearLRScheduler(baseLr = 2e-3f, startFactor = 1f, endFactor = 0.5f, totalIters = 4)
        repeat(3) { s.step() }
        s.loadFromState(SchedulerState(4, 0, 1e-3f, 2e-3f, 0))
        assertEquals(2e-3f, s.getLR(), eps)
    }

    @Test
    fun linearClampsToEndFactorPastTotalIters() {
        val s = LinearLRScheduler(baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 3)
        repeat(10) { s.step() }
        assertEquals(1e-3f / 3f, s.getLR(), eps)
    }

    // --- CosineLRScheduler (the reference implementation; guards against regressions) --------------

    @Test
    fun cosineResumesMidScheduleWithoutRestartingIt() {
        val a = CosineLRScheduler(totalSteps = 20, warmupSteps = 2, minLr = 0f, initialLr = 1e-3f)
        repeat(7) { a.step() }

        val b = CosineLRScheduler(totalSteps = 20, warmupSteps = 2, minLr = 0f, initialLr = 1e-3f)
        b.loadFromState(a.stateDict())

        assertEquals(a.getLR(), b.getLR(), eps)
        repeat(5) { assertEquals(a.step(), b.step(), eps) }
    }

    /** Both schedulers must project onto the SAME record — `training_state.json` has one shape. */
    @Test
    fun bothSchedulersRoundTripThroughTheSharedStateRecord() {
        for (scheduler in listOf(
            LinearLRScheduler(baseLr = 1e-3f, totalIters = 8),
            CosineLRScheduler(totalSteps = 8, initialLr = 1e-3f),
        )) {
            repeat(3) { scheduler.step() }
            val state = scheduler.stateDict()
            assertEquals(8, state.totalSteps)
            assertEquals(3, state.currentStep)
            scheduler.loadFromState(state) // must not throw
        }
    }
}

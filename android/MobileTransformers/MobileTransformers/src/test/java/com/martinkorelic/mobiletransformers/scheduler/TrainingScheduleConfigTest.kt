package com.martinkorelic.mobiletransformers.scheduler

import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #34: the pure half of the scheduler — how a schedule config bounds one chunk.
 *
 * The device leg proves a chunk runs; these prove it is bounded and resumable in the first place,
 * which is what makes chunk N+1 a continuation rather than a restart.
 */
class TrainingScheduleConfigTest {

    @Test
    fun `chunk bounds map onto the training config`() {
        val config = TrainingScheduleConfig(maxStepsPerChunk = 25, checkpointEverySteps = 5)
        val bounded = config.applyTo(ORTTrainingConfig(repoName = "m"))

        assertEquals(25, bounded.maxSteps)
        assertEquals(5, bounded.saveSteps)
    }

    @Test
    fun `the chunk budget is cumulative, not per-chunk`() {
        // The defect this pins: `ORTTrainerNative` computes `totalSteps = maxSteps ?: ...` and loops
        // `while (globalStep < totalSteps)` AFTER restoring globalStep. So `maxSteps` is a target, not
        // a budget. Passing the chunk size directly made chunk 2 onward no-ops — restore 2, "train for
        // 2 steps", `2 < 2` is false, exit having trained nothing, report success.
        val config = TrainingScheduleConfig(maxStepsPerChunk = 25)

        assertEquals(25, config.applyTo(ORTTrainingConfig(repoName = "m"), resumedGlobalStep = 0).maxSteps)
        assertEquals(50, config.applyTo(ORTTrainingConfig(repoName = "m"), resumedGlobalStep = 25).maxSteps)
        assertEquals(75, config.applyTo(ORTTrainingConfig(repoName = "m"), resumedGlobalStep = 50).maxSteps)

        // Every chunk must be able to do a FULL chunk of work, no matter how far in it starts.
        for (resumed in listOf(0, 1, 7, 999)) {
            val bounded = config.applyTo(ORTTrainingConfig(repoName = "m"), resumedGlobalStep = resumed)
            assertEquals(25, bounded.maxSteps!! - resumed)
        }
    }

    @Test
    fun `a chunk always resumes and always checkpoints`() {
        // These two are what make chunking correct at all. `loadFromState` restores globalStep/epoch
        // AND the LR scheduler position; `saveModelAtEnd` makes the chunk boundary a checkpoint.
        val bounded = TrainingScheduleConfig().applyTo(ORTTrainingConfig(repoName = "m"))

        assertTrue("chunk N+1 must continue, not restart", bounded.loadFromState)
        assertTrue("the chunk boundary must itself be a checkpoint", bounded.saveModelAtEnd)
    }

    @Test
    fun `merging is not done per chunk`() {
        // Merge rewrites every trainable tensor on disk. Doing it at each of N chunk boundaries would
        // multiply the most expensive I/O in the system by N for no benefit — merge is an
        // end-of-job act, and the default ORTTrainingConfig has it on.
        val base = ORTTrainingConfig(repoName = "m")
        assertTrue("precondition: the default merges at end", base.mergeWeightsAtEnd)
        assertFalse(TrainingScheduleConfig().applyTo(base).mergeWeightsAtEnd)
    }

    @Test
    fun `caller settings unrelated to chunking survive`() {
        val base = ORTTrainingConfig(repoName = "m", batchSize = 7, numTrainEpochs = 3)
        val bounded = TrainingScheduleConfig().applyTo(base)

        assertEquals(7, bounded.batchSize)
        assertEquals(3, bounded.numTrainEpochs)
        assertEquals("m", bounded.repoName)
    }

    @Test
    fun `non-positive bounds fail closed at construction`() {
        for (bad in listOf(
            { TrainingScheduleConfig(maxRuntimeMinutes = 0) },
            { TrainingScheduleConfig(maxStepsPerChunk = 0) },
            { TrainingScheduleConfig(checkpointEverySteps = -1) },
        )) {
            try {
                bad()
                throw AssertionError("expected an IllegalArgumentException")
            } catch (expected: IllegalArgumentException) {
                assertTrue(expected.message!!.contains("must be positive"))
            }
        }
    }

    @Test
    fun `thermal pause boundary is SEVERE, and an unknown reading does not pause`() {
        // android.os.PowerManager.THERMAL_STATUS_* : NONE=0 LIGHT=1 MODERATE=2 SEVERE=3 CRITICAL=4
        assertFalse(ThermalGuard.shouldPause(0))
        assertFalse(ThermalGuard.shouldPause(2))
        assertTrue(ThermalGuard.shouldPause(3))
        assertTrue(ThermalGuard.shouldPause(4))

        // -1 means the platform predates API 29. Refusing to train because the device is too old to
        // report its temperature would disable the feature on exactly the hardware it targets.
        assertFalse(ThermalGuard.shouldPause(-1))
    }

    /**
     * The #34 assertion that single-resume tests do not make: **N** chunk boundaries, not one.
     *
     * A schedule that restarts, or that loses one step per boundary, still passes a single
     * save/restore round trip — the error only accumulates across chunks. So the comparison is
     * against the uninterrupted run at the same global step: a relative assertion, with no LR
     * constant written down anywhere that could encode one schedule and silently measure another.
     */
    @Test
    fun `the LR schedule survives repeated chunk boundaries`() {
        val stepsPerChunk = 5
        val chunks = 3
        val eps = 1e-9f

        val uninterrupted =
            com.martinkorelic.mobiletransformers.LinearLRScheduler(
                baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 20,
            )
        val expected = (1..stepsPerChunk * chunks).map { uninterrupted.step() }

        // Each chunk is a FRESH scheduler restored from the previous chunk's persisted state —
        // which is what a new process, or a new Worker instance, actually gets.
        val actual = mutableListOf<Float>()
        var carried =
            com.martinkorelic.mobiletransformers.LinearLRScheduler(
                baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 20,
            ).stateDict()
        repeat(chunks) {
            val chunkScheduler =
                com.martinkorelic.mobiletransformers.LinearLRScheduler(
                    baseLr = 1e-3f, startFactor = 1f, endFactor = 1f / 3f, totalIters = 20,
                )
            chunkScheduler.loadFromState(carried)
            repeat(stepsPerChunk) { actual.add(chunkScheduler.step()) }
            carried = chunkScheduler.stateDict()
        }

        assertEquals(expected.size, actual.size)
        expected.forEachIndexed { i, lr ->
            assertEquals("LR diverged at global step ${i + 1}", lr, actual[i], eps)
        }
        // And the schedule really did move, so the comparison is not two flat lines agreeing.
        assertTrue("the schedule must decay across the run", actual.last() < actual.first())
    }

    @Test
    fun `a trace row carries every column its header declares`() {
        val sample = ThermalSample(
            thermalStatus = 2,
            batteryPercent = 88,
            batteryTemperatureDeciC = 301,
            chargeCounterMicroAh = 3_210_000L,
            timestampMillis = 1_700_000_000_000L,
        )
        val row = sample.toCsvRow(chunk = 3, globalStep = 150)

        assertEquals(ThermalSample.CSV_HEADER.split(",").size, row.split(",").size)
        assertEquals(
            "1700000000000,3,150,2,88,301,3210000",
            row,
        )
    }
}

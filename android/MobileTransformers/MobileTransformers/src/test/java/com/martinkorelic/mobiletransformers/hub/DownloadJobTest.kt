package com.martinkorelic.mobiletransformers.hub

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * How a background pull reports itself to a facade-only caller.
 *
 * `PackageDownloadWorker` was complete, JVM-tested and had **zero call sites**: it emitted its
 * progress as raw `WorkInfo` keys, so reading a pull it had started meant depending on
 * `androidx.work` and decoding the worker's own constants — past the facade the app is supposed to
 * be a worked example of. [DownloadJob] is that interpretation, and this pins the two decisions in it
 * that are easy to get quietly wrong.
 */
class DownloadJobTest {

    private fun job(
        state: DownloadJob.State = DownloadJob.State.Running,
        bytesDone: Long = 0L,
        bytesTotal: Long? = null,
    ) = DownloadJob(state = state, bytesDone = bytesDone, bytesTotal = bytesTotal)

    /**
     * The manifest does not always size its files, and the worker signals that with `-1`
     * (`KEY_BYTES_TOTAL`), mirroring `DownloadProgress.bytesTotal == null`. A caller that treated the
     * sentinel as a real total would divide by a negative and render a bar running backwards.
     */
    @Test
    fun anUnknownTotalYieldsNoFractionRatherThanAWrongOne() {
        assertNull(job(bytesDone = 5_000, bytesTotal = null).fraction)
        assertNull("zero is not a denominator", job(bytesDone = 5_000, bytesTotal = 0).fraction)
    }

    @Test
    fun aKnownTotalYieldsTheFraction() {
        assertEquals(0.25, job(bytesDone = 250, bytesTotal = 1_000).fraction!!, 1e-9)
        assertEquals(0.0, job(bytesDone = 0, bytesTotal = 1_000).fraction!!, 1e-9)
        assertEquals(1.0, job(bytesDone = 1_000, bytesTotal = 1_000).fraction!!, 1e-9)
    }

    /** Resume re-reports bytes already on disk, so done can briefly exceed the remaining total. */
    @Test
    fun overshootIsClampedRatherThanExceedingOne() {
        assertEquals(1.0, job(bytesDone = 1_200, bytesTotal = 1_000).fraction!!, 1e-9)
    }

    /**
     * `WaitingForConstraints` is the state that matters most and explains itself worst: with the
     * default `requireUnmetered = true` it means "waiting for Wi-Fi", an indefinite and entirely
     * normal wait. Treating it as terminal would report a queued pull as finished; treating it as
     * running would show a progress bar that never moves with no reason given.
     */
    @Test
    fun waitingForConstraintsIsNeitherRunningNorFinished() {
        val waiting = job(state = DownloadJob.State.WaitingForConstraints)

        assertFalse(waiting.isTerminal)
        assertEquals(DownloadJob.State.WaitingForConstraints, waiting.state)
    }

    @Test
    fun onlyTheEndStatesAreTerminal() {
        for (state in listOf(
            DownloadJob.State.Finished,
            DownloadJob.State.Failed,
            DownloadJob.State.Cancelled,
        )) {
            assertTrue("$state must be terminal", job(state = state).isTerminal)
        }
        for (state in listOf(
            DownloadJob.State.WaitingForConstraints,
            DownloadJob.State.Running,
            DownloadJob.State.Blocked,
        )) {
            assertFalse("$state must not be terminal", job(state = state).isTerminal)
        }
    }

    /** Every enum entry is classified, so a new state cannot silently default to "still going". */
    @Test
    fun everyStateIsAccountedFor() {
        assertEquals(6, DownloadJob.State.entries.size)
    }
}

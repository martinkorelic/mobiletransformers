package com.martinkorelic.mobiletransformers.internal.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/**
 * Pins the mask/position-ids invariant that the `transformers` 4.57.6 bump exposed.
 *
 * These assertions could not exist before: the planning lived inside `ORTGeneratorNative`, whose `init`
 * loads the native library, so the only way to reach a second conversation turn was a device run. That
 * is exactly why a wrong position id survived every host gate — the same shape as the C++ side, which
 * is why `plan_training_inputs` was pulled out of `train.cpp` for #33 B2.
 */
class GenerationInputsTest {

    @Test
    fun firstTurnStartsAtZero() {
        val plan = GenerationInputs.plan(intArrayOf(10, 11, 12), pastLength = 0)

        assertEquals(listOf(10L, 11L, 12L), plan.inputIds)
        assertEquals(listOf(1L, 1L, 1L), plan.attentionMask)
        assertEquals(listOf(0L, 1L, 2L), plan.positionIds)
    }

    @Test
    fun secondTurnContinuesFromTheCacheInsteadOfRestartingAtZero() {
        // The regression, stated directly: with 5 tokens cached, the next 3 are positions 5,6,7.
        // The old code produced 0,1,2 here while still claiming a mask of length 8.
        val plan = GenerationInputs.plan(intArrayOf(20, 21, 22), pastLength = 5)

        assertEquals(listOf(5L, 6L, 7L), plan.positionIds)
        assertEquals(8, plan.attentionMask.size)
    }

    @Test
    fun maskAndPositionsAlwaysAgreeAboutWhereTheSequenceEnds() {
        // The invariant across the seam, not either half of it: the last position the model is told
        // about must be the last slot the mask admits. Any off-by-N in either input breaks this.
        for (pastLength in 0..40 step 7) {
            for (newTokens in 1..6) {
                val plan = GenerationInputs.plan(IntArray(newTokens) { it }, pastLength)

                assertEquals(
                    "mask length must cover past + new (past=$pastLength, new=$newTokens)",
                    pastLength + newTokens,
                    plan.attentionMask.size,
                )
                assertEquals(
                    "last position must index the last mask slot (past=$pastLength, new=$newTokens)",
                    (plan.attentionMask.size - 1).toLong(),
                    plan.positionIds.last(),
                )
                assertEquals(
                    "positions must be contiguous (past=$pastLength, new=$newTokens)",
                    (pastLength until pastLength + newTokens).map { it.toLong() },
                    plan.positionIds,
                )
            }
        }
    }

    @Test
    fun positionIdsAreOneEntryPerNewTokenNotPerCachedToken() {
        // The cached prefix is NOT re-fed; only the delta gets a position.
        val plan = GenerationInputs.plan(intArrayOf(99), pastLength = 12)

        assertEquals(1, plan.inputIds.size)
        assertEquals(1, plan.positionIds.size)
        assertEquals(listOf(12L), plan.positionIds)
    }

    @Test
    fun aNegativeCacheLengthFailsClosedRatherThanBindingANonsenseMask() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            GenerationInputs.plan(intArrayOf(1, 2), pastLength = -1)
        }
        assertEquals(true, error.message!!.contains("-1"))
    }
}

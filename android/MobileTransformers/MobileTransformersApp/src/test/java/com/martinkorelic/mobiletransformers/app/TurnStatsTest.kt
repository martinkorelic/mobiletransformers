package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.viewmodels.TurnStats
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The per-turn line under an assistant message.
 *
 * Chat reported `"N tokens · X tok/s"` and nothing about the window, so the number that predicts the
 * next turn being truncated was the one number missing. Context used is prompt **plus** completion:
 * reporting only the completion understates it by the whole conversation so far, which is precisely
 * the part that grows.
 */
class TurnStatsTest {

    private fun result(
        tokens: Int = 37,
        rate: Double = 4.2,
        promptTokens: Int = 475,
        limit: Int = 32768,
    ) = GenerationResult(
        text = "hello",
        tokenCount = tokens,
        avgTokensPerSecond = rate,
        promptTokenCount = promptTokens,
        contextLimit = limit,
    )

    @Test
    fun contextUsedIsPromptPlusCompletion() {
        assertEquals(512, result().contextUsedTokens)
    }

    @Test
    fun theLineCarriesSpeedAndWindow() {
        val line = TurnStats.of(result())!!.render()

        assertTrue(line, line.contains("37 tokens"))
        assertTrue(line, line.contains("4.2 tok/s"))
        assertTrue("the window is the point of the change: $line", line.contains("32,768"))
        assertTrue("and how full it is: $line", line.contains("2%"))
    }

    @Test
    fun aPackageDeclaringNoContextLimitStillReportsWhatItKnows() {
        // contextLimit is 0 for a package whose tokenizer config declares no model_max_length.
        // Inventing a denominator would be worse than omitting the percentage.
        val line = TurnStats.of(result(limit = 0))!!.render()

        assertTrue(line, line.contains("512 tokens"))
        assertFalse("no percentage without a limit: $line", line.contains("%"))
    }

    @Test
    fun anUnmeasuredRateIsOmittedRatherThanShownAsZero() {
        val line = TurnStats.of(result(rate = 0.0))!!.render()

        assertFalse("0.0 tok/s is a measurement that was not taken, not a speed: $line", line.contains("tok/s"))
    }

    @Test
    fun thereAreNoStatsWhenThereWasNoGeneration() {
        assertNull(TurnStats.of(null as GenerationResult?))
    }

    @Test
    fun aNearlyFullWindowIsVisible() {
        // The case the line exists for.
        val line = TurnStats.of(result(promptTokens = 31_000, tokens = 500, limit = 32_768))!!.render()

        assertTrue(line, line.contains("96%"))
    }
}

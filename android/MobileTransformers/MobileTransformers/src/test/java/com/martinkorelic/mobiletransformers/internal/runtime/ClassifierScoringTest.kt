package com.martinkorelic.mobiletransformers.internal.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #33: turning a classification head's logits into named, ranked labels.
 *
 * The forward pass itself needs a device and a classification package, so it is device-gated. This is
 * everything that happens *after* it, which is where a wrong answer would be silent rather than loud:
 * a mis-scaled softmax or an off-by-one label mapping produces a confident, plausible, wrong class.
 */
class ClassifierScoringTest {

    private val labels = mapOf(0 to "unacceptable", 1 to "acceptable")

    @Test
    fun scoresAreAProbabilityDistributionOverTheLabels() {
        val scores = ClassifierSession.softmaxToLabels(floatArrayOf(1.0f, 2.0f), labels)
        assertEquals(2, scores.size)
        assertEquals(1.0, scores.sumOf { it.score }, 1e-9)
        assertEquals("acceptable", scores.first().label)
    }

    @Test
    fun theRankingIsHighestFirstAndKeepsEachLabelsOwnIndex() {
        val scores = ClassifierSession.softmaxToLabels(floatArrayOf(5.0f, 0.5f), labels)
        assertEquals(listOf("unacceptable", "acceptable"), scores.map { it.label })
        // The index travels with the label: a name need not be unique or stable across exports, so a
        // caller comparing predictions between runs has to compare something that is.
        assertEquals(0, scores.first().index)
        assertEquals(1, scores.last().index)
    }

    /**
     * The reason for subtracting the max.
     *
     * Classification logits routinely exceed 80, and `exp(80f)` overflows a Float to infinity — so the
     * naive softmax returns NaN for exactly the confident predictions it matters most to report. This
     * is the case that would ship looking fine on a toy fixture and fail on a real head.
     */
    @Test
    fun aLargeLogitDoesNotOverflowToNaN() {
        val scores = ClassifierSession.softmaxToLabels(floatArrayOf(120.0f, 3.0f), labels)
        assertTrue("softmax overflowed", scores.all { it.score.isFinite() })
        assertEquals(1.0, scores.sumOf { it.score }, 1e-9)
        assertEquals("unacceptable", scores.first().label)
        assertEquals(1.0, scores.first().score, 1e-6)
    }

    @Test
    fun equalLogitsSplitEvenly() {
        val scores = ClassifierSession.softmaxToLabels(floatArrayOf(2.0f, 2.0f), labels)
        assertEquals(0.5, scores[0].score, 1e-9)
        assertEquals(0.5, scores[1].score, 1e-9)
    }

    /**
     * A graph that emits more values than the package names labels for.
     *
     * Reading past the declared labels would invent classes; the extra logits are dropped and the
     * distribution is normalised over what is actually named.
     */
    @Test
    fun extraLogitsBeyondTheDeclaredLabelsAreIgnored() {
        val scores = ClassifierSession.softmaxToLabels(floatArrayOf(1f, 2f, 9f, 9f), labels)
        assertEquals(2, scores.size)
        assertEquals(1.0, scores.sumOf { it.score }, 1e-9)
        assertEquals("acceptable", scores.first().label)
    }

    /** A gap in `id2label` falls back to the index rather than dropping the class silently. */
    @Test
    fun anUnnamedClassStillAppears() {
        val scores = ClassifierSession.softmaxToLabels(floatArrayOf(0f, 5f), mapOf(0 to "yes", 2 to "maybe"))
        assertEquals(setOf("yes", "LABEL_1"), scores.map { it.label }.toSet())
    }

    @Test
    fun noLabelsMeansNoScoresRatherThanAnIndexPretendingToBeAnAnswer() {
        assertEquals(emptyList<Any>(), ClassifierSession.softmaxToLabels(floatArrayOf(1f, 2f), emptyMap()))
    }
}

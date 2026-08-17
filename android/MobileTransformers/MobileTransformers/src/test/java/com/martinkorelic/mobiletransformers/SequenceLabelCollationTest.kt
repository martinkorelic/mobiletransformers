package com.martinkorelic.mobiletransformers

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * #33: per-SEQUENCE labels survive collation unpadded, which is what makes the native rank inference
 * work.
 *
 * The seam these pin: `training_inputs.h::labels_shape` decides `[batch, seq]` vs `[batch]` from **how
 * many label elements the caller supplied**. So the collator is the half that determines the exported
 * graph's label rank, and padding a single class index out to `maxLength` silently converts a
 * classification batch into a per-token one. Both halves were correct alone; only the seam decides.
 */
class SequenceLabelCollationTest {

    private fun sample(inputIds: List<Int>, labels: List<Int>, perSequence: Boolean) =
        ORTDataCurator.TrainingSample(inputIds = inputIds, labels = labels, perSequenceLabel = perSequence)

    @Test
    fun perSequenceLabelsAreNotPaddedToTheSequenceLength() {
        val batch = listOf(
            sample(listOf(1, 2, 3, 4, 5), listOf(1), perSequence = true),
            sample(listOf(6, 7), listOf(0), perSequence = true),
        )

        val collated = DataCollatorForSupervisedDataset(padToken = 0).collate(batch)

        // One label per example. `batchSize` elements total is exactly what makes the native side bind
        // `labels[batch]`; padding to maxLength (5) would yield batch*seq and bind `[batch, seq]`.
        assertEquals(2, collated.labels.size)
        assertEquals(1, collated.labels[0].size)
        assertEquals(1, collated.labels[1].size)
        assertEquals(listOf(1L, 0L), collated.labels.map { it[0] })
        // Inputs are still padded — only the LABEL axis differs between the two objectives.
        assertEquals(5, collated.sequenceLength)
        assertEquals(5, collated.inputIds[1].size)
    }

    @Test
    fun perTokenLabelsAreStillPaddedExactlyAsBefore() {
        val batch = listOf(
            sample(listOf(1, 2, 3, 4, 5), listOf(-100, -100, 3, 4, 5), perSequence = false),
            sample(listOf(6, 7), listOf(-100, 7), perSequence = false),
        )

        val collated = DataCollatorForSupervisedDataset(padToken = 0).collate(batch)

        // The decoder's shipped shape is unchanged: one label per position, short rows padded to -100.
        assertEquals(5, collated.labels[0].size)
        assertEquals(5, collated.labels[1].size)
        assertEquals(listOf(-100L, 7L, -100L, -100L, -100L), collated.labels[1].toList())
    }

    @Test
    fun theTotalLabelCountIsWhatTheNativeBinderReadsTheRankFrom() {
        val perSequence = listOf(
            sample(listOf(1, 2, 3), listOf(1), perSequence = true),
            sample(listOf(4, 5, 6), listOf(0), perSequence = true),
        )
        val perToken = listOf(
            sample(listOf(1, 2, 3), listOf(1, 2, 3), perSequence = false),
            sample(listOf(4, 5, 6), listOf(4, 5, 6), perSequence = false),
        )
        val collator = DataCollatorForSupervisedDataset(padToken = 0)

        val seqBatch = collator.collate(perSequence)
        val tokBatch = collator.collate(perToken)

        // `batch` vs `batch * seq` — the two counts the native side distinguishes.
        assertEquals(seqBatch.batchSize, seqBatch.labels.sumOf { it.size })
        assertEquals(tokBatch.batchSize * tokBatch.sequenceLength, tokBatch.labels.sumOf { it.size })
    }
}

package com.martinkorelic.mobiletransformers.rag

import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Where a retrieved passage came from, at the public boundary.
 *
 * A match is a **chunk**: ingestion splits each file into pieces and stores every piece as its own
 * row, keeping the file's name as the title and `<docId>#<n>` as the id. Both survived into the
 * vector store and were then dropped by the mapping into `RetrievalMatch`, so a caller could show
 * the retrieved text and could not say what it was retrieved *from* — and "found in 2 documents"
 * was not derivable at all, because two chunks of one file were indistinguishable from two files.
 *
 * The grouping asserted here is what the Chat screen's retrieval report is built on.
 */
class RetrievalProvenanceTest {

    private fun match(text: String, score: Double, title: String, chunkId: String) =
        RetrievalMatch(text = text, score = score, title = title, chunkId = chunkId)

    @Test
    fun chunksOfOneFileCountAsOneDocument() {
        val result = RetrievalResult(
            matches = listOf(
                match("batching halves the wall clock", 0.81, "notes.md", "notes#3"),
                match("a batch of 4 fits in memory", 0.74, "notes.md", "notes#7"),
                match("the encoder is all-MiniLM", 0.52, "setup.txt", "setup#0"),
            ),
        )

        assertEquals(2, result.documentCount)
        // Order follows the ranking, so the best-scoring source is named first.
        assertEquals(listOf("notes.md", "setup.txt"), result.documentTitles)
    }

    @Test
    fun theDocumentIdIsTheChunkIdWithoutItsIndex() {
        assertEquals("notes", match("t", 1.0, "notes.md", "notes#3").documentId)
        // A JSONL record may name itself anything, including something containing a '#'. Only the
        // LAST '#' is the chunk index, so splitting on the first would truncate the real id.
        assertEquals("issue#42", match("t", 1.0, "bugs.jsonl", "issue#42#1").documentId)
        // No index at all: the whole id is the document.
        assertEquals("whole", match("t", 1.0, "whole.txt", "whole").documentId)
    }

    @Test
    fun anUnattributedHitIsNeverFoldedIntoAnotherDocument() {
        // Defaults, i.e. a hit from a store written before provenance was carried. Claiming these
        // are "the same document" would be an invention; each counts for itself.
        val result = RetrievalResult(
            matches = listOf(
                RetrievalMatch("first", 0.9),
                RetrievalMatch("second", 0.8),
            ),
        )

        assertEquals(2, result.documentCount)
        assertTrue(result.documentTitles.isEmpty())
    }

    @Test
    fun attributedAndUnattributedHitsBothCount() {
        val result = RetrievalResult(
            matches = listOf(
                match("a", 0.9, "notes.md", "notes#0"),
                match("b", 0.8, "notes.md", "notes#1"),
                RetrievalMatch("c", 0.7),
            ),
        )

        assertEquals(2, result.documentCount)
        assertEquals(listOf("notes.md"), result.documentTitles)
    }

    @Test
    fun anEmptyResultClaimsNothing() {
        assertEquals(0, RetrievalResult().documentCount)
        assertTrue(RetrievalResult().documentTitles.isEmpty())
    }
}

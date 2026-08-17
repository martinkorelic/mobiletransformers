package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.viewmodels.RetrievalCard
import com.martinkorelic.mobiletransformers.app.viewmodels.SourceCard
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The line the retrieval turn leads with, before the answer it produced.
 *
 * It is the whole claim of that message — everything else is behind a "Show passages" toggle — so it
 * has to be accurate about the two numbers that differ: passages are chunks, documents are files,
 * and one file routinely contributes several chunks.
 */
class RetrievalCardTest {

    private fun card(vararg passages: Pair<String, String>) = RetrievalCard(
        passages = passages.map { (title, text) -> SourceCard(text, 0.5, title) },
        documents = passages.map { it.first }.filter { it.isNotBlank() }.distinct(),
    )

    @Test
    fun itCountsChunksAndFilesSeparately() {
        val c = card("notes.md" to "one", "notes.md" to "two", "setup.txt" to "three")
        assertEquals("Found 3 passages in 2 documents", c.headline)
    }

    @Test
    fun singularsReadAsSentences() {
        assertEquals("Found 1 passage in 1 document", card("notes.md" to "only").headline)
    }

    @Test
    fun nothingRetrievedSaysWhatThatMeansForTheAnswer() {
        // "0 passages" is a number; what the reader needs is that the answer about to appear is not
        // grounded in anything, which is the difference between a wrong answer and an unsupported one.
        assertEquals(
            "No matching passages — the answer will be ungrounded",
            RetrievalCard(passages = emptyList(), documents = emptyList()).headline,
        )
    }

    @Test
    fun unattributedPassagesDropTheDocumentClauseRatherThanInventOne() {
        val c = RetrievalCard(
            passages = listOf(SourceCard("a", 0.5), SourceCard("b", 0.4)),
            documents = emptyList(),
        )
        assertEquals("Found 2 passages", c.headline)
    }
}

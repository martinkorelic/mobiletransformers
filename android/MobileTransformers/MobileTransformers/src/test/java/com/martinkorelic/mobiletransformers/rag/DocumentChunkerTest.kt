package com.martinkorelic.mobiletransformers.rag

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** #26: deterministic character-based chunking. */
class DocumentChunkerTest {

    @Test
    fun exactWindowsForSizeAndOverlap() {
        val text = (0 until 100).joinToString("") { (it % 10).toString() } // 100 chars
        val chunks = DocumentChunker.split(text, chunkSize = 40, chunkOverlap = 10)
        // stride = 30 -> windows [0,40) [30,70) [60,100)
        assertEquals(3, chunks.size)
        assertEquals(text.substring(0, 40), chunks[0])
        assertEquals(text.substring(30, 70), chunks[1])
        assertEquals(text.substring(60, 100), chunks[2])
        // last window reaches the end (no gap)
        assertTrue(chunks.last().endsWith(text.takeLast(1)))
    }

    @Test
    fun singleChunkWhenShorterThanSize() {
        assertEquals(listOf("hello"), DocumentChunker.split("hello", chunkSize = 40, chunkOverlap = 10))
    }

    @Test
    fun emptyTextYieldsNoChunks() {
        assertEquals(emptyList<String>(), DocumentChunker.split("", chunkSize = 40, chunkOverlap = 10))
    }

    @Test
    fun overlapEqualToOrGreaterThanSizeRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            DocumentChunker.split("abcdef", chunkSize = 10, chunkOverlap = 10)
        }
        assertThrows(IllegalArgumentException::class.java) {
            DocumentChunker.split("abcdef", chunkSize = 10, chunkOverlap = 20)
        }
    }

    @Test
    fun zeroSizeRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            DocumentChunker.split("abc", chunkSize = 0, chunkOverlap = 0)
        }
    }

    @Test
    fun noOverlapTilesExactly() {
        val text = "abcdefghij" // 10
        val chunks = DocumentChunker.split(text, chunkSize = 5, chunkOverlap = 0)
        assertEquals(listOf("abcde", "fghij"), chunks)
    }
}

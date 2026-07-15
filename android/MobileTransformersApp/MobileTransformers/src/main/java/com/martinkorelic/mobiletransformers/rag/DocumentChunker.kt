package com.martinkorelic.mobiletransformers.rag

/**
 * #26: pure, deterministic **character-based** text chunker (no Android/JNI deps → JVM-testable).
 *
 * Windows of [chunkSize] characters advance by `chunkSize - chunkOverlap`. The last window is clamped to
 * the end of the text (no gap, no out-of-range). `chunkOverlap` counts characters, not tokens.
 */
object DocumentChunker {
    fun split(text: String, chunkSize: Int, chunkOverlap: Int): List<String> {
        require(chunkSize > 0) { "chunkSize must be > 0, got $chunkSize" }
        require(chunkOverlap in 0 until chunkSize) {
            "chunkOverlap must be in 0 until chunkSize ($chunkSize), got $chunkOverlap"
        }
        if (text.isEmpty()) return emptyList()
        if (text.length <= chunkSize) return listOf(text)

        val stride = chunkSize - chunkOverlap
        val chunks = ArrayList<String>()
        var start = 0
        while (start < text.length) {
            val end = minOf(start + chunkSize, text.length)
            chunks.add(text.substring(start, end))
            if (end == text.length) break
            start += stride
        }
        return chunks
    }
}

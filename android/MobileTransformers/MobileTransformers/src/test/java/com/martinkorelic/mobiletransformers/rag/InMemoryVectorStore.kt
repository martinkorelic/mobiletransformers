package com.martinkorelic.mobiletransformers.rag

import kotlin.math.sqrt

/**
 * Pure-Kotlin [VectorStore] for JVM tests — no Android, no ObjectBox (#25). Mirrors the ObjectBox
 * semantics: cosine similarity ordering (ObjectBox returns COSINE distance and the store converts to
 * similarity = `1 - distance`; direct cosine similarity here is the algebraically identical result),
 * `minScore` filtering on similarity, and a separate non-ranked text-search path ([TEXT_SEARCH_SCORE]).
 * Constructing with an unsupported dimension fails closed via [DimensionRegistry].
 */
class InMemoryVectorStore(private val dimension: Int) : VectorStore {

    init {
        DimensionRegistry.requireSupported(dimension)
    }

    private data class Row(val id: Long, val document: RagDocument, val embedding: FloatArray)

    private val rows = mutableListOf<Row>()
    private var nextId = 1L

    override fun insert(document: RagDocument, embedding: FloatArray): Long {
        require(embedding.size == dimension) {
            "embedding size ${embedding.size} doesn't match store dimension $dimension"
        }
        val id = nextId++
        rows.add(Row(id, document, embedding.copyOf()))
        return id
    }

    override fun search(queryEmbedding: FloatArray, topK: Int, minScore: Double): List<RagMatch> {
        require(queryEmbedding.size == dimension) {
            "query embedding size ${queryEmbedding.size} doesn't match store dimension $dimension"
        }
        return rows
            .map { RagMatch(it.document, cosineSimilarity(queryEmbedding, it.embedding)) }
            .filter { it.score >= minScore }
            .sortedByDescending { it.score }
            .take(topK)
    }

    override fun textSearch(query: String, topK: Int): List<RagMatch> =
        rows.filter { it.document.text.contains(query, ignoreCase = true) }
            .map { RagMatch(it.document, TEXT_SEARCH_SCORE) }
            .take(topK)

    override fun count(): Long = rows.size.toLong()

    override fun close() {
        rows.clear()
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i].toDouble() * b[i].toDouble()
            na += a[i].toDouble() * a[i].toDouble()
            nb += b[i].toDouble() * b[i].toDouble()
        }
        if (na == 0.0 || nb == 0.0) return 0.0
        return dot / (sqrt(na) * sqrt(nb))
    }
}

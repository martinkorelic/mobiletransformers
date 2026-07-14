package com.martinkorelic.mobiletransformers.rag

/**
 * A small, testable boundary around on-device vector search (#25, 03_code_plans/03).
 *
 * ObjectBox is the default backing store on-device ([ObjectBoxVectorStore]); [VectorStore] lets
 * chunking / ingestion / retrieval be unit-tested on the JVM with no Android/ObjectBox via a pure
 * `InMemoryVectorStore` (test source set). Callers depend on this interface, not on ObjectBox.
 *
 * Score semantics (preserved from `ORTVectorDatabase`): ObjectBox HNSW uses COSINE **distance**, and
 * similarity = `1 - distance` (`ORTVectorDatabase.kt:262`). [RagMatch.score] always carries the
 * **similarity** (post-conversion, higher = closer), so callers never re-convert. `minScore` filters
 * on that similarity. Text search is a separate, non-ranked path (see [TEXT_SEARCH_SCORE]).
 */

/** A stored document. `text` is the searchable/embeddable body; `id` identifies the source. */
data class RagDocument(
    val id: String,
    val title: String,
    val text: String,
    val metadata: Map<String, String> = emptyMap(),
)

/** A retrieval hit. `score` is the similarity (already `1 - distance`), higher = closer. */
data class RagMatch(val document: RagDocument, val score: Double)

/**
 * Fixed similarity assigned to text-search hits. Text matches are a substring/VALUE-index lookup
 * (`ORTVectorDatabase.queryByContent`) and are **not** similarity-ranked — they all carry this score.
 */
const val TEXT_SEARCH_SCORE: Double = 1.0

interface VectorStore {
    /** Insert `document` with its `embedding`; returns the assigned row id (or a negative id on failure). */
    fun insert(document: RagDocument, embedding: FloatArray): Long

    /** Top-`topK` nearest documents by cosine similarity, keeping only hits with similarity >= `minScore`. */
    fun search(queryEmbedding: FloatArray, topK: Int, minScore: Double = 0.0): List<RagMatch>

    /** Non-ranked substring/content lookup; every hit carries [TEXT_SEARCH_SCORE]. */
    fun textSearch(query: String, topK: Int): List<RagMatch>

    fun count(): Long

    fun close()
}

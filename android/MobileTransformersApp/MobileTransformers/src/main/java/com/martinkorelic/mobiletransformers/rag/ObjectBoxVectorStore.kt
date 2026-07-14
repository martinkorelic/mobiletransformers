package com.martinkorelic.mobiletransformers.rag

import com.martinkorelic.mobiletransformers.ORTVectorDatabase
import com.martinkorelic.mobiletransformers.entity.VectorEntityInterface

/**
 * The default on-device [VectorStore] — a thin wrapper over [ORTVectorDatabase] that preserves its
 * exact semantics: COSINE distance, `1 - distance` similarity (already applied by `queryDocuments`),
 * `minScore` filtering, embedding vectors stripped from results, and the separate text-search path.
 * It only reshapes the store's `Pair<VectorEntityInterface, Double>` results into [RagMatch].
 */
class ObjectBoxVectorStore(private val db: ORTVectorDatabase) : VectorStore {

    init {
        DimensionRegistry.requireSupported(db.ortRagConfig.embeddingDimension)
    }

    override fun insert(document: RagDocument, embedding: FloatArray): Long =
        db.insertVector(
            name = document.title,
            embedding = embedding,
            content = document.text,
            document = document.id,
            metadata = encodeMetadata(document.metadata),
        )

    override fun search(queryEmbedding: FloatArray, topK: Int, minScore: Double): List<RagMatch> =
        // queryDocuments already returns similarity (1 - distance) and applies minScore.
        db.queryDocuments(queryEmbedding, topK, minScore).map { (entity, similarity) ->
            RagMatch(entity.toRagDocument(), similarity)
        }

    override fun textSearch(query: String, topK: Int): List<RagMatch> =
        db.queryByContent(query, topK.toLong()).map { RagMatch(it.toRagDocument(), TEXT_SEARCH_SCORE) }

    override fun count(): Long = db.getVectorCount()

    override fun close() = db.close()
}

/** Results carry document + similarity only; the embedding vector is already stripped by the store. */
private fun VectorEntityInterface.toRagDocument(): RagDocument =
    RagDocument(
        id = document,
        title = name,
        text = content,
        metadata = if (metadata.isBlank()) emptyMap() else mapOf("metadata" to metadata),
    )

private fun encodeMetadata(metadata: Map<String, String>): String =
    metadata["metadata"] ?: metadata.entries.joinToString(";") { "${it.key}=${it.value}" }

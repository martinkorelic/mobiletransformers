package com.martinkorelic.mobiletransformers.rag

import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive

/**
 * #26: the pure ingestion loop — chunk → embed → insert — with an **injectable embedder** so it is
 * JVM-unit-testable with a fake embedder + `InMemoryVectorStore` (the real path is JNI-bound). Cooperative
 * cancellation between chunks/documents; a per-document failure is reported via [IngestionProgress.onError]
 * and skipped (cancellation always propagates). Returns the number of chunks inserted.
 */
object IngestionPipeline {
    suspend fun ingest(
        documents: List<RagDocument>,
        chunkSize: Int,
        chunkOverlap: Int,
        embed: (String) -> FloatArray?,
        store: VectorStore,
        progress: IngestionProgress? = null,
    ): Int {
        var inserted = 0
        for (record in documents) {
            coroutineContext.ensureActive()
            progress?.onDocumentStart(record.id, documents.size)
            try {
                val chunks = DocumentChunker.split(record.text, chunkSize, chunkOverlap)
                chunks.forEachIndexed { i, chunk ->
                    coroutineContext.ensureActive()
                    val embedding =
                        embed(chunk) ?: throw IllegalStateException("embedding failed for ${record.id} chunk $i")
                    store.insert(
                        RagDocument("${record.id}#$i", record.title, chunk, record.metadata),
                        embedding,
                    )
                    inserted++
                    progress?.onChunkEmbedded(record.id, i, chunks.size)
                }
                progress?.onDocumentComplete(record.id)
            } catch (ce: CancellationException) {
                throw ce
            } catch (e: Throwable) {
                progress?.onError(record.id, e)
            }
        }
        return inserted
    }
}

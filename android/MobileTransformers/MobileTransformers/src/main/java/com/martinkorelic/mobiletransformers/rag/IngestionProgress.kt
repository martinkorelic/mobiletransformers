package com.martinkorelic.mobiletransformers.rag

/**
 * #26: progress callback for document ingestion (chunk → embed → store). All methods optional; carries
 * only neutral types so it is safe on the public facade surface.
 */
interface IngestionProgress {
    fun onDocumentStart(id: String, totalDocs: Int) {}

    fun onChunkEmbedded(docId: String, chunkIndex: Int, totalChunks: Int) {}

    fun onDocumentComplete(id: String) {}

    fun onError(id: String?, error: Throwable) {}
}

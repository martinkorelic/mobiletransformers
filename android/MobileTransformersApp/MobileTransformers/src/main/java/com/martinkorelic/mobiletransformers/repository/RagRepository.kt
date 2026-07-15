package com.martinkorelic.mobiletransformers.repository

import android.util.Log
import com.martinkorelic.mobiletransformers.ORTRagArguments
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.loadDocuments

class RagRepository(private val llmRepository: LLMRepository) {

    private val LOG_TAG = "RagRepository"

    /**
     * #26: ingest a `.txt`/`.md`/`.jsonl` file into the vector store (chunk → embed → insert). Owner of
     * ingestion; [loadDocuments] resolves the loader (F3). Returns the number of chunks inserted.
     */
    suspend fun ingest(
        path: String,
        ragConfig: ORTRagConfig? = null,
        progress: IngestionProgress? = null,
    ): Int {
        initialize(ragConfig)
        val retriever = llmRepository.ortRetriever
        if (retriever == null) {
            Log.e(LOG_TAG, "ORTRetriever is not set; cannot ingest.")
            return 0
        }
        return retriever.ingestData(loadDocuments(path), progress)
    }

    suspend fun initialize(
        ragConfig: ORTRagConfig? = null,
        ragCallback: RagCallback? = null
    ) {
        if (ragCallback != null) llmRepository.ragCallback = ragCallback

        if (llmRepository.ortRetriever == null) {
            llmRepository.ragCallback?.onModelLoadStart()
            val job = llmRepository.prepareRetriever(ragConfig)
            job.join()
            llmRepository.ragCallback?.onModelLoadEnd()
        }

    }

    suspend fun query(prompt : String, ragConfig: ORTRagArguments? = null, ragCallback: RagCallback? = null) {

        if (llmRepository.ortRetriever == null) {
            Log.e(LOG_TAG, "ORTRetriever is not currently set or does not exist.")
            return
        }

        // Update RAG callback
        if (ragCallback != null) llmRepository.ragCallback = ragCallback

        // Run the retriever
        val job = llmRepository.runRetriever(prompt, ragConfig)
        job.join()
    }
}
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

    /**
     * Ensure a retriever exists AND that it is running [ragConfig].
     *
     * #27 fix: this used to short-circuit entirely once a retriever existed, so only the FIRST
     * config of a session ever took effect — a later `retrieve`/`generateWithRag` with a changed
     * `topK`/`minScore`/`searchType` was silently ignored. A changed config now always applies:
     * query-shaping fields are pushed onto the live retriever, while a change to the embedding
     * model's identity ([requiresReload]) rebuilds it, since assigning those alone would leave a
     * stale embedding session loaded.
     */
    suspend fun initialize(
        ragConfig: ORTRagConfig? = null,
        ragCallback: RagCallback? = null
    ) {
        if (ragCallback != null) llmRepository.ragCallback = ragCallback

        val existing = llmRepository.ortRetriever
        if (existing == null || (ragConfig != null && requiresReload(existing.ragConfig, ragConfig))) {
            llmRepository.ragCallback?.onModelLoadStart()
            val job = llmRepository.prepareRetriever(ragConfig)
            job.join()
            llmRepository.ragCallback?.onModelLoadEnd()
            return
        }

        // Same embedding model, possibly different query shaping — the LLMRepository setter fans the
        // new config out to the live retriever.
        if (ragConfig != null && ragConfig != existing.ragConfig) {
            llmRepository.ragConfig = ragConfig
        }
    }

    /**
     * True when [next] selects a different embedding model (or device placement) than [current], and
     * the retriever must therefore be rebuilt rather than reconfigured in place. Query-shaping fields
     * (`topK`, `minScore`, `searchType`, `indexingMode`, chunking) deliberately do NOT appear here.
     */
    private fun requiresReload(current: ORTRagConfig, next: ORTRagConfig): Boolean =
        current.repoName != next.repoName ||
            // ORTRetriever.createEmbeddingModel appends ".onnx" in place, so a loaded config's
            // onnxName has the suffix while a freshly-mapped one does not. Compare normalized, or
            // every single retrieve would look like a model change and rebuild the session.
            normalizeOnnxName(current.onnxName) != normalizeOnnxName(next.onnxName) ||
            current.embeddingDimension != next.embeddingDimension ||
            current.deviceOptions != next.deviceOptions

    private fun normalizeOnnxName(name: String): String = name.removeSuffix(".onnx")

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
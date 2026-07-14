package com.martinkorelic.mobiletransformers.repository

import android.util.Log
import com.martinkorelic.mobiletransformers.ORTRagArguments
import com.martinkorelic.mobiletransformers.ORTRagConfig

class RagRepository(private val llmRepository: LLMRepository) {

    private val LOG_TAG = "RagRepository"

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
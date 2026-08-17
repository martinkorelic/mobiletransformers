package com.martinkorelic.mobiletransformers.repository

import com.martinkorelic.mobiletransformers.ORTGenerationConfig

class InferenceRepository(private val llmRepository: LLMRepository) {

    suspend fun generate(
        userMessage: String,
        generationConfig: ORTGenerationConfig? = null,
        callback: GenerationCallback? = null
    ) {

        if (callback != null) llmRepository.generationCallback = callback

        if (llmRepository.modelRuntime == null) {
            llmRepository.generationCallback?.onModelLoadStart()
            val job = llmRepository.prepareGeneration(generationConfig)
            job.join()
            llmRepository.generationCallback?.onModelLoadEnd()
        }

        llmRepository.runGenerationStream(userMessage, generationConfig)
    }

    suspend fun reloadSession(generationConfig: ORTGenerationConfig? = null, callback: GenerationCallback? = null) {

        if (callback != null) llmRepository.generationCallback = callback

        if (llmRepository.llmState != LLMState.NotInitialized) {
            llmRepository.resetInference()
        }

        if (llmRepository.modelRuntime == null) {
            llmRepository.generationCallback?.onModelLoadStart()
            val job = llmRepository.prepareGeneration(generationConfig)
            job.join()
            llmRepository.generationCallback?.onModelLoadEnd()
        }

    }
}

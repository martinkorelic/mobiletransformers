package com.martinkorelic.ortmobile.repository

import com.martinkorelic.ortmobile.ORTGenerationConfig

class InferenceRepository(private val llmRepository: LLMRepository) {

    suspend fun generate(
        userMessage: String,
        generationConfig: ORTGenerationConfig? = null,
        callback: GenerationCallback? = null
    ) {

        if (callback != null) llmRepository.generationCallback = callback

        if (llmRepository.ortNativeInference == null) {
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

        if (llmRepository.ortNativeInference == null) {
            llmRepository.generationCallback?.onModelLoadStart()
            val job = llmRepository.prepareGeneration(generationConfig)
            job.join()
            llmRepository.generationCallback?.onModelLoadEnd()
        }

    }
}

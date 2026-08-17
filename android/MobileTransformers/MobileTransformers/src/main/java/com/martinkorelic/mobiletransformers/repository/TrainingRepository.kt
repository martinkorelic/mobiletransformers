package com.martinkorelic.mobiletransformers.repository

import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.TaskPreprocessor
import org.json.JSONObject

class TrainingRepository(private val llmRepository: LLMRepository) {

    suspend fun performTraining(trainingConfig: ORTTrainingConfig? = null, trainingCallback : TrainingCallback? = null, dataPreprocessFunction: TaskPreprocessor? = null) {

        if (trainingCallback != null) llmRepository.trainingCallback = trainingCallback

        if (llmRepository.llmState != LLMState.Training && llmRepository.llmState != LLMState.ReadyTrain) {
            llmRepository.trainingCallback?.onModelLoadStart()
            val job = llmRepository.prepareTraining(trainingConfig, dataPreprocessFunction)
            job.join()
            // `join()` never rethrows, and `prepareTraining` deliberately swallows so the failure
            // cannot reach an uncaught handler. Re-raise it here, on the caller's own coroutine,
            // which is the first frame that can both see it and report it. Without this a setup
            // failure read as "runTraining says the model is not ready" — a symptom, on a later line.
            llmRepository.consumeTrainingSessionFailure()?.let { throw it }
            llmRepository.trainingCallback?.onModelLoadEnd()
        }

        val job = llmRepository.runTraining()
        job?.join()
    }

    suspend fun endTraining(saveModel : Boolean) {

        if (llmRepository.llmState != LLMState.Training) {
            val job = llmRepository.saveTraining(saveModel)
            job?.join()
        }
    }

    suspend fun reloadSession(trainingConfig: ORTTrainingConfig? = null, trainingCallback : TrainingCallback? = null, dataPreprocessFunction: TaskPreprocessor? = null) {
        if (trainingCallback != null) llmRepository.trainingCallback = trainingCallback

        if (llmRepository.llmState != LLMState.NotInitialized) {
            llmRepository.resetTraining()
        }

        if (llmRepository.llmState != LLMState.Training && llmRepository.llmState != LLMState.ReadyTrain) {
            llmRepository.trainingCallback?.onModelLoadStart()
            val job = llmRepository.prepareTraining(trainingConfig, dataPreprocessFunction)
            job.join()
            llmRepository.trainingCallback?.onModelLoadEnd()
        }
    }
}
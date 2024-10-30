package com.example.orttransformer.repository

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

enum class TrainingUiState {
    Training,
    FinishedTraining,
    ReadyTrain,
    SavingModel,
    FinishedSavingModel
}

class TrainingRepository(private val llmRepository: LLMRepository) {
    private val _isTraining = MutableStateFlow(TrainingUiState.ReadyTrain)
    val isTraining: MutableStateFlow<TrainingUiState> = _isTraining

    private val _loss = MutableStateFlow<Float>(-1.0F)
    val loss: MutableStateFlow<Float> = _loss

    init {
        CoroutineScope(Dispatchers.IO).launch {
            llmRepository.lossFlow.collect { loss ->
                _loss.value = loss
            }
        }
    }

    suspend fun performTraining(trainData: List<String>) {
        _isTraining.value = TrainingUiState.Training

        if (llmRepository.llmState != LLMState.Training && llmRepository.llmState != LLMState.ReadyTrain) {
            val job = llmRepository.prepareTraining()
            job.join()
        }

        val job = llmRepository.runTraining(trainData)
        job?.join()

        _isTraining.value = TrainingUiState.FinishedTraining
    }

    suspend fun endTraining(saveModel : Boolean) {
        _isTraining.value = TrainingUiState.SavingModel

        if (llmRepository.llmState != LLMState.Training) {
            val job = llmRepository.saveTraining(saveModel)
            job?.join()
        }

        _isTraining.value = TrainingUiState.FinishedSavingModel
    }
}

package com.example.orttransformer.repository

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch


class TrainingRepository(private val llmRepository: LLMRepository) {
    private val _isTraining = MutableStateFlow(false)
    val isTraining: MutableStateFlow<Boolean> = _isTraining

    private val _loss = MutableStateFlow<Float>(-1.0F)
    val loss: MutableStateFlow<Float> = _loss

    init {
        CoroutineScope(Dispatchers.Main).launch {
            llmRepository.lossFlow.collect { loss ->
                _loss.value = loss
            }
        }
    }

    fun performTraining(trainData: List<String>) {
        _isTraining.value = true

        llmRepository.prepareTraining()

        val trainLoss = llmRepository.runTraining(trainData)

        _isTraining.value = false
        _loss.value = trainLoss
    }
}

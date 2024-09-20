package com.example.orttransformer.viewmodels

import androidx.lifecycle.ViewModel
import com.example.orttransformer.repository.TrainingRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class TrainingViewModel(private val trainingRepository: TrainingRepository) : ViewModel() {

    val isTraining : StateFlow<Boolean> = trainingRepository.isTraining
    val trainLoss : MutableStateFlow<Float> = trainingRepository.loss

    private val _trainingData = MutableStateFlow<List<String>>(emptyList())
    val trainingData: StateFlow<List<String>> = _trainingData

    fun addTrainingData(newData : String) {
        _trainingData.value += newData
    }

    fun removeTrainingData(data: String) {
        _trainingData.value -= data
    }

    fun startTraining() {
        trainingRepository.performTraining(trainingData.value)
        _trainingData.value = listOf()
    }
}
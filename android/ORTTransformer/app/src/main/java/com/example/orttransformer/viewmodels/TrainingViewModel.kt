package com.example.orttransformer.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.orttransformer.repository.TrainingRepository
import com.example.orttransformer.repository.TrainingUiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch



class TrainingViewModel(private val trainingRepository: TrainingRepository) : ViewModel() {

    val isTraining : MutableStateFlow<TrainingUiState> = trainingRepository.isTraining
    val trainLoss : MutableStateFlow<Float> = trainingRepository.loss

    private val _trainingData = MutableStateFlow<List<String>>(emptyList())
    val trainingData: StateFlow<List<String>> = _trainingData

    fun addTrainingData(newData : String) {
        _trainingData.value += newData
    }

    fun removeTrainingData(data: String) {
        _trainingData.value -= data
    }

    fun readyForTraining() {
        isTraining.value = TrainingUiState.ReadyTrain
    }

    fun startTraining() {
        viewModelScope.launch {
            trainingRepository.performTraining(trainingData.value)
            _trainingData.value = listOf()
        }
    }
}
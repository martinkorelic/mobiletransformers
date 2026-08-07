package com.martinkorelic.mobiletransformers.app.viewmodels

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.repository.TrainingRepository
import com.martinkorelic.mobiletransformers.repository.TrainingCallback
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

enum class TrainingUiState {
    Training,
    FinishedLoadingData,
    LoadingModel,
    FinishedLoadingModel,
    LoadingData,
    ReadyTrain,
    SavingModel,
    FinishedSavingModel,
    MergingWeights,
    FinishedMergingWeights,
    Error
}

class TrainingViewModel(private val llmRepository: LLMRepository, private val trainingRepository: TrainingRepository) : ViewModel() {

    private val LOG_TAG = "TrainingViewModel"

    private val _trainingState = MutableStateFlow(TrainingUiState.ReadyTrain)
    val trainingState: StateFlow<TrainingUiState> = _trainingState

    private val _trainLoss = MutableStateFlow(-1.0f)
    val trainLoss: StateFlow<Float> = _trainLoss

    private val _currentStepDuration = MutableStateFlow(0L)
    val currentStepDuration: StateFlow<Long> = _currentStepDuration

    private val _averageStepDuration = MutableStateFlow(0L)
    val averageStepDuration: StateFlow<Long> = _averageStepDuration

    private val _totalTrainingTime = MutableStateFlow(0L)
    val totalTrainingTime: StateFlow<Long> = _totalTrainingTime

    private val _currentStep = MutableStateFlow(0)
    val currentStep: StateFlow<Int> = _currentStep

    private val _currentEpoch = MutableStateFlow(0)
    val currentEpoch: StateFlow<Int> = _currentEpoch

    private val _learningRate = MutableStateFlow(0F)
    val learningRate: StateFlow<Float> = _learningRate

    fun startTraining() {
        viewModelScope.launch {
            trainingRepository.performTraining(
                trainingCallback = object : TrainingCallback {

                    override fun onModelLoadStart() {
                        _trainingState.value = TrainingUiState.LoadingModel
                    }

                    override fun onModelLoadEnd() {
                        _trainingState.value = TrainingUiState.FinishedLoadingModel
                    }

                    override fun onDataLoadStart() {
                        _trainingState.value = TrainingUiState.LoadingData
                    }

                    override fun onDataLoadEnd(totalSteps: Int, stepsPerEpoch: Int) {
                        _trainingState.value = TrainingUiState.FinishedLoadingData
                    }

                    override fun onSaveModelStart(trainingProgress: TrainingProgress) {
                        _trainingState.value = TrainingUiState.SavingModel
                    }

                    override fun onSaveModelEnd(trainingProgress: TrainingProgress) {
                        _trainingState.value = TrainingUiState.FinishedSavingModel
                    }

                    override fun onStepStart(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "Step start")
                        _currentStep.value = trainingProgress.currentStep
                        _currentEpoch.value = trainingProgress.currentEpoch
                    }

                    override fun onStepEnd(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "Step end")
                        _trainLoss.value = trainingProgress.stepLoss
                        _currentStepDuration.value = trainingProgress.stepDurationMs
                        _totalTrainingTime.value = trainingProgress.totalDurationMs
                        _learningRate.value = trainingProgress.learningRate

                        // Calculate average step duration
                        if (trainingProgress.currentStep > 0) {
                            _averageStepDuration.value = trainingProgress.totalDurationMs / (trainingProgress.currentStep + 1)
                        }
                    }

                    override fun onEpochStart(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "Epoch start")
                        _trainingState.value = TrainingUiState.Training
                        _currentEpoch.value = trainingProgress.currentEpoch
                    }

                    override fun onEpochEnd(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "Epoch end")
                    }

                    override fun onMergeStart(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "Merge start")
                        _trainingState.value = TrainingUiState.MergingWeights
                    }

                    override fun onMergeEnd(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "Merge end")
                        _trainingState.value = TrainingUiState.FinishedMergingWeights
                    }

                    override fun onCompletion(trainingProgress: TrainingProgress) {
                        Log.d(LOG_TAG, "On completion")
                        _trainingState.value = TrainingUiState.ReadyTrain
                    }

                    override fun onError(error: Throwable) {
                        Log.d(LOG_TAG, "On Error", error)
                        _trainingState.value = TrainingUiState.Error
                    }
                }
            )
        }
    }

    fun endTraining(saveModel: Boolean) {
        viewModelScope.launch {
            trainingRepository.endTraining(saveModel)
        }
    }
}
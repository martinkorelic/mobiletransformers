package com.martinkorelic.orttransformer.viewmodels

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import com.martinkorelic.ortmobile.ORTGenerationConfig
import com.martinkorelic.ortmobile.ORTRagConfig
import com.martinkorelic.ortmobile.ORTTrainingConfig
import com.martinkorelic.ortmobile.repository.LLMRepository

class ConfigurationViewModel(private val llmRepository: LLMRepository) : ViewModel() {

    private val _generationConfig = mutableStateOf(llmRepository.generationConfig)
    val generationConfig: MutableState<ORTGenerationConfig> = _generationConfig

    private val _trainingConfig = mutableStateOf(llmRepository.trainingConfig)
    val trainingConfig: MutableState<ORTTrainingConfig> = _trainingConfig

    private val _ragConfig = mutableStateOf(llmRepository.ragConfig)
    val ragConfig: MutableState<ORTRagConfig> = _ragConfig

    private val _ragEnabled = mutableStateOf(false)
    val ragEnabled : MutableState<Boolean> = _ragEnabled

    val availableModels = llmRepository.availableModels

    // Add availability states
    private val _isRagAvailable = mutableStateOf(llmRepository.isRagAvailable)
    val isRagAvailable: MutableState<Boolean> = _isRagAvailable

    private val _isTrainingAvailable = mutableStateOf(llmRepository.isTrainingAvailable)
    val isTrainingAvailable: MutableState<Boolean> = _isTrainingAvailable

    private val _isGenerationAvailable = mutableStateOf(llmRepository.isGenerationAvailable)
    val isGenerationAvailable: MutableState<Boolean> = _isGenerationAvailable

    init {
        // Initialize availability and disable RAG if not available
        updateAvailability()
    }

    private fun updateAvailability() {
        _isRagAvailable.value = llmRepository.isRagAvailable
        _isTrainingAvailable.value = llmRepository.isTrainingAvailable
        _isGenerationAvailable.value = llmRepository.isGenerationAvailable

        // Disable RAG if not available
        if (!llmRepository.isRagAvailable) {
            _ragEnabled.value = false
        }
    }

    fun updateGenerationConfig(config: ORTGenerationConfig) {
        _generationConfig.value = config
        llmRepository.generationConfig = config // Persist to repository
    }

    fun updateTrainingConfig(config: ORTTrainingConfig) {
        _trainingConfig.value = config
        llmRepository.trainingConfig = config // Persist to repository
    }

    fun updateRagConfig(config: ORTRagConfig) {
        _ragConfig.value = config
        llmRepository.ragConfig = config // Persist to repository
    }

    fun updateRagEnabled(enabled: Boolean) {
        _ragEnabled.value = enabled
    }

    fun onGenerationModelChanged(modelName: String) {
        // Reload configuration from repository when model changes
        llmRepository.modelName = modelName
        val newConfig = llmRepository.generationConfig.copy(repoName = modelName)
        _generationConfig.value = newConfig

        val newRagConfig = llmRepository.ragConfig.copy(repoName = modelName)
        _ragConfig.value = newRagConfig

        updateAvailability()
    }

    fun onTrainingModelChanged(modelName: String) {
        // Reload configuration from repository when model changes
        llmRepository.modelName = modelName
        val newConfig = llmRepository.trainingConfig.copy(repoName = modelName)
        _trainingConfig.value = newConfig

        updateAvailability()
    }

}
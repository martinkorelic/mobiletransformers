package com.example.orttransformer.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.orttransformer.repository.ChatMessage
import com.example.orttransformer.repository.InferenceRepository
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class InferenceViewModel(private val inferenceRepository: InferenceRepository) : ViewModel() {
    val chatHistory: StateFlow<List<ChatMessage>> = inferenceRepository.chatHistory
    val chatStream : StateFlow<List<String>> = inferenceRepository.chatStream
    val isStreaming : StateFlow<Boolean> = inferenceRepository.isStreaming
    val generationConfig : MutableMap<String, String> = mutableMapOf()

    fun sendMessage(message: String) {
        viewModelScope.launch {
            inferenceRepository.sendMessage(message, generationConfig)
        }
    }
}

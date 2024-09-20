package com.example.orttransformer.viewmodels

import androidx.lifecycle.ViewModel
import com.example.orttransformer.repository.ChatMessage
import com.example.orttransformer.repository.InferenceRepository
import kotlinx.coroutines.flow.StateFlow

class InferenceViewModel(private val inferenceRepository: InferenceRepository) : ViewModel() {
    val chatHistory: StateFlow<List<ChatMessage>> = inferenceRepository.chatHistory
    val chatStream : StateFlow<List<String>> = inferenceRepository.chatStream
    val isStreaming : StateFlow<Boolean> = inferenceRepository.isStreaming

    fun sendMessage(message: String) {
        inferenceRepository.sendMessage(message)
    }
}

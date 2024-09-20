package com.example.orttransformer.repository

import android.util.Log
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class ChatMessage(val message: String, val isUserMessage: Boolean)

class InferenceRepository(private val llmRepository: LLMRepository) {
    private val _chatHistory = MutableStateFlow<List<ChatMessage>>(emptyList())
    val chatHistory: StateFlow<List<ChatMessage>> = _chatHistory

    private val _isStreaming = MutableStateFlow(false)
    val isStreaming: StateFlow<Boolean> = _isStreaming

    private val _chatStream = MutableStateFlow<List<String>>(emptyList())
    val chatStream: StateFlow<List<String>> = _chatStream

    init {
        // Observe token generation flow
        CoroutineScope(Dispatchers.Main).launch {
            llmRepository.tokenFlow.collect { token ->
                // Append new token to chat stream
                when (token) {
                    "[STOP]" -> {
                        _isStreaming.value = false
                        _chatHistory.value += ChatMessage(
                            message = chatStream.value.joinToString(separator = ""),
                            isUserMessage = false
                        )
                        _chatStream.value = listOf()
                    }
                    else -> {
                        _isStreaming.value = true
                        _chatStream.value += token
                    }
                }
            }
        }
    }

    fun sendMessage(userMessage: String) {
        // Append user message to chat history
        _chatHistory.value += ChatMessage(message = userMessage, isUserMessage = true)

        // Prepare generation
        llmRepository.prepareGeneration()

        // Start inference
        llmRepository.runInferenceStream(userMessage)
    }

}
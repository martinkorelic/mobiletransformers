package com.example.orttransformer.repository

import android.util.Log
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

data class ChatMessage(val message: String, val isUserMessage: Boolean)

class InferenceRepository(private val llmRepository: LLMRepository) {

    private val couroutineScope = CoroutineScope(Dispatchers.IO)

    private val _chatHistory = MutableStateFlow<List<ChatMessage>>(emptyList())
    val chatHistory: StateFlow<List<ChatMessage>> = _chatHistory

    private val _isStreaming = MutableStateFlow(false)
    val isStreaming: StateFlow<Boolean> = _isStreaming

    private val _chatStream = MutableStateFlow<List<String>>(emptyList())
    val chatStream: StateFlow<List<String>> = _chatStream

    // Time metrics
    private val _ttlmStream = MutableStateFlow<Double>(0.0)
    val ttlmStream: StateFlow<Double> = _ttlmStream

    private val _prefillTimeStream = MutableStateFlow<Double>(0.0)
    val prefillTimeStream: StateFlow<Double> = _prefillTimeStream

    private val _generationTimeStream = MutableStateFlow<Double>(0.0)
    val generationTimeStream: StateFlow<Double> = _generationTimeStream

    init {
        // Observe token generation flow
        couroutineScope.launch {
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

        if (llmRepository.trackMetrics) {

            couroutineScope.launch {
                launch {
                    llmRepository.ttlmStream.collect { time ->
                        _ttlmStream.value = time
                    }
                }
                launch {
                    llmRepository.prefillTimeStream.collect { time ->
                        _prefillTimeStream.value = time
                    }
                }
                launch {
                    llmRepository.generationTimeStream.collect { time ->
                        _generationTimeStream.value = time
                    }
                }
            }
        }

    }

    suspend fun sendMessage(userMessage: String, generationConfig : Map<String, String>?) {
        // Append user message to chat history
        _chatHistory.value += ChatMessage(message = userMessage, isUserMessage = true)
        _isStreaming.value = true
        // Prepare generation
        if (llmRepository.llmState != LLMState.ReadyGenerate) {
            val job = llmRepository.prepareGeneration(generationConfig)
            job.join()
        }

        // Start inference
        llmRepository.runGenerationStream(userMessage)
    }

}
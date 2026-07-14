package com.martinkorelic.mobiletransformers.app.viewmodels

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.InferenceProgress
import com.martinkorelic.mobiletransformers.repository.InferenceRepository
import com.martinkorelic.mobiletransformers.RagResult
import com.martinkorelic.mobiletransformers.entity.VectorEntityInterface
import com.martinkorelic.mobiletransformers.repository.GenerationCallback
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.repository.RagCallback
import com.martinkorelic.mobiletransformers.repository.RagRepository
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

interface Message {
    val timestamp: Long
    val id: String
}

// Helper function to generate unique IDs
private fun generateId(): String = "msg_${System.currentTimeMillis()}_${kotlin.random.Random.nextInt(1000)}"

data class ChatMessage(
    val message: String,
    val isUserMessage: Boolean,
    override val timestamp: Long = System.currentTimeMillis(),
    override val id: String = generateId()
) : Message

data class ChunkDetails(
    val file : String,
    val content : String,
    val score : Double
)

data class RagMessage(
    val documents : List<ChunkDetails>,
    override val timestamp: Long = System.currentTimeMillis(),
    override val id: String = generateId()
) : Message

enum class InferenceUiState {
    LoadingModel,
    LoadingRetriever,
    FinishedLoadingModel,
    Querying,
    ReadyGenerate,
    Generating,
    Error
}

class InferenceViewModel(private val llmRepository: LLMRepository, private val inferenceRepository: InferenceRepository, private val ragRepository: RagRepository? = null) : ViewModel() {

    private val LOG_TAG = "InferenceViewModel"

    private val _inferenceState = MutableStateFlow(InferenceUiState.ReadyGenerate)
    val inferenceState: StateFlow<InferenceUiState> = _inferenceState

    private val _chatHistory = MutableStateFlow<List<Message>>(emptyList())
    val chatHistory: StateFlow<List<Message>> = _chatHistory

    private val _chatStream = MutableStateFlow<List<String>>(emptyList())
    val chatStream: StateFlow<List<String>> = _chatStream

    private val _isStreaming = MutableStateFlow(false)
    val isStreaming: StateFlow<Boolean> = _isStreaming

    private val _ttlmTime = MutableStateFlow(0.0)
    val ttlmTime: StateFlow<Double> = _ttlmTime

    private val _prefillTime = MutableStateFlow(0.0)
    val prefillTime: StateFlow<Double> = _prefillTime

    private val _generationTime = MutableStateFlow(0.0)
    val generationTime: StateFlow<Double> = _generationTime

    private val _queryTime = MutableStateFlow(0.0)
    val queryTime: StateFlow<Double> = _queryTime

    private val _embeddingTime = MutableStateFlow(0.0)
    val embeddingTime: StateFlow<Double> = _embeddingTime

    init {

        // Generation callbacks
        llmRepository.generationCallback = object : GenerationCallback {

            override fun onModelLoadStart() {
                Log.i(LOG_TAG,"Loading...")
                _inferenceState.value = InferenceUiState.LoadingModel
            }

            override fun onModelLoadEnd() {
                Log.i(LOG_TAG,"Finished loading...")
                _inferenceState.value = InferenceUiState.ReadyGenerate
            }

            override fun onStartGeneration(inferenceProgress: InferenceProgress) {
                _inferenceState.value = InferenceUiState.Generating
            }

            override fun onPartialResult(inferenceProgress: InferenceProgress) {
                _isStreaming.value = true

                if (llmRepository.ortTokenizerNative?.isSpecialToken(inferenceProgress.tokenId) == false)
                    _chatStream.value += inferenceProgress.token

                _ttlmTime.value = inferenceProgress.timeToLoadModelMs / 1000.0
                _prefillTime.value = inferenceProgress.prefillTimeMs / 1000.0
                _generationTime.value = inferenceProgress.avgTokensPerSecond
            }

            override fun onCompletion(inferenceProgress: InferenceProgress) {
                _chatHistory.value += ChatMessage(message = _chatStream.value.joinToString(separator = ""), isUserMessage = false)
                _chatStream.value = listOf()
                _isStreaming.value = false
                _inferenceState.value = InferenceUiState.ReadyGenerate
            }

            override fun onError(error: Throwable) {
                _isStreaming.value = false
                _inferenceState.value = InferenceUiState.Error
                Log.e(LOG_TAG, "Generation error: ${error.message}")
            }
        }


        // Rag repository callbacks if defined
        llmRepository.ragCallback = object : RagCallback {
            override fun onModelLoadStart() {
                _inferenceState.value = InferenceUiState.LoadingRetriever
            }

            override fun onModelLoadEnd() {
                _inferenceState.value = InferenceUiState.ReadyGenerate
            }
        }
    }

    fun reloadInferenceSession() {
        // Reloads session with new configuration and model

        viewModelScope.launch {
            inferenceRepository.reloadSession()
            _chatHistory.value = listOf()
            _chatStream.value = listOf()
        }
    }

    fun sendMessage(message: String, useRag : Boolean = false) {
        viewModelScope.launch {

            // Initialize RAG Repository if it hasn't been before
            if (ragRepository != null
                && llmRepository.ortRetriever == null
                && useRag
                && llmRepository.isRagAvailable) {
                ragRepository.initialize()
            }

            // Query RAG repository if useRag is requested
            if (ragRepository != null
                && useRag
                && llmRepository.ortRetriever != null) {

                // Query the RAG repository and then on callback when results are delivered query with question and context
                ragRepository.query(message, ragCallback = object : RagCallback {

                    override fun onModelLoadStart() {
                        _inferenceState.value = InferenceUiState.LoadingRetriever
                    }

                    override fun onModelLoadEnd() {
                        _inferenceState.value = InferenceUiState.ReadyGenerate
                    }

                    override fun onQueryStart() {
                        _inferenceState.value = InferenceUiState.Querying
                    }

                    override fun onQueryEnd() {
                        _inferenceState.value = InferenceUiState.ReadyGenerate
                    }

                    override fun onQueryResults(queryResult: RagResult) {

                        _embeddingTime.value = queryResult.embeddingTimeMs / 1000.0
                        _queryTime.value = queryResult.queryTimeMs / 1000.0

                        val augmentedMessage = insertContextIntoMessage(message, queryResult.documents)

                        // Add user message to history
                        _chatHistory.value += ChatMessage(message = message, isUserMessage = true)
                        _isStreaming.value = true
                        _chatStream.value = listOf()

                        // Add Rag results to history
                        _chatHistory.value += RagMessage(
                            documents = queryResult.documents?.map { d -> ChunkDetails(
                                content = d.first.content,
                                file = d.first.document,
                                score = d.second
                            ) } ?: listOf()
                        )

                        // Generate with augmented message
                        // TODO: Should have a cleaner approach
                        viewModelScope.launch {
                            inferenceRepository.generate(
                                userMessage = augmentedMessage
                            )
                        }
                    }
                })

                // Return @launch so we don't trigger normal generation
                return@launch
            }

            // Add user message to history
            _chatHistory.value += ChatMessage(message = message, isUserMessage = true)
            _isStreaming.value = true
            _chatStream.value = listOf()

            // Generate message without RAG
            inferenceRepository.generate(
                userMessage = message
            )
        }
    }

    /**
     * Custom function to insert context into the prompt.
     */
    fun insertContextIntoMessage(
        prompt: String,
        documents: List<Pair<VectorEntityInterface, Double>>?,
        maxContextLength: Int = 2000,
        contextTemplate: String = "\nContext: {context}\n\nQuestion: {question}"
    ): String {
        if (documents == null) return prompt

        if (documents.isEmpty()) {
            return prompt
        }

        // Sort documents by relevance score (higher scores first)
        val sortedDocuments = documents.sortedByDescending { it.second }

        // Build context string from documents
        val contextBuilder = StringBuilder()
        var currentLength = 0

        for ((document, score) in sortedDocuments) {
            val content = document.content.trim()

            // Check if adding this document would exceed max length
            val additionalLength = content.length + 2 // +2 for newlines
            if (currentLength + additionalLength > maxContextLength) {
                // Try to fit partial content if there's space
                val remainingSpace = maxContextLength - currentLength
                if (remainingSpace > 50) { // Only add if meaningful space remains
                    val truncatedContent = content.take(remainingSpace - 3) + "..."
                    contextBuilder.append(truncatedContent)
                }
                break
            }

            // Add document content
            if (contextBuilder.isNotEmpty()) {
                contextBuilder.append("\n\n")
            }
            contextBuilder.append(content)
            currentLength += additionalLength
        }

        val contextText = contextBuilder.toString()

        // Replace placeholders in template
        return contextTemplate
            .replace("{context}", contextText)
            .replace("{question}", prompt)
    }
}

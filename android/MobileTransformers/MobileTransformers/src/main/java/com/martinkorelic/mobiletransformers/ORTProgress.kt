package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.rag.RagMatch

data class TrainingProgress(
    val currentStep: Int,
    val currentEpoch: Int,
    val totalLoss: Float = 0f,
    val epochLoss: Float,
    val stepLoss: Float,
    val learningRate: Float,
    val stepDurationMs: Long,
    val epochDurationMs: Long,
    val totalDurationMs: Long,
    val isCompleted: Boolean = false
)

data class InferenceProgress(
    val token: String,
    val tokenId : Int,
    val totalDecodedTokens: Int,
    val prefillTimeMs: Long = 0L,
    val timeToLoadModelMs: Long = 0L,
    val generationTimeMs: Long = 0L,
    val avgTokensPerSecond: Double = 0.0,
    val isCompleted: Boolean = false,
    /**
     * Tokens the prompt occupied, measured after templating and after any trim.
     *
     * Together with [totalDecodedTokens] and [contextLimit] this is what lets a caller say how much
     * of the window a turn consumed. Nothing reported it before, so an app could show tokens/second
     * but could not answer "how close am I to the limit" — the question that actually predicts the
     * next turn being truncated.
     */
    val promptTokenCount: Int = 0,
    /** Tokens the model can attend to at once, or 0 when the package declares none. */
    val contextLimit: Int = 0,
)

data class RagResult(
    val documents: List<RagMatch>?,
    val embeddingTimeMs : Long = 0L,
    val queryTimeMs : Long = 0L
)
package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.entity.VectorEntityInterface

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
    val isCompleted: Boolean = false
)

data class RagResult(
    val documents: List<Pair<VectorEntityInterface, Double>>?,
    val embeddingTimeMs : Long = 0L,
    val queryTimeMs : Long = 0L
)
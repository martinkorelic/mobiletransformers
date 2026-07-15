package com.martinkorelic.mobiletransformers

/**
 * Public per-token generation progress (#19), mapped 1:1 from the internal `InferenceProgress` so app code
 * never imports repository/`ORT*` types.
 */
data class GenerateProgress(
    val token: String,
    val tokenId: Int,
    val totalDecodedTokens: Int,
    val prefillTimeMs: Long = 0L,
    val timeToLoadModelMs: Long = 0L,
    val generationTimeMs: Long = 0L,
    val avgTokensPerSecond: Double = 0.0,
    val isCompleted: Boolean = false,
)

/**
 * Public streaming callback for [MobileTransformerModel.generate] (#19). Mirrors the internal
 * `GenerationCallback`; the facade drives the identical ordered sequence on every engine
 * (`onStartGeneration` → N×`onPartialResult` → `onCompletion`, or `onError`) — cross-engine parity is
 * locked by #24.
 */
interface GenerateCallback {
    fun onStartGeneration(progress: GenerateProgress) {}

    fun onPartialResult(progress: GenerateProgress) {}

    fun onCompletion(progress: GenerateProgress) {}

    fun onError(error: Throwable) {}
}

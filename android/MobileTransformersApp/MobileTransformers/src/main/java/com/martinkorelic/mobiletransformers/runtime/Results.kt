package com.martinkorelic.mobiletransformers.runtime

/**
 * Public result types returned by the facade (#17). They wrap the existing callback payloads
 * (`TrainingProgress`, `InferenceProgress`, `RagResult`) without leaking the `ORT*`/`*Native` types.
 *
 * NOTE: [TrainingResult] is the SAME type #18 enriches (it adds `checkpoint`/`summary` fields) — it is not
 * a second type. #24 refines generation metrics over [GenerationResult].
 */

data class TrainingResult(
    val finalStep: Int = 0,
    val finalEpoch: Int = 0,
    val finalLoss: Float = 0f,
    val totalDurationMs: Long = 0L,
    val merged: Boolean = false,
    // Enriched by #18 (training lifecycle). Null until a checkpoint/summary is available.
    val checkpoint: com.martinkorelic.mobiletransformers.training.CheckpointInfo? = null,
    val summary: com.martinkorelic.mobiletransformers.ORTTrainerNative.TrainingSummary? = null,
)

data class MergeResult(
    val merged: Boolean,
    /** Path to the inference-ready package produced by the handoff-validated merge (#8/#9). */
    val inferencePackagePath: String? = null,
)

data class GenerationResult(
    val text: String,
    val tokenCount: Int = 0,
    val generationTimeMs: Long = 0L,
    val avgTokensPerSecond: Double = 0.0,
)

data class RetrievalMatch(val text: String, val score: Double)

data class RetrievalResult(
    val matches: List<RetrievalMatch> = emptyList(),
    val queryTimeMs: Long = 0L,
)

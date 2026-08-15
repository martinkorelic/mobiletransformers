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
    val summary: TrainingSummary? = null,
)

/**
 * Public mirror of the run-level training metrics (#17).
 *
 * This field used to be typed `ORTTrainerNative.TrainingSummary` — a nested type of the JNI-holding
 * class — on [TrainingResult], which `MobileTransformerModel.train()` returns. That put an `ORT*`
 * type on the public surface, contradicting this file's own contract above. The payload is eight
 * plain scalars, so mirroring costs nothing; field names are also normalized to Kotlin camelCase.
 */
data class TrainingSummary(
    val trainRuntimeSeconds: Float = 0f,
    val trainStepsPerSecond: Float = 0f,
    val trainSamplesPerSecond: Float = 0f,
    val totalSteps: Int = 0,
    val totalSamples: Int = 0,
    val finalLoss: Float = 0f,
    val peakMemoryMb: Long = 0L,
    val averageMemoryMb: Float = 0f,
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

/** Result of pushing a trained adapter to the Hub (#19 surface; real upload lands with #22). */
data class PushResult(
    val repoId: String,
    val url: String? = null,
)

/** Result of ingesting documents into the RAG vector store (#26). */
data class IngestResult(
    val chunkCount: Int,
)

/** One class the head predicts, with the probability assigned to it. */
data class LabelScore(
    /** The package's own name for this class, from `id2label`. */
    val label: String,
    /** Softmax probability in `0.0..1.0`. */
    val score: Double,
    /** The class index, kept because a label name need not be unique or stable across exports. */
    val index: Int,
)

/**
 * Result of sequence classification (#33).
 *
 * Encoder fine-tuning worked end to end and the resulting model could not be *run* — the facade had
 * generate/retrieve/ingest/train and nothing that returns a class. Training a classifier and then
 * being unable to ask it anything is what this closes.
 */
data class ClassificationResult(
    /** Every class, highest probability first. */
    val scores: List<LabelScore> = emptyList(),
    /** How many entries a caller asked to see; [top] is capped by it. */
    val topK: Int = 5,
) {
    /** The predicted class, or `null` for a head with no labels. */
    val best: LabelScore? get() = scores.firstOrNull()

    val top: List<LabelScore> get() = scores.take(topK.coerceAtLeast(1))
}

/** Result of grounded generation (#27): the answer, the retrieved matches, and the exact assembled prompt. */
data class GroundedResult(
    val text: String,
    val matches: List<RetrievalMatch> = emptyList(),
    val prompt: String = "",
)

package com.martinkorelic.mobiletransformers.packages

/**
 * Feature groups a caller can request from a package (#17).
 *
 * Critical semantics: [GenAI] and [ManualInference] are
 * **engine selectors over the same shared package**, not separate downloadable feature groups. The package
 * on disk (`train/`, `inference/`, `embedding/`) is consumable by both the Native ORT engine and the GenAI
 * engine — requesting [GenAI]/[ManualInference] sets/validates the [com.martinkorelic.mobiletransformers.runtime.InferenceEngine],
 * it does not trigger a different download. [Inference], [Training], [Rag], [Embedding], [Adapter] are the
 * genuine feature groups.
 */
enum class ModelFeature {
    Inference,
    Training,
    Rag,
    Embedding,
    GenAI,
    ManualInference,
    Adapter,
    ;

    /** True when this value only selects an engine over the shared package (no separate download). */
    val isEngineSelector: Boolean
        get() = this == GenAI || this == ManualInference
}

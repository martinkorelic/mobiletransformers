package com.martinkorelic.mobiletransformers.runtime

import com.martinkorelic.mobiletransformers.packages.ModelFeature

/**
 * Model-level capability flags (#17). Distinct from #11's engine-level `EngineCapabilities`: these describe
 * what the whole [com.martinkorelic.mobiletransformers.MobileTransformerModel] can do, derived directly from
 * `LLMRepository.isTrainingAvailable/isGenerationAvailable/isRagAvailable` + the selected engine.
 */
data class RuntimeCapabilities(
    val engine: InferenceEngine,
    val supportsTraining: Boolean,
    val supportsMerge: Boolean,
    val supportsRag: Boolean,
    val supportsEmbedding: Boolean,
    /** #34: `TrainingScheduler.schedule()` can run charging-constrained chunks for this model. */
    val supportsScheduledTraining: Boolean = false,
    val supportsAdapterTensorExport: Boolean = false, // future (#35/#36)
    val availableFeatures: Set<ModelFeature> = emptySet(),
)

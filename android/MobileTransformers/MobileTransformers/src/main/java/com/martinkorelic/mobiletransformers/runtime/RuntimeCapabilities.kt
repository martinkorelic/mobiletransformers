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
    /**
     * The engines this package can actually be run with **on this device**, which is what an engine
     * picker must offer.
     *
     * #17/#19 gap found building the showcase app's Chat screen. `engine` said which engine this
     * handle resolved to, but nothing said which *others* were selectable, so a picker could only
     * offer both and discover the answer by catching `EngineUnavailableException` — i.e. by using an
     * exception as control flow for a question the SDK already knows the answer to.
     *
     * Always contains [InferenceEngine.NATIVE]: it is #11's guaranteed floor. It contains
     * [InferenceEngine.GENAI] only when the installed package ships `inference/genai_config.json`
     * **and** the GenAI native probe succeeds — the same two conditions `ModelRuntimeFactory` applies,
     * so offering an engine from this set and then being refused it would be a bug in one of them.
     */
    val availableEngines: Set<InferenceEngine> = setOf(InferenceEngine.NATIVE),
    /**
     * What objective this package's inference graph was exported for.
     *
     * Until this existed, capabilities answered only "can it train / retrieve / run GenAI" and never
     * "what kind of model is it", so a sequence-classification encoder was indistinguishable from a
     * chat decoder to every caller. An app could only find out by asking for generation and reading
     * the failure. Read from the `inference/optimum_config.json` the exporter has always written —
     * see [com.martinkorelic.mobiletransformers.packages.PackageTask].
     */
    val task: com.martinkorelic.mobiletransformers.packages.PackageTask =
        com.martinkorelic.mobiletransformers.packages.PackageTask.UNKNOWN,
) {
    /** This package predicts a class per input rather than generating tokens. */
    val isClassifier: Boolean get() = task.isClassifier

    /**
     * `classify()` will work: the package is a classifier **and** it names its labels.
     *
     * Both halves are required. A classification graph whose labels are unknown can still be run,
     * but every prediction comes back as `LABEL_3` — which is a number in a costume, not an answer,
     * so the honest report is that the capability is not usable on this package.
     */
    val supportsClassification: Boolean get() = task.isClassifier && task.labelCount > 0
}

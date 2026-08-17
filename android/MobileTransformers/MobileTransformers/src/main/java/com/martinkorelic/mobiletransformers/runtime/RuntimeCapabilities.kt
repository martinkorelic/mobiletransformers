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
    /**
     * Whether this model was trained to emit tool calls, and in which grammar.
     *
     * Read from the package's own chat template — see
     * [com.martinkorelic.mobiletransformers.packages.ToolCallSupport]. Two things depended on
     * knowing this and neither could: `generateToolCall` chose its parser by looking for the string
     * `"functiongemma"` in the *architecture* name (`gemma3_text`), and an app had no way to tell a
     * user which of its models can call tools at all.
     */
    val toolCalling: com.martinkorelic.mobiletransformers.packages.ToolCallSupport =
        com.martinkorelic.mobiletransformers.packages.ToolCallSupport.NONE,
    /**
     * Every parameter this package's training graph materialises, or 0 when it declares none.
     *
     * Exposed because it is the input to
     * [com.martinkorelic.mobiletransformers.runtime.MemoryHeadroom] — the only way an app can warn a
     * user that a run will not fit **before** Android kills the process for it, which it does with
     * SIGKILL and no recoverable error. The exporter has written this since the training stage
     * existed and nothing on the device read it.
     */
    val trainingParameterCount: Long = 0L,
    /**
 * The PEFT method(s) this package was exported for — `lora`, `lora-xs`, `mars`, …
     *
 * The exporter has recorded this in the manifest since the training stage existed and nothing on
 * the device read it, so an app could not tell a MARS package from a LoRA one. That matters most
 * for MARS, which is this project's own contribution: someone watching the fine-tuning demo could
 * not see which technique they were watching.
     *
 * Empty for a package with no training stage, and for older exports that predate the field —
 * absence means "not declared", never "no PEFT".
 */
    val peftMethods: Set<String> = emptySet(),
) {
    /**
 * The PEFT method to show when there is room for exactly one, or `null` when none is declared.
     *
 * Every package produced so far declares a single method; the field is a list because the format
 * allows more, not because a package has ever had two.
 */
    val primaryPeftMethod: String? get() = peftMethods.firstOrNull()

    /** The precision measured in the shipped graph — see [PackageTask.inferenceGraphPrecision]. */
    val graphPrecision: String? get() = task.inferenceGraphPrecision

    /** This model has a tool-call grammar of its own; asking it for a call is reasonable. */
    val supportsToolCalling: Boolean get() = toolCalling.supported

    /** This package predicts a class per input rather than generating tokens. */
    val isClassifier: Boolean get() = task.isClassifier

    /**
 * This package has no generative head at all, so nothing that produces tokens can work with it.
     *
 * Broader than [isClassifier] on purpose. A plain embedding model (`feature-extraction`, e.g.
 * `all-MiniLM-L6-v2` on its own) is not a classifier and cannot generate either — a check that
 * tested only for classifiers offered it a chat box, which is a promise the package cannot keep.
 * An UNKNOWN task is deliberately not included: an older package that declares nothing must keep
 * working, and withholding generation from it would be a narrower answer than the evidence
 * supports.
 */
    val isEncoderOnly: Boolean
    get() = task.isClassifier ||
    task.taskType == com.martinkorelic.mobiletransformers.constants.TaskType.FEATURE_EXTRACTION

    /**
     * `classify()` will work: the package is a classifier **and** it names its labels.
     *
     * Both halves are required. A classification graph whose labels are unknown can still be run,
     * but every prediction comes back as `LABEL_3` — which is a number in a costume, not an answer,
     * so the honest report is that the capability is not usable on this package.
     */
    val supportsClassification: Boolean get() = task.isClassifier && task.labelCount > 0
}

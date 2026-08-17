package com.martinkorelic.mobiletransformers

data class SamplingOptions (
    val method : String = "greedy",
    val temperature : Float = 1F,
    val topK : Int = 10,
    val topP : Float = 0.9F,
    val seed : Int = 42
)

data class ORTGenerationConfig(
    val repoName : String = "",
    var onnxName : String = "",
    val type: String = "native",
    val maxSequenceLength: Int = 128,              // Max length of generated sequence
    val trackMetrics: Boolean = true,
    val timeStepUpdate : Int = 5,
    val systemPrompt : String? = null,
    var loadMergedWeights : Boolean = false,
    val sampling : SamplingOptions = SamplingOptions(),
    val deviceOptions: DeviceOptions = DeviceOptions(),
    // #11: engine selector over the one shared package (null = auto-select; Native is the floor).
    val engine: com.martinkorelic.mobiletransformers.runtime.InferenceEngine? = null,
    /**
 * Wrap the prompt in the package's chat template when it ships one Pebble can render.
 *
 * False means **the caller has already framed its own turns** and a second framing would nest one
 * inside the other. `generateToolCall` sets it: `ToolPromptBuilder` emits a complete
 * `<start_of_turn>…` prompt deliberately, because it cannot rely on a Jinja engine rendering
 * FunctionGemma's 13 KB template on a phone.
 *
 * This became load-bearing the moment the tokenizer started reading `chat_template.jinja`. Before
 * that `chatTemplate` was null for every package, so nothing templated and the conflict could not
 * arise — which is precisely why the double-framing hazard sat unnoticed.
 */
    val applyChatTemplate: Boolean = true,
) {
    fun overrideConfig(override: ORTGenerationConfig?): ORTGenerationConfig {
        if (override == null) return this

        // Create a default config to compare against
        val defaultConfig = ORTGenerationConfig()

        return this.copy(
            repoName = override.repoName.ifBlank { this.repoName },
            onnxName = override.onnxName.ifBlank { this.onnxName },
            type = if (override.type != defaultConfig.type) override.type else this.type,
            maxSequenceLength = if (override.maxSequenceLength != defaultConfig.maxSequenceLength) override.maxSequenceLength else this.maxSequenceLength,
            trackMetrics = if (override.trackMetrics != defaultConfig.trackMetrics) override.trackMetrics else this.trackMetrics,
            loadMergedWeights = if (override.loadMergedWeights != defaultConfig.loadMergedWeights) override.loadMergedWeights else this.loadMergedWeights,
            timeStepUpdate = if (override.timeStepUpdate != defaultConfig.timeStepUpdate) override.timeStepUpdate else this.timeStepUpdate,
            systemPrompt = override.systemPrompt ?: this.systemPrompt,
            deviceOptions = if (override.deviceOptions != defaultConfig.deviceOptions) override.deviceOptions else this.deviceOptions,
            sampling = if (override.sampling != defaultConfig.sampling) override.sampling else this.sampling,
            engine = override.engine ?: this.engine,
            // Follows the same "differs from the default means the caller meant it" rule as the rest.
            // The default is true, so only an explicit false overrides.
            applyChatTemplate =
            if (override.applyChatTemplate != defaultConfig.applyChatTemplate) {
                override.applyChatTemplate
            } else {
                this.applyChatTemplate
            },
        )
    }
}
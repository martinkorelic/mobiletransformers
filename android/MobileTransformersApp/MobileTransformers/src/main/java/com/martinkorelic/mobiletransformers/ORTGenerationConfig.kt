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
    val deviceOptions: DeviceOptions = DeviceOptions()
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
            sampling = if (override.sampling != defaultConfig.sampling) override.sampling else this.sampling
        )
    }
}
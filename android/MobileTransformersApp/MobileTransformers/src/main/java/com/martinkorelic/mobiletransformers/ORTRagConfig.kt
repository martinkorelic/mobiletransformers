package com.martinkorelic.mobiletransformers

data class ORTRagArguments(
    val repoName: String? = null,
    val onnxName: String? = null,
    val embeddingDimension: Int? = null,
    val topK: Int? = null,
    val searchType: String? = null,
    val maxTextLength: Int? = null,
    val chunkSize: Int? = null,
    val chunkOverlap: Int? = null,
    val deviceOptions: DeviceOptions? = null
)

data class ORTRagConfig(

    val repoName : String = "model",
    var onnxName : String = "embedding_model",
    val embeddingDimension : Int = 256,
    val topK: Int = 10,
    val searchType : String = "semantic", // semantic, text

    // Text processing
    val maxTextLength: Int = 1024,               // Max chars per document
    val chunkSize: Int = 512,                    // For splitting long texts
    val chunkOverlap: Int = 50,                  // Overlap between chunks

    val deviceOptions: DeviceOptions = DeviceOptions()
) {
    fun overwriteWith(override: ORTRagArguments?): ORTRagConfig {
        if (override == null) return this

        return this.copy(
            repoName = override.repoName ?: this.repoName,
            onnxName = override.onnxName ?: this.onnxName,
            embeddingDimension = override.embeddingDimension ?: this.embeddingDimension,
            topK = override.topK ?: this.topK,
            searchType = override.searchType ?: this.searchType,
            maxTextLength = override.maxTextLength ?: this.maxTextLength,
            chunkSize = override.chunkSize ?: this.chunkSize,
            chunkOverlap = override.chunkOverlap ?: this.chunkOverlap,
            deviceOptions = override.deviceOptions ?: this.deviceOptions
        )
    }
}
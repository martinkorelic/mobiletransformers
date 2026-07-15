package com.martinkorelic.mobiletransformers

import android.content.res.AssetManager
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream


fun parseTrainingArguments(jsonPath: String): ORTTrainingConfig {
    if (jsonPath.isBlank()) {
        return ORTTrainingConfig()
    }

    val file = File(jsonPath)
    if (!file.exists()) {
        return ORTTrainingConfig()
    }

    val json = JSONObject(File(jsonPath).readText())

    // Parse main training config
    val trainConfig = if (json.has("train_config")) {
        json.getJSONObject("train_config")
    } else {
        json // fallback to root level
    }

    // Parse scheduler type
    val schedulerType = trainConfig.optString("schedulerType", "linear")

    // Parse scheduler-specific options
    val schedulerConfig = if (trainConfig.has("schedulerOptions")) {
        val schedulerOptions = trainConfig.getJSONObject("schedulerOptions")
        parseSchedulerConfig(schedulerType, schedulerOptions)
    } else {
        // Fallback: try to parse from root level for backward compatibility
        parseSchedulerConfigFromRoot(schedulerType, trainConfig)
    }

    // Parse device options
    val deviceOptions = if (trainConfig.has("deviceOptions")) {
        val deviceOptionsJson = trainConfig.getJSONObject("deviceOptions")
        parseDeviceOptions(deviceOptionsJson)
    } else {
        // Fallback: try to parse from root level for backward compatibility
        parseDeviceOptionsFromRoot(trainConfig)
    }

    // Parse dataset options
    val datasetOptions = if (trainConfig.has("datasetOptions")) {
        val datasetOptionsJson = trainConfig.getJSONObject("datasetOptions")
        parseDatasetOptions(datasetOptionsJson)
    } else {
        // Fallback: try to parse from root level for backward compatibility
        parseDatasetOptionsFromRoot(trainConfig)
    }

    return ORTTrainingConfig(
        repoName = trainConfig.optString("repoName", trainConfig.optString("modelName", "model")),
        onnxName = trainConfig.optString("onnxName", ".onnx"),
        taskName = trainConfig.optString("taskName", "none"),

        batchSize = trainConfig.optInt("batchSize", 4),
        numTrainEpochs = trainConfig.optInt("numTrainEpochs", 1),
        maxSteps = trainConfig.optInt("maxSteps", -1).let {
            if (it > 0) it else null
        },
        saveSteps = trainConfig.optInt("saveSteps", 100),
        gradAccumSteps = trainConfig.optInt("gradAccumSteps", 4),

        mergeWeightsAtEnd = trainConfig.optBoolean("mergeWeightsAtEnd", true),
        saveModelAtEnd = trainConfig.optBoolean("saveModelAtEnd", true),
        loadFromState = trainConfig.optBoolean("loadFromState", true),
        profileMetrics = trainConfig.optBoolean("profileMetrics", false),

        schedulerType = schedulerType,
        schedulerConfig = schedulerConfig,
        deviceOptions = deviceOptions,
        datasetOptions = datasetOptions,

        customPreprocess = null
    )
}

// Helper function to parse dataset options from a dedicated JSON object
fun parseDatasetOptions(datasetOptionsJson: JSONObject): DatasetOptions {
    return DatasetOptions(
        trainFile = datasetOptionsJson.optString("trainFile", "arc_e"),
        datasetBatchSize = datasetOptionsJson.optInt("datasetBatchSize", -1).let {
            if (it > 0) it else null
        },
        maxSequenceLength = datasetOptionsJson.optInt("maxSequenceLength", -1).let {
            if (it > 0) it else null
        },
        maxDatasetLength = datasetOptionsJson.optInt("maxDatasetLength", -1).let {
            if (it > 0) it else null
        },
        removeLongSamples = datasetOptionsJson.optBoolean("removeLongSamples", false),
        datasetSplit = datasetOptionsJson.optBoolean("datasetSplit", false),
        datasetShuffle = datasetOptionsJson.optBoolean("datasetShuffle", false),
        testRatio = datasetOptionsJson.optDouble("testRatio", 0.0).toFloat()
    )
}

// Helper function to parse dataset options from root level (backward compatibility)
fun parseDatasetOptionsFromRoot(trainConfig: JSONObject): DatasetOptions {
    return DatasetOptions(
        trainFile = trainConfig.optString("trainFile", "arc_e"),
        datasetBatchSize = trainConfig.optInt("datasetBatchSize", 64),
        maxSequenceLength = trainConfig.optInt("maxSequenceLength", 512),
        maxDatasetLength = trainConfig.optInt("maxDatasetLength", 256),
        removeLongSamples = trainConfig.optBoolean("removeLongSamples", false),
        datasetSplit = trainConfig.optBoolean("datasetSplit", false),
        datasetShuffle = trainConfig.optBoolean("datasetShuffle", false),
        testRatio = trainConfig.optDouble("testRatio", 0.0).toFloat()
    )
}

// Helper function to parse device options from a dedicated JSON object
fun parseDeviceOptions(deviceOptionsJson: JSONObject): DeviceOptions {
    return DeviceOptions(
        enableProfiling = deviceOptionsJson.optBoolean("enableProfiling", false),
        coreConfigId = deviceOptionsJson.optString("coreConfigId", "opt1"),
        memoryConfigId = deviceOptionsJson.optString("memoryConfigId", "high_perf"),
        executionProvider = deviceOptionsJson.optString("executionProvider", "cpu")
    )
}

// Helper function to parse device options from root level (backward compatibility)
fun parseDeviceOptionsFromRoot(trainConfig: JSONObject): DeviceOptions {
    return DeviceOptions(
        enableProfiling = trainConfig.optBoolean("enableProfiling", false),
        coreConfigId = trainConfig.optString("coreConfigId", "opt1"),
        memoryConfigId = trainConfig.optString("memoryConfigId", "high_perf"),
        executionProvider = trainConfig.optString("executionProvider", "cpu")
    )
}

private fun parseSchedulerConfig(schedulerType: String, schedulerOptions: JSONObject): SchedulerConfig {
    return when (schedulerType.lowercase()) {
        "linear" -> SchedulerConfig.Linear(
            learningRate = schedulerOptions.optDouble("learningRate", 1e-4).toFloat(),
            startFactor = schedulerOptions.optDouble("startFactor", 1.0).toFloat(),
            endFactor = schedulerOptions.optDouble("endFactor", 0.333).toFloat(),
        )

        "cosine" -> SchedulerConfig.Cosine(
            learningRate = schedulerOptions.optDouble("learningRate", 1e-4).toFloat(),
            minLearningRate = schedulerOptions.optDouble("minLearningRate", 0.0).toFloat(),
            warmupSteps = schedulerOptions.optInt("warmupSteps", 10)
        )

        else -> {
            println("Warning: Unknown scheduler type '$schedulerType', using linear scheduler")
            SchedulerConfig.Linear()
        }
    }
}

private fun parseSchedulerConfigFromRoot(schedulerType: String, trainConfig: JSONObject): SchedulerConfig {
    // Backward compatibility: parse from root level
    return when (schedulerType.lowercase()) {
        "linear" -> SchedulerConfig.Linear(
            learningRate = trainConfig.optDouble("learningRate", 1e-4).toFloat(),
            startFactor = trainConfig.optDouble("startFactor", 1.0).toFloat(),
            endFactor = trainConfig.optDouble("endFactor", 0.333).toFloat(),
        )

        "cosine" -> SchedulerConfig.Cosine(
            learningRate = trainConfig.optDouble("learningRate", 1e-4).toFloat(),
            minLearningRate = trainConfig.optDouble("minLearningRate", 0.0).toFloat(),
            warmupSteps = trainConfig.optInt("warmupSteps", 10)
        )

        else -> {
            println("Warning: Unknown scheduler type '$schedulerType', using linear scheduler")
            SchedulerConfig.Linear()
        }
    }
}

fun parseGenerationArguments(jsonPath: String): ORTGenerationConfig {

    if (jsonPath.isBlank()) {
        return ORTGenerationConfig()
    }

    val file = File(jsonPath)
    if (!file.exists()) {
        return ORTGenerationConfig()
    }

    val jsonText = File(jsonPath).readText()
    val json = JSONObject(jsonText)

    // Parse sampling options if present
    val sampling = if (json.has("sampling")) {
        val samplingJson = json.getJSONObject("sampling")
        SamplingOptions(
            method = samplingJson.optString("method", "greedy"),
            temperature = samplingJson.optDouble("temperature", 1.0).toFloat(),
            topK = samplingJson.optInt("topK", 10),
            topP = samplingJson.optDouble("topP", 0.9).toFloat(),
            seed = samplingJson.optInt("seed", 42)
        )
    } else {
        SamplingOptions() // Use default values
    }

    // Parse device options
    val deviceOptions = if (json.has("deviceOptions")) {
        val deviceOptionsJson = json.getJSONObject("deviceOptions")
        parseDeviceOptions(deviceOptionsJson)
    } else {
        // Fallback: try to parse from root level for backward compatibility
        parseDeviceOptionsFromRoot(json)
    }

    return ORTGenerationConfig(
        repoName = json.optString("repoName", "model"),
        onnxName = json.optString("onnxName", ".onnx"),
        type = json.optString("type", "native"),
        maxSequenceLength = json.optInt("maxSequenceLength", 128),
        trackMetrics = json.optBoolean("trackMetrics", true),
        loadMergedWeights = json.optBoolean("loadMergedWeights", true),
        timeStepUpdate = json.optInt("timeStepUpdate", 5),
        systemPrompt = if (json.isNull("systemPrompt")) null else json.optString("systemPrompt", ""),
        sampling = sampling,
        deviceOptions = deviceOptions
    )
}

fun parseRagArguments(jsonPath: String): ORTRagConfig {

    if (jsonPath.isBlank()) {
        return ORTRagConfig()
    }

    val file = File(jsonPath)
    if (!file.exists()) {
        return ORTRagConfig()
    }

    val jsonText = File(jsonPath).readText()
    val json = JSONObject(jsonText)

    // Parse device options
    val deviceOptions = if (json.has("deviceOptions")) {
        val deviceOptionsJson = json.getJSONObject("deviceOptions")
        parseDeviceOptions(deviceOptionsJson)
    } else {
        // Fallback: try to parse from root level for backward compatibility
        parseDeviceOptionsFromRoot(json)
    }

    return ORTRagConfig(
        repoName = json.optString("repoName", "model"),
        onnxName = json.optString("onnxName", "embedding_model"),
        embeddingDimension = json.optInt("embeddingDimension", 256),
        topK = json.optInt("topK", 10),
        searchType = json.optString("searchType", "semantic"),
        minScore = json.optDouble("minScore", 0.0),
        indexingMode = json.optString("indexingMode", "precompute"),
        maxTextLength = json.optInt("maxTextLength", 1024),
        chunkSize = json.optInt("chunkSize", 512),
        chunkOverlap = json.optInt("chunkOverlap", 50),
        deviceOptions = deviceOptions
    )
}

fun copyAssetFile(assetManager: AssetManager, assetPath: String, dstFile: File) {
    // This function copies the asset file named by `assetPath` to the file specified by `dstFile`.
    check(!dstFile.exists() || dstFile.isFile)

    dstFile.parentFile?.mkdirs()

    val assetContents = assetManager.open(assetPath).use { assetStream ->
        val size: Int = assetStream.available()
        val buffer = ByteArray(size)
        assetStream.read(buffer)
        buffer
    }

    java.io.FileOutputStream(dstFile).use { dstStream ->
        dstStream.write(assetContents)
    }
}

fun copyAssetFileOrDir(assetManager: AssetManager, assetPath: String, dstFileOrDir: File) {
    // This function copies the asset file or directory named by `assetPath` to the file or
    // directory specified by `dstFileOrDir`.
    val assets: Array<String>? = assetManager.list(assetPath)
    if (assets!!.isEmpty()) {
        // asset is a file
        copyAssetFile(assetManager, assetPath, dstFileOrDir)
    } else {
        // asset is a dir. loop over dir and copy all files or sub dirs to cache dir
        for (i in assets.indices) {
            val assetChild = (if (assetPath.isEmpty()) "" else "$assetPath/") + assets[i]
            val dstChild = dstFileOrDir.resolve(assets[i])
            copyAssetFileOrDir(assetManager, assetChild, dstChild)
        }
    }
}

fun loadTrainableLayerNamesJSON(fileName: String): Array<String>? {
    val jsonString: String
    try {
        // Get the file path from internal storage
        val file = File(fileName)
        // Open the file input stream to read the file
        val fis = FileInputStream(file)
        val size = fis.available()
        // Create a byte array to hold the file contents
        val buffer = ByteArray(size)
        // Read the file into the buffer
        fis.read(buffer)
        fis.close()
        // Convert the buffer into a string
        jsonString = String(buffer)
    } catch (ex: Exception) {
        ex.printStackTrace()
        return null
    }

    val jsonObject = JSONObject(jsonString)
    val jsonArray = jsonObject.getJSONArray("requires_grad")
    val requiresGradList = mutableListOf<String>()

    for (i in 0 until jsonArray.length()) {
        requiresGradList.add(jsonArray.getString(i))
    }

    return requiresGradList.toTypedArray()
}
package com.martinkorelic.mobiletransformers

import android.content.res.AssetManager
import com.martinkorelic.mobiletransformers.constants.CoreConfigId
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.constants.IndexingMode
import com.martinkorelic.mobiletransformers.constants.MemoryConfigId
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.constants.SearchType
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

    // Parse device options.
    //
    // `low_mem` is the default HERE and nowhere else: exported training configs carry no device
    // section, so this branch decides the allocator for every training run, and `high_perf`'s arena
    // is what got a 270M LoRA run killed at 3.4 GB. Inference keeps `high_perf` — a forward-only
    // session benefits from the arena and does not accumulate a backward plan.
    val deviceOptions = if (trainConfig.has("deviceOptions")) {
        val deviceOptionsJson = trainConfig.getJSONObject("deviceOptions")
        parseDeviceOptions(deviceOptionsJson, defaultMemoryConfigId = MemoryConfigId.LOW_MEM.wire)
    } else {
        parseDeviceOptions(trainConfig, defaultMemoryConfigId = MemoryConfigId.LOW_MEM.wire)
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

/**
 * Parse device options, validating every closed-set field through its enum's `fromWire` (#6).
 *
 * The fields stay `String` because they cross the JNI boundary as strings, but an unrecognized value
 * now throws at the parse boundary instead of being handed to native code that would quietly fall
 * back to a default execution provider or memory profile.
 */
fun parseDeviceOptions(
    deviceOptionsJson: JSONObject,
    /**
     * What to use when the config declares no `memoryConfigId`.
     *
     * Exported `training_config.json` files carry no device section at all, so this default IS the
     * setting for every training run — and `high_perf` (arena + memory pattern) is what took a
     * 270M-parameter LoRA run to 3.4 GB and got it killed. See [ORTTrainingConfig.deviceOptions].
     */
    defaultMemoryConfigId: String = MemoryConfigId.HIGH_PERF.wire,
): DeviceOptions {
    return DeviceOptions(
        enableProfiling = deviceOptionsJson.optBoolean("enableProfiling", false),
        coreConfigId = CoreConfigId.fromWire(deviceOptionsJson.optString("coreConfigId", "opt1")).wire,
        memoryConfigId =
            MemoryConfigId.fromWire(
                deviceOptionsJson.optString("memoryConfigId", defaultMemoryConfigId),
            ).wire,
        executionProvider =
            ExecutionProvider.fromWire(deviceOptionsJson.optString("executionProvider", "cpu")).wire,
    )
}

// Backward compatibility: the same fields read from the config root.
fun parseDeviceOptionsFromRoot(trainConfig: JSONObject): DeviceOptions = parseDeviceOptions(trainConfig)

/**
 * Build the scheduler config from [options], dispatching on the TYPED [SchedulerType].
 *
 * #6/#10 fix: this used to `when` on the raw string and fall through to
 * `println("Warning: Unknown scheduler type ...")` + a silent Linear default. `println` goes to
 * stdout, which Android drops on release builds — so a typo'd `"consine"` trained on a completely
 * different LR schedule with no visible signal. `SchedulerType.fromWire` throws instead, which is the
 * canonical "typed fail-closed parsing" rule. The root-level variant was a verbatim copy of this
 * function; both now share it, differing only in which JSONObject the values are read from.
 */
private fun parseSchedulerConfig(schedulerType: String, options: JSONObject): SchedulerConfig =
    when (SchedulerType.fromWire(schedulerType.lowercase())) {
        SchedulerType.LINEAR -> SchedulerConfig.Linear(
            learningRate = options.optDouble("learningRate", 1e-4).toFloat(),
            startFactor = options.optDouble("startFactor", 1.0).toFloat(),
            endFactor = options.optDouble("endFactor", 0.333).toFloat(),
        )

        SchedulerType.COSINE -> SchedulerConfig.Cosine(
            learningRate = options.optDouble("learningRate", 1e-4).toFloat(),
            minLearningRate = options.optDouble("minLearningRate", 0.0).toFloat(),
            warmupSteps = options.optInt("warmupSteps", 10),
        )
    }

// Backward compatibility: the same fields read from the config root instead of a "scheduler" object.
private fun parseSchedulerConfigFromRoot(schedulerType: String, trainConfig: JSONObject): SchedulerConfig =
    parseSchedulerConfig(schedulerType, trainConfig)

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
            // #6: validated at the boundary — an unknown method used to reach the native
            // sampler, and ORTGeneratorGenAI silently treated it as greedy.
            method = SamplingMethod.fromWire(samplingJson.optString("method", "greedy")).wire,
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
        searchType = SearchType.fromWire(json.optString("searchType", "semantic")).wire,
        minScore = json.optDouble("minScore", 0.0),
        indexingMode = IndexingMode.fromWire(json.optString("indexingMode", "precompute")).wire,
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
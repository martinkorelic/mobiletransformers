package com.martinkorelic.mobiletransformers.internal.config

import com.martinkorelic.mobiletransformers.DatasetOptions
import com.martinkorelic.mobiletransformers.NotImplementedFeatureException
import com.martinkorelic.mobiletransformers.DeviceOptions
import com.martinkorelic.mobiletransformers.ORTGenerationConfig
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.SamplingOptions
import com.martinkorelic.mobiletransformers.SchedulerConfig
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.SamplingConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.IndexingMode
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine

/**
 * Maps the public HF-flavored configs (#17) onto the existing `ORT*Config` data classes verbatim. Defaults
 * on both sides match, so `<PublicConfig>().toOrt()` equals `ORT*Config()` (round-trip unit test) — no
 * behavior shifts. The `ORT*` types never leak into a public signature; this mapping is the only bridge.
 */

fun DeviceConfig.toOrt(): DeviceOptions =
    DeviceOptions(
        enableProfiling = enableProfiling,
        coreConfigId = coreConfigId.wire,
        memoryConfigId = memoryConfigId.wire,
        executionProvider = executionProvider.wire,
    )

fun SamplingConfig.toOrt(): SamplingOptions =
    SamplingOptions(
        method = method.wire,
        temperature = temperature,
        topK = topK,
        topP = topP,
        seed = seed,
    )

fun TrainConfig.toOrt(): ORTTrainingConfig {
    val schedulerConfig =
        when (scheduler) {
            SchedulerType.LINEAR -> SchedulerConfig.Linear(learningRate = learningRate)
            SchedulerType.COSINE ->
                SchedulerConfig.Cosine(
                    learningRate = learningRate,
                    minLearningRate = minLearningRate,
                    warmupSteps = warmupSteps,
                )
        }
    return ORTTrainingConfig(
        batchSize = batchSize,
        numTrainEpochs = epochs,
        maxSteps = maxSteps,
        saveSteps = saveSteps,
        gradAccumSteps = gradientAccumulationSteps,
        mergeWeightsAtEnd = mergeAtEnd,
        saveModelAtEnd = saveAtEnd,
        loadFromState = resumeFromState,
        schedulerType = scheduler.wire,
        schedulerConfig = schedulerConfig,
        deviceOptions = device.toOrt(),
    )
}

fun DatasetConfig.toOrt(): DatasetOptions =
    DatasetOptions(
        trainFile = trainFile,
        datasetBatchSize = datasetBatchSize,
        maxSequenceLength = maxSequenceLength,
        maxDatasetLength = maxDatasetLength,
    )

/**
 * #19/#24: `engine` drives `type` (`"native"`/`"genai"`) and the runtime's post-merge state drives
 * `loadMergedWeights`. Defaults preserve the #17 behavior so `GenerationConfig().toOrt()` still equals
 * `ORTGenerationConfig()` (round-trip test): Native engine + the config's own `loadMerged` flag.
 */
fun GenerationConfig.toOrt(
    engine: InferenceEngine = InferenceEngine.NATIVE,
    mergedLoaded: Boolean = loadMerged,
): ORTGenerationConfig =
    ORTGenerationConfig(
        type = if (engine == InferenceEngine.GENAI) "genai" else "native",
        // #11: `engine` is what ModelRuntimeFactory.selectEngine actually reads. Leaving it null (the
        // pre-fix behavior) pinned every session to Native regardless of `type`, so a GenAI request
        // silently constructed nothing and generate() never completed.
        engine = engine,
        maxSequenceLength = maxNewTokens,
        systemPrompt = systemPrompt,
        loadMergedWeights = mergedLoaded,
        sampling = sampling.toOrt(),
        deviceOptions = device.toOrt(),
    )

fun RagConfig.toOrt(): ORTRagConfig {
    // #27 F7: dynamic indexing is a fail-closed stub in v1.
    if (indexingMode == IndexingMode.DYNAMIC) {
        throw NotImplementedFeatureException("indexingMode=dynamic (v1 supports 'precompute' only)")
    }
    return ORTRagConfig(
        repoName = embeddingRepoId,
        onnxName = embeddingModelFile,
        embeddingDimension = embeddingDimension,
        topK = topK,
        searchType = searchType.wire,
        minScore = minScore,
        indexingMode = indexingMode.wire,
        maxTextLength = maxTextLength,
        chunkSize = chunkSize,
        chunkOverlap = chunkOverlap,
        deviceOptions = device.toOrt(),
    )
}

package com.martinkorelic.mobiletransformers.internal.config

import com.martinkorelic.mobiletransformers.DeviceOptions
import com.martinkorelic.mobiletransformers.ORTGenerationConfig
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.SamplingOptions
import com.martinkorelic.mobiletransformers.SchedulerConfig
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.SamplingConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.SchedulerType

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

fun GenerationConfig.toOrt(): ORTGenerationConfig =
    ORTGenerationConfig(
        type = "native",
        maxSequenceLength = maxNewTokens,
        systemPrompt = systemPrompt,
        loadMergedWeights = loadMerged,
        sampling = sampling.toOrt(),
        deviceOptions = device.toOrt(),
    )

fun RagConfig.toOrt(): ORTRagConfig =
    ORTRagConfig(
        embeddingDimension = embeddingDimension,
        topK = topK,
        searchType = searchType.wire,
        maxTextLength = maxTextLength,
        chunkSize = chunkSize,
        chunkOverlap = chunkOverlap,
        deviceOptions = device.toOrt(),
    )

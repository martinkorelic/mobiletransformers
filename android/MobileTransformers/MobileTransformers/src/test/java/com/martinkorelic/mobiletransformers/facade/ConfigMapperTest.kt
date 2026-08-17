package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.ORTGenerationConfig
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.SchedulerConfig
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import org.junit.Assert.assertEquals
import org.junit.Test

/** #17: public configs map 1:1 to the existing ORT*Config defaults (no behavior shift). */
class ConfigMapperTest {

    @Test
    fun trainConfigDefaultsMatchOrtDefaults() {
        assertEquals(ORTTrainingConfig(), TrainConfig().toOrt())
    }

    @Test
    fun generationConfigDefaultsMatchOrtDefaults() {
        // #11: the mapper now states the engine explicitly rather than leaving it unset. That is the
        // one deliberate delta from the bare ORT defaults — `engine` stays nullable on
        // ORTGenerationConfig so `overrideConfig`'s `override.engine ?: this.engine` fallback keeps
        // working (a non-null default would let an indifferent override clobber a GENAI base).
        assertEquals(ORTGenerationConfig(engine = InferenceEngine.NATIVE), GenerationConfig().toOrt())
    }

    @Test
    fun ragConfigDefaultsMatchOrtDefaults() {
        assertEquals(ORTRagConfig(), RagConfig().toOrt())
    }

    @Test
    fun trainConfigNonDefaultsPropagate() {
        val ort = TrainConfig(epochs = 3, batchSize = 8, maxSteps = 50, mergeAtEnd = false).toOrt()
        assertEquals(3, ort.numTrainEpochs)
        assertEquals(8, ort.batchSize)
        assertEquals(50, ort.maxSteps)
        assertEquals(false, ort.mergeWeightsAtEnd)
    }

    @Test
    fun cosineSchedulerMapsToCosineConfig() {
        val ort =
            TrainConfig(scheduler = SchedulerType.COSINE, warmupSteps = 20, minLearningRate = 1e-5f).toOrt()
        assertEquals("cosine", ort.schedulerType)
        val cfg = ort.schedulerConfig
        assertEquals(true, cfg is SchedulerConfig.Cosine)
        cfg as SchedulerConfig.Cosine
        assertEquals(20, cfg.warmupSteps)
    }

    @Test
    fun generationMaxNewTokensMapsToMaxSequenceLength() {
        assertEquals(256, GenerationConfig(maxNewTokens = 256).toOrt().maxSequenceLength)
        assertEquals("native", GenerationConfig().toOrt().type)
    }

    @Test
    fun deviceConfigMapsProvider() {
        val ort = DeviceConfig(executionProvider = ExecutionProvider.NNAPI).toOrt()
        assertEquals("nnapi", ort.executionProvider)
    }
}

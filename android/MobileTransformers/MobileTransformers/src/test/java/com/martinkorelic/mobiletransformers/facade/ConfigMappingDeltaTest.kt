package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #19/#24: the engine- and merge-state-driven bits of the generation mapping, plus the dataset mapping.
 * (The 1:1 defaults round-trip is covered by [ConfigMapperTest].)
 */
class ConfigMappingDeltaTest {

    @Test
    fun engineDrivesGenerationType() {
        assertEquals("native", GenerationConfig().toOrt(InferenceEngine.NATIVE).type)
        assertEquals("genai", GenerationConfig().toOrt(InferenceEngine.GENAI).type)
    }

    /**
     * #11 regression: the mapper used to set only [ORTGenerationConfig.type] and leave `engine` null.
     * `ModelRuntimeFactory.selectEngine` reads `engine`, not `type`, so every session — including an
     * explicit GENAI request — resolved to NATIVE, and the GenAI path was unreachable end to end.
     */
    @Test
    fun engineFieldIsCarriedThroughForRuntimeSelection() {
        assertEquals(InferenceEngine.NATIVE, GenerationConfig().toOrt(InferenceEngine.NATIVE).engine)
        assertEquals(InferenceEngine.GENAI, GenerationConfig().toOrt(InferenceEngine.GENAI).engine)
        // The no-arg overload keeps the #17 default rather than leaving selection unspecified.
        assertEquals(InferenceEngine.NATIVE, GenerationConfig().toOrt().engine)
    }

    /** The two fields must never disagree — `type` is derived from `engine` by the same mapper. */
    @Test
    fun engineAndTypeAgree() {
        for (engine in InferenceEngine.entries) {
            val ort = GenerationConfig().toOrt(engine)
            assertEquals(engine, ort.engine)
            assertEquals(if (engine == InferenceEngine.GENAI) "genai" else "native", ort.type)
        }
    }

    @Test
    fun mergeStateDrivesLoadMergedWeights() {
        assertFalse(GenerationConfig().toOrt(InferenceEngine.NATIVE, mergedLoaded = false).loadMergedWeights)
        assertTrue(GenerationConfig().toOrt(InferenceEngine.NATIVE, mergedLoaded = true).loadMergedWeights)
    }

    @Test
    fun defaultsPreserveNativeAndConfigLoadFlag() {
        // No-arg overload must still equal the #17 behavior (native + the config's own loadMerged=false).
        val ort = GenerationConfig().toOrt()
        assertEquals("native", ort.type)
        assertFalse(ort.loadMergedWeights)
    }

    @Test
    fun datasetConfigMapsToDatasetOptions() {
        val ds = DatasetConfig(trainFile = "squad", maxSequenceLength = 99, datasetBatchSize = 7, maxDatasetLength = 33)
        val opts = ds.toOrt()
        assertEquals("squad", opts.trainFile)
        assertEquals(99, opts.maxSequenceLength)
        assertEquals(7, opts.datasetBatchSize)
        assertEquals(33, opts.maxDatasetLength)
    }
}

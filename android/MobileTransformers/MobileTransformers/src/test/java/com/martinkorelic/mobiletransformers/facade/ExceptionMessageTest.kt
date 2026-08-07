package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.EngineUnavailableException
import com.martinkorelic.mobiletransformers.FeatureNotInstalledException
import com.martinkorelic.mobiletransformers.MissingArtifactException
import com.martinkorelic.mobiletransformers.ModelNotInstalledException
import com.martinkorelic.mobiletransformers.NotImplementedFeatureException
import com.martinkorelic.mobiletransformers.PeftMismatchException
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #19: the facade error hierarchy fails closed with friendly, path-naming messages.
 */
class ExceptionMessageTest {

    @Test
    fun missingArtifactNamesTheExactPath() {
        val path = "/data/pkg/train/training_config.json"
        val ex = MissingArtifactException(ModelFeature.Training, path)
        assertTrue(ex.message!!.contains(path))
        assertTrue(ex.message!!.contains("Training"))
    }

    @Test
    fun modelNotInstalledNamesRepoAndCache() {
        val ex = ModelNotInstalledException("org/model", "/data/cache")
        assertTrue(ex.message!!.contains("org/model"))
        assertTrue(ex.message!!.contains("/data/cache"))
    }

    @Test
    fun featureNotInstalledListsInstalled() {
        val ex = FeatureNotInstalledException(ModelFeature.Rag, setOf(ModelFeature.Inference, ModelFeature.Training))
        assertTrue(ex.message!!.contains("Rag"))
        assertTrue(ex.message!!.contains("Inference"))
        assertTrue(ex.message!!.contains("Training"))
    }

    @Test
    fun engineUnavailableNamesEngineAndReason() {
        val ex = EngineUnavailableException(InferenceEngine.GENAI, "genai_config.json not found")
        assertTrue(ex.message!!.contains("GENAI"))
        assertTrue(ex.message!!.contains("genai_config.json"))
    }

    @Test
    fun peftMismatchListsSupported() {
        val ex = PeftMismatchException("lora", listOf("mars (optimization_level=1)"))
        assertTrue(ex.message!!.contains("lora"))
        assertTrue(ex.message!!.contains("mars"))
    }

    @Test
    fun notImplementedNamesFeature() {
        assertTrue(NotImplementedFeatureException("pushAdapter").message!!.contains("pushAdapter"))
    }
}

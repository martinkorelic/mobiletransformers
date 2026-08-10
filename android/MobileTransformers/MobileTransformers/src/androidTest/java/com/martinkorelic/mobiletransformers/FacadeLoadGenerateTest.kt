package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** #17 device leg: fromPretrained → generate one token over a pushed package. */
@RunWith(AndroidJUnit4::class)
class FacadeLoadGenerateTest {

    @Test
    fun fromPretrainedGeneratesAndReportsInferenceFeature() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
        )
        try {
            assertTrue(model.capabilities.availableFeatures.contains(ModelFeature.Inference))
            val result = model.generate("Hello", GenerationConfig(maxNewTokens = 1))
            assertTrue("generation produced no tokens", result.tokenCount >= 0)
        } finally {
            model.close()
        }
    }
}

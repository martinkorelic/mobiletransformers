package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.SamplingConfig
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.runtime.GenAiSupport
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #11 + #24 device leg (Gate 0.1 #1): the SAME `inference/` package under the Native and GenAI engines
 * must yield the same greedy first token. Skips unless the package carries a genai_config AND GenAI is
 * available on the device.
 */
@RunWith(AndroidJUnit4::class)
class DualEngineParityTest {

    @Test
    fun nativeAndGenaiAgreeOnGreedyFirstToken() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        assumeTrue(
            "package has no genai_config.json",
            File(root, "$repoId/inference/genai_config.json").isFile,
        )
        assumeTrue("GenAI engine unavailable on this device", GenAiSupport.available())

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val greedy = GenerationConfig(maxNewTokens = 1, sampling = SamplingConfig(method = SamplingMethod.GREEDY))

        val native = MobileTransformers.fromPretrained(ctx, repoId, cacheDir = root.absolutePath, engine = InferenceEngine.NATIVE)
        val nativeText = try {
            native.generate("Hello", greedy).text
        } finally {
            native.close()
        }

        val genai = MobileTransformers.fromPretrained(ctx, repoId, cacheDir = root.absolutePath, engine = InferenceEngine.GENAI)
        val genaiText = try {
            genai.generate("Hello", greedy).text
        } finally {
            genai.close()
        }

        assertEquals("dual-engine greedy first-token mismatch", nativeText, genaiText)
    }
}

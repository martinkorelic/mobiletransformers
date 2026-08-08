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
import org.junit.Assert.assertTrue
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
        // ModelRuntimeFactory falls back to Native transparently when GenAI cannot load. Without this
        // assertion the test happily compared Native with Native and reported cross-engine parity —
        // Gate 0.1 #1 "passing" while GenAI had never run. Fail instead of proving nothing.
        assertEquals(
            "requested GENAI but the runtime fell back (see logcat ModelRuntimeFactory); " +
                "this test cannot demonstrate parity",
            InferenceEngine.GENAI,
            genai.capabilities.engine,
        )
        val genaiText = try {
            genai.generate("Hello", greedy).text
        } finally {
            genai.close()
        }

        assertEquals("dual-engine greedy first-token mismatch", nativeText, genaiText)
    }

    /**
     * #11 + #24 callback-sequence parity lock.
     *
     * [GenerateCallback]'s docstring promises every engine drives the *identical ordered sequence* —
     * `onStartGeneration` → N×`onPartialResult` → `onCompletion` — but nothing asserted it, so the two
     * engines could have differed in event order, in whether the final token also arrives as a partial,
     * or in emitting a start event at all, and every test would still have passed.
     *
     * Recording the ordered event names (not just counts) is the point: a sequence that emits the right
     * events in the wrong order, or completes before its last partial, is a real API break for a caller
     * driving a UI off these callbacks.
     */
    @Test
    fun bothEnginesEmitTheSameOrderedCallbackSequence(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        assumeTrue(
            "package has no genai_config.json",
            File(root, "$repoId/inference/genai_config.json").isFile,
        )
        assumeTrue("GenAI engine unavailable on this device", GenAiSupport.available())

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val greedy = GenerationConfig(
            maxNewTokens = 6,
            sampling = SamplingConfig(method = SamplingMethod.GREEDY),
        )

        fun sequenceFor(engine: InferenceEngine): List<String> = runBlocking {
            val model = MobileTransformers.fromPretrained(
                ctx, repoId, cacheDir = root.absolutePath, engine = engine,
            )
            // An explicitly requested engine no longer falls back silently, but assert anyway: a
            // sequence recorded off the wrong engine would "prove" parity between Native and Native.
            assertEquals(
                "requested $engine but got ${model.capabilities.engine}",
                engine,
                model.capabilities.engine,
            )
            val events = mutableListOf<String>()
            try {
                model.generate(
                    "The capital of France is",
                    greedy,
                    object : GenerateCallback {
                        override fun onStartGeneration(progress: GenerateProgress) {
                            events.add("start")
                        }

                        override fun onPartialResult(progress: GenerateProgress) {
                            events.add("partial")
                        }

                        override fun onCompletion(progress: GenerateProgress) {
                            events.add("completion")
                        }

                        override fun onError(error: Throwable) {
                            events.add("error:${error::class.java.simpleName}")
                        }
                    },
                )
            } finally {
                model.close()
            }
            events
        }

        val nativeEvents = sequenceFor(InferenceEngine.NATIVE)
        val genaiEvents = sequenceFor(InferenceEngine.GENAI)

        // The contract itself, checked on the floor engine before the engines are compared — otherwise
        // two engines that are identically wrong would pass as "parity".
        assertTrue("native emitted no callbacks at all", nativeEvents.isNotEmpty())
        assertEquals("sequence must open with onStartGeneration", "start", nativeEvents.first())
        assertEquals("sequence must close with onCompletion", "completion", nativeEvents.last())
        assertTrue(
            "no onPartialResult between start and completion: $nativeEvents",
            nativeEvents.count { it == "partial" } > 0,
        )
        assertTrue(
            "start/completion must occur exactly once: $nativeEvents",
            nativeEvents.count { it == "start" } == 1 && nativeEvents.count { it == "completion" } == 1,
        )

        assertEquals(
            "cross-engine callback sequence mismatch (ordered): native=$nativeEvents genai=$genaiEvents",
            nativeEvents,
            genaiEvents,
        )
    }
}

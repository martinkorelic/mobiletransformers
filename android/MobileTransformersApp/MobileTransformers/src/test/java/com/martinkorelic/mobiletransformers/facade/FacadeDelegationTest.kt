package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #17: [MobileTransformerModel] delegates each call to its [ModelSession] and closes it. Uses a hand-written
 * fake session (no mock framework on the test classpath) — proving the adapter contract without a device.
 */
class FacadeDelegationTest {

    private class FakeSession(override val capabilities: RuntimeCapabilities) : ModelSession {
        val calls = mutableListOf<String>()

        override suspend fun train(dataset: DatasetConfig, config: TrainConfig): TrainingResult {
            calls += "train"
            return TrainingResult(finalStep = 7, merged = config.mergeAtEnd)
        }

        override suspend fun merge(): MergeResult {
            calls += "merge"
            return MergeResult(merged = true)
        }

        override suspend fun generate(prompt: String, config: GenerationConfig): GenerationResult {
            calls += "generate:$prompt"
            return GenerationResult(text = "hello", tokenCount = 1)
        }

        override suspend fun retrieve(query: String, config: RagConfig): RetrievalResult {
            calls += "retrieve:$query"
            return RetrievalResult()
        }

        override fun close() {
            calls += "close"
        }
    }

    private fun caps() =
        RuntimeCapabilities(
            engine = InferenceEngine.NATIVE,
            supportsTraining = true,
            supportsMerge = true,
            supportsRag = false,
            supportsEmbedding = false,
        )

    @Test
    fun delegatesEveryMethodToSession() = runBlocking {
        val fake = FakeSession(caps())
        val model = MobileTransformerModel(fake, caps())

        assertEquals(7, model.train(DatasetConfig(), TrainConfig(mergeAtEnd = true)).finalStep)
        assertTrue(model.merge().merged)
        assertEquals("hello", model.generate("hi").text)
        model.retrieve("q")
        model.close()

        assertEquals(listOf("train", "merge", "generate:hi", "retrieve:q", "close"), fake.calls)
    }

    @Test
    fun capabilitiesArePassedThrough() {
        val model = MobileTransformerModel(FakeSession(caps()), caps())
        assertEquals(InferenceEngine.NATIVE, model.capabilities.engine)
        assertTrue(model.capabilities.supportsTraining)
    }
}

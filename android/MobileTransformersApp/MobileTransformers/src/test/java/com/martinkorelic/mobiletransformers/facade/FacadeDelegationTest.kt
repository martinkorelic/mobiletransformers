package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.GenerateCallback
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.RetrieveCallback
import com.martinkorelic.mobiletransformers.TrainCallback
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.PushResult
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #17/#19: [MobileTransformerModel] delegates each call to its [ModelSession] and closes it. Uses a
 * hand-written fake session (no mock framework on the test classpath) — proving the adapter contract
 * (now including `applyPeft` + the public callbacks) without a device.
 */
class FacadeDelegationTest {

    private class FakeSession(override val capabilities: RuntimeCapabilities) : ModelSession {
        val calls = mutableListOf<String>()

        override suspend fun applyPeft(peft: PeftConfig) {
            calls += "applyPeft:${peft::class.simpleName}"
        }

        override suspend fun train(
            dataset: DatasetConfig,
            config: TrainConfig,
            callback: TrainCallback?,
        ): TrainingResult {
            calls += "train"
            return TrainingResult(finalStep = 7, merged = config.mergeAtEnd)
        }

        override suspend fun merge(): MergeResult {
            calls += "merge"
            return MergeResult(merged = true)
        }

        override suspend fun generate(
            prompt: String,
            config: GenerationConfig,
            callback: GenerateCallback?,
        ): GenerationResult {
            calls += "generate:$prompt"
            return GenerationResult(text = "hello", tokenCount = 1)
        }

        override suspend fun retrieve(
            query: String,
            config: RagConfig,
            callback: RetrieveCallback?,
        ): RetrievalResult {
            calls += "retrieve:$query"
            return RetrievalResult()
        }

        override suspend fun ingest(
            path: String,
            config: RagConfig,
            progress: com.martinkorelic.mobiletransformers.rag.IngestionProgress?,
        ): com.martinkorelic.mobiletransformers.runtime.IngestResult {
            calls += "ingest:$path"
            return com.martinkorelic.mobiletransformers.runtime.IngestResult(0)
        }

        override suspend fun generateWithRag(
            query: String,
            rag: RagConfig,
            generation: GenerationConfig,
            promptStrategy: com.martinkorelic.mobiletransformers.rag.PromptStrategy,
        ): com.martinkorelic.mobiletransformers.runtime.GroundedResult {
            calls += "generateWithRag:$query"
            return com.martinkorelic.mobiletransformers.runtime.GroundedResult("grounded")
        }

        override suspend fun pushAdapter(hubConfig: HubConfig, repoId: String): PushResult {
            calls += "pushAdapter:$repoId"
            return PushResult(repoId)
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
        val model = MobileTransformerModel(fake, caps(), "test/repo")

        model.applyPeft(PeftConfig.Lora())
        assertEquals(7, model.train(DatasetConfig(), TrainConfig(mergeAtEnd = true)).finalStep)
        assertTrue(model.merge().merged)
        assertEquals("hello", model.generate("hi").text)
        model.retrieve("q")
        model.close()

        assertEquals(
            listOf("applyPeft:Lora", "train", "merge", "generate:hi", "retrieve:q", "close"),
            fake.calls,
        )
    }

    @Test
    fun capabilitiesAndRepoIdArePassedThrough() {
        val model = MobileTransformerModel(FakeSession(caps()), caps(), "test/repo")
        assertEquals(InferenceEngine.NATIVE, model.capabilities.engine)
        assertEquals(InferenceEngine.NATIVE, model.engine)
        assertEquals("test/repo", model.repoId)
        assertTrue(model.capabilities.supportsTraining)
    }
}

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
import com.martinkorelic.mobiletransformers.federated.FederatedConfig
import com.martinkorelic.mobiletransformers.federated.FederatedRoundResult
import com.martinkorelic.mobiletransformers.federated.LocalRoundTraining
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.PushResult
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import com.martinkorelic.mobiletransformers.training.TrainingJob
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
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

        // #18: TrainingJob is inherently LLMRepository-backed (native handles + Android Context), so a
        // JVM fake cannot construct one. Recording the call still proves the facade delegates rather
        // than building its own job — which is the contract this test exists to pin.
        override fun trainingJob(): TrainingJob {
            calls += "trainingJob"
            throw UnsupportedOperationException("fake session has no repository")
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

        override suspend fun federatedRound(
            config: FederatedConfig,
            globalRecord: ByteArray?,
            roundNumber: Int,
            localTraining: LocalRoundTraining,
            metrics: Map<String, Double>,
            train: Boolean,
        ): FederatedRoundResult {
            calls += "federatedRound:$roundNumber:import=${globalRecord != null}:train=$train"
            return FederatedRoundResult(
                round = roundNumber,
                importedTensors = if (globalRecord == null) 0 else 3,
                update = byteArrayOf(1, 2, 3, 4),
                trainedLocally = train,
            )
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

    /**
     * #18: `trainingJob()` must reach the session. Before it existed, the entire `training/` package
     * (status/events flows, cooperative cancel, checkpoint/resume) had zero non-test callers and was
     * unreachable from the public API.
     */
    @Test
    fun trainingJobIsReachableFromTheFacadeAndDelegates() {
        val fake = FakeSession(caps())
        val model = MobileTransformerModel(fake, caps(), "test/repo")
        assertThrows(UnsupportedOperationException::class.java) { model.trainingJob() }
        assertEquals(listOf("trainingJob"), fake.calls)
    }

    /**
     * #35/#36 was the one shipped capability with no facade door: `FederatedTrainingRepository.forSession`
     * is `internal` and hand-assembly needs `NativeCheckpointTensorStore(trainer: ORTTrainerNative)`, so a
     * facade-only app could not run a round at all. This pins that the door exists and that every
     * argument reaches the session rather than being defaulted away en route.
     */
    @Test
    fun federatedRoundIsReachableFromTheFacadeAndPassesItsArgumentsThrough() = runBlocking {
        val fake = FakeSession(caps())
        val model = MobileTransformerModel(fake, caps(), "test/repo")

        val result = model.federatedRound(
            config = FederatedConfig(gatewayUrl = "https://gw.example", clientAuthToken = "t"),
            globalRecord = byteArrayOf(9),
            roundNumber = 4,
            localTraining = { },
        )

        assertEquals(4, result.round)
        assertEquals(3, result.importedTensors)
        assertEquals(4, result.payloadBytes)
        assertTrue(result.trainedLocally)
        // Round number, the presence of a global record and the train flag all had to survive the hop.
        assertEquals(listOf("federatedRound:4:import=true:train=true"), fake.calls)
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

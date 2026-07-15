package com.martinkorelic.mobiletransformers.internal.runtime

import com.martinkorelic.mobiletransformers.InferenceProgress
import com.martinkorelic.mobiletransformers.RagResult
import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import com.martinkorelic.mobiletransformers.repository.GenerationCallback
import com.martinkorelic.mobiletransformers.repository.InferenceRepository
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.repository.RagCallback
import com.martinkorelic.mobiletransformers.repository.RagRepository
import com.martinkorelic.mobiletransformers.repository.TrainingCallback
import com.martinkorelic.mobiletransformers.repository.TrainingRepository
import com.martinkorelic.mobiletransformers.runtime.GenerationResult
import com.martinkorelic.mobiletransformers.runtime.MergeResult
import com.martinkorelic.mobiletransformers.runtime.ModelSession
import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import kotlinx.coroutines.CompletableDeferred

/**
 * The only [ModelSession] implementation this pass (#17): adapts the existing `LLMRepository` +
 * `Training/Inference/Rag` repositories to the facade contract. It maps the public configs via
 * [toOrt] and the repositories' callback streams to the public result types. No engine logic lives here —
 * generation is delegated through the repositories to whichever engine #11's factory selected.
 */
internal class RepositoryBackedModelSession(
    private val repo: LLMRepository,
    override val capabilities: RuntimeCapabilities,
    private val inferencePackagePath: String? = null,
) : ModelSession {

    private val training = TrainingRepository(repo)
    private val inference = InferenceRepository(repo)
    private val rag = RagRepository(repo)

    override suspend fun train(dataset: DatasetConfig, config: TrainConfig): TrainingResult {
        var last: TrainingProgress? = null
        val callback =
            object : TrainingCallback {
                override fun onStepEnd(trainingProgress: TrainingProgress) {
                    last = trainingProgress
                }

                override fun onCompletion(trainingProgress: TrainingProgress) {
                    last = trainingProgress
                }
            }
        training.performTraining(config.toOrt(), callback)
        val p = last
        return TrainingResult(
            finalStep = p?.currentStep ?: 0,
            finalEpoch = p?.currentEpoch ?: 0,
            finalLoss = p?.totalLoss ?: 0f,
            totalDurationMs = p?.totalDurationMs ?: 0L,
            merged = config.mergeAtEnd,
        )
    }

    override suspend fun merge(): MergeResult {
        training.endTraining(saveModel = true)
        return MergeResult(merged = true, inferencePackagePath = inferencePackagePath)
    }

    override suspend fun generate(prompt: String, config: GenerationConfig): GenerationResult {
        val done = CompletableDeferred<InferenceProgress?>()
        val text = StringBuilder()
        var tokens = 0
        val callback =
            object : GenerationCallback {
                override fun onPartialResult(inferenceProgress: InferenceProgress) {
                    text.append(inferenceProgress.token)
                    tokens = inferenceProgress.totalDecodedTokens
                }

                override fun onCompletion(inferenceProgress: InferenceProgress) {
                    if (!done.isCompleted) done.complete(inferenceProgress)
                }

                override fun onError(error: Throwable) {
                    if (!done.isCompleted) done.completeExceptionally(error)
                }
            }
        inference.generate(prompt, config.toOrt(), callback)
        val finalProgress = done.await()
        return GenerationResult(
            text = text.toString(),
            tokenCount = finalProgress?.totalDecodedTokens ?: tokens,
            generationTimeMs = finalProgress?.generationTimeMs ?: 0L,
            avgTokensPerSecond = finalProgress?.avgTokensPerSecond ?: 0.0,
        )
    }

    override suspend fun retrieve(query: String, config: RagConfig): RetrievalResult {
        var result: RagResult? = null
        val callback =
            object : RagCallback {
                override fun onQueryResults(queryResult: RagResult) {
                    result = queryResult
                }
            }
        rag.initialize(config.toOrt(), callback)
        rag.query(query, ragCallback = callback)
        val matches =
            result?.documents?.map { RetrievalMatch(it.document.text, it.score) } ?: emptyList()
        return RetrievalResult(matches = matches, queryTimeMs = result?.queryTimeMs ?: 0L)
    }

    override fun close() {
        repo.resetInference()
        repo.resetTraining()
    }
}

package com.martinkorelic.mobiletransformers.internal.runtime

import com.martinkorelic.mobiletransformers.GenerateCallback
import com.martinkorelic.mobiletransformers.GenerateProgress
import com.martinkorelic.mobiletransformers.InferenceProgress
import com.martinkorelic.mobiletransformers.MissingArtifactException
import com.martinkorelic.mobiletransformers.NotImplementedFeatureException
import com.martinkorelic.mobiletransformers.RagResult
import com.martinkorelic.mobiletransformers.RetrieveCallback
import com.martinkorelic.mobiletransformers.Tasks
import com.martinkorelic.mobiletransformers.TrainCallback
import com.martinkorelic.mobiletransformers.TrainProgress
import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.federated.FederatedConfig
import com.martinkorelic.mobiletransformers.federated.FederatedRoundResult
import com.martinkorelic.mobiletransformers.federated.FederatedTrainingRepository
import com.martinkorelic.mobiletransformers.federated.LocalRoundTraining
import com.martinkorelic.mobiletransformers.hub.AdapterUploadDisabledException
import com.martinkorelic.mobiletransformers.hub.AdapterUploader
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import com.martinkorelic.mobiletransformers.internal.config.PeftSupport
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.PromptAssembler
import com.martinkorelic.mobiletransformers.rag.PromptStrategy
import com.martinkorelic.mobiletransformers.runtime.GroundedResult
import com.martinkorelic.mobiletransformers.runtime.IngestResult
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
import com.martinkorelic.mobiletransformers.runtime.PushResult
import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import com.martinkorelic.mobiletransformers.runtime.TrainingResult
import com.martinkorelic.mobiletransformers.training.TrainingJob
import com.martinkorelic.mobiletransformers.training.TrainingJobManager
import java.io.File
import kotlinx.coroutines.CompletableDeferred

/**
 * The only [ModelSession] implementation (#17, extended by #19): adapts the existing `LLMRepository` +
 * `Training/Inference/Rag` repositories to the facade contract. It maps the public configs via [toOrt],
 * validates PEFT selection, threads public callbacks over the repository callback streams, and drives
 * the engine-aware `ORTGenerationConfig.type` / post-merge `loadMergedWeights`. No engine logic lives
 * here — generation is delegated through the repositories to whichever engine #11's factory selected.
 */
internal class RepositoryBackedModelSession(
    private val repo: LLMRepository,
    override val capabilities: RuntimeCapabilities,
    private val modelDir: File,
    private val inferencePackagePath: String? = null,
) : ModelSession {

    private val training = TrainingRepository(repo)
    private val inference = InferenceRepository(repo)
    private val rag = RagRepository(repo)

    // #18: one lifecycle job per repo, so status/events/cancel/canResume are reachable from the facade.
    private val trainingJobs = TrainingJobManager(repo)

    private val engine = capabilities.engine

    // #19: once merge()/mergeAtEnd has run, generation loads the merged external initializers (#23).
    private var mergedWeightsLoaded = false

    // #19: the validated PEFT selection to apply on the next train() (rank/alpha overrides).
    private var appliedPeft: PeftConfig? = null

    /** #33: opened on the first `classify`, because most packages never call it. */
    private var classifier: ClassifierSession? = null

    override suspend fun applyPeft(peft: PeftConfig) {
        if (!repo.isTrainingAvailable) {
            throw MissingArtifactException(
                ModelFeature.Training,
                File(modelDir, "train/training_config.json").absolutePath,
            )
        }
        val cfgFile = File(modelDir, "train/training_config.json")
        val pkg = if (cfgFile.isFile) PeftSupport.packageTaxonomy(cfgFile.readText()) else null
        PeftSupport.validate(peft, pkg) // throws PeftMismatchException on a mismatch
        appliedPeft = peft
    }

    override suspend fun train(
        dataset: DatasetConfig,
        config: TrainConfig,
        callback: TrainCallback?,
    ): TrainingResult {
        var last: TrainingProgress? = null
        val adapter =
            object : TrainingCallback {
                override fun onModelLoadStart() = callback?.onModelLoadStart() ?: Unit

                override fun onModelLoadEnd() = callback?.onModelLoadEnd() ?: Unit

                override fun onDataLoadEnd(totalSteps: Int, stepsPerEpoch: Int) =
                    callback?.onDataLoadEnd(totalSteps, stepsPerEpoch) ?: Unit

                override fun onStepEnd(trainingProgress: TrainingProgress) {
                    last = trainingProgress
                    callback?.onStepEnd(trainingProgress.toPublic())
                }

                override fun onEpochEnd(trainingProgress: TrainingProgress) {
                    last = trainingProgress
                    callback?.onEpochEnd(trainingProgress.toPublic())
                }

                override fun onMergeStart(trainingProgress: TrainingProgress) =
                    callback?.onMergeStart(trainingProgress.toPublic()) ?: Unit

                override fun onMergeEnd(trainingProgress: TrainingProgress) {
                    mergedWeightsLoaded = true
                    callback?.onMergeEnd(trainingProgress.toPublic())
                }

                override fun onCompletion(trainingProgress: TrainingProgress) {
                    last = trainingProgress
                    callback?.onCompletion(trainingProgress.toPublic())
                }

                override fun onError(error: Throwable) = callback?.onError(error) ?: Unit
            }

        val ortConfig = config.toOrt(repo.trainingConfig).copy(
            datasetOptions = dataset.toOrt(),
            // The caller supplies the data, so the caller names its preprocessor; fall back to
            // whatever the package declared.
            // Resolved (and rejected) here rather than in the trainer's constructor, which runs on
            // LLMRepository's scope where a throw kills the process instead of reaching the caller.
            taskName = Tasks.resolve(dataset.task, repo.trainingConfig.taskName),
        )
        training.performTraining(ortConfig, adapter)
        if (config.mergeAtEnd) mergedWeightsLoaded = true

        val p = last
        return TrainingResult(
            finalStep = p?.currentStep ?: 0,
            finalEpoch = p?.currentEpoch ?: 0,
            finalLoss = p?.totalLoss ?: 0f,
            totalDurationMs = p?.totalDurationMs ?: 0L,
            merged = config.mergeAtEnd,
            // #18: these two were declared on TrainingResult and never populated — the checkpoint
            // projection is exactly what TrainingJob already reads from training_state.json.
            checkpoint = trainingJob().checkpoint(),
            // Null unless trainingConfig.profileMetrics was on — see ORTTrainerNative.lastSummary.
            summary = repo.ortTrainerNative?.lastSummary?.toPublic(),
        )
    }

    /** #18 [ModelSession.trainingJob]. */
    override fun trainingJob(): TrainingJob = trainingJobs.getOrCreate(modelDir.name)

    override suspend fun merge(): MergeResult {
        training.endTraining(saveModel = true)
        mergedWeightsLoaded = true
        return MergeResult(merged = true, inferencePackagePath = inferencePackagePath)
    }

    override suspend fun generate(
        prompt: String,
        config: GenerationConfig,
        callback: GenerateCallback?,
    ): GenerationResult {
        val done = CompletableDeferred<InferenceProgress?>()
        val text = StringBuilder()
        var tokens = 0
        val adapter =
            object : GenerationCallback {
                override fun onStartGeneration(inferenceProgress: InferenceProgress) {
                    callback?.onStartGeneration(inferenceProgress.toPublic())
                }

                override fun onPartialResult(inferenceProgress: InferenceProgress) {
                    text.append(inferenceProgress.token)
                    tokens = inferenceProgress.totalDecodedTokens
                    callback?.onPartialResult(inferenceProgress.toPublic())
                }

                override fun onCompletion(inferenceProgress: InferenceProgress) {
                    callback?.onCompletion(inferenceProgress.toPublic())
                    if (!done.isCompleted) done.complete(inferenceProgress)
                }

                override fun onError(error: Throwable) {
                    callback?.onError(error)
                    if (!done.isCompleted) done.completeExceptionally(error)
                }
            }
        inference.generate(prompt, config.toOrt(engine, mergedWeightsLoaded), adapter)
        val finalProgress = done.await()
        return GenerationResult(
            text = text.toString(),
            tokenCount = finalProgress?.totalDecodedTokens ?: tokens,
            generationTimeMs = finalProgress?.generationTimeMs ?: 0L,
            avgTokensPerSecond = finalProgress?.avgTokensPerSecond ?: 0.0,
            promptTokenCount = finalProgress?.promptTokenCount ?: 0,
            contextLimit = finalProgress?.contextLimit ?: 0,
        )
    }

    override suspend fun retrieve(
        query: String,
        config: RagConfig,
        callback: RetrieveCallback?,
    ): RetrievalResult {
        var result: RagResult? = null
        val adapter =
            object : RagCallback {
                override fun onQueryResults(queryResult: RagResult) {
                    result = queryResult
                    callback?.onQueryResults(queryResult.toPublic())
                }

                override fun onQueryEnd() = callback?.onQueryEnd() ?: Unit

                override fun onError(error: Throwable) = callback?.onError(error) ?: Unit
            }
        rag.initialize(config.toOrt(repo.ragConfig), adapter)
        rag.query(query, ragCallback = adapter)
        return result?.toPublic() ?: RetrievalResult()
    }

    override suspend fun ingest(
        path: String,
        config: RagConfig,
        progress: IngestionProgress?,
    ): IngestResult = IngestResult(chunkCount = rag.ingest(path, config.toOrt(repo.ragConfig), progress))

    /**
     * #33: run the classification head.
     *
     * Guarded on [RuntimeCapabilities.supportsClassification] rather than attempted and allowed to
     * fail somewhere in the runtime: a decoder asked to classify would run its graph and hand back
     * `vocabSize` floats read as labels, which is not an error anywhere — just a confident nonsense
     * answer. Fail at the door instead.
     */
    override suspend fun classify(
        text: String,
        device: com.martinkorelic.mobiletransformers.config.DeviceConfig,
        topK: Int,
    ): com.martinkorelic.mobiletransformers.runtime.ClassificationResult {
        if (!capabilities.isClassifier) {
            throw NotImplementedFeatureException(
                "this package's task is '${capabilities.task.declaredTask ?: "undeclared"}', not " +
                    "text-classification — classify() would read generation logits as class scores",
            )
        }
        val session = classifier ?: ClassifierSession(
            context = repo.applicationContext,
            cacheDir = modelDir.parent ?: modelDir.absolutePath,
            sanitizedRepoId = modelDir.name,
            task = capabilities.task,
        ).also { classifier = it }
        return session.classify(text, device, topK)
    }

    override suspend fun generateWithRag(
        query: String,
        rag: RagConfig,
        generation: GenerationConfig,
        promptStrategy: PromptStrategy,
    ): GroundedResult {
        // #27: reuse the existing retrieve + generate legs — retrieve → assemble → generate.
        val retrieval = retrieve(query, rag, null)
        val prompt = PromptAssembler.assemble(query, retrieval.matches, promptStrategy)
        val generated = generate(prompt, generation, null)
        return GroundedResult(
            text = generated.text,
            matches = retrieval.matches,
            prompt = prompt,
            generation = generated,
        )
    }

    override suspend fun pushAdapter(hubConfig: HubConfig, repoId: String): PushResult {
        // #22: default-off, privacy-gated. When disabled (default), fail closed pointing at the desktop
        // path. When enabled, prepare + validate the card (pure); the authenticated Hub POST is the device
        // leg (still NotImplemented here). Product path is device -> desktop -> Python `push-adapter`.
        if (!AdapterUploader.uploadEnabled()) throw AdapterUploadDisabledException()
        val cacheDir = modelDir.parentFile
            ?: throw NotImplementedFeatureException("pushAdapter (no cache dir)")
        AdapterUploader.prepareCard(cacheDir, repoId) // builds metadata + gate + card; fails closed
        throw NotImplementedFeatureException("pushAdapter upload (device leg)")
    }

    override suspend fun federatedRound(
        config: FederatedConfig,
        globalRecord: ByteArray?,
        roundNumber: Int,
        localTraining: LocalRoundTraining,
        metrics: Map<String, Double>,
        train: Boolean,
    ): FederatedRoundResult {
        // Fail closed, and say which precondition is missing: consent, TLS, auth and the default-off
        // BuildConfig.FEDERATION_ENABLED flag are all checked here, BEFORE a native handle is touched
        // or a single tensor is read.
        config.requireRoundIsPermitted()

        val trainer = repo.ortTrainerNative
            ?: throw MissingArtifactException(
                "a federated round needs a live training session; this package has no train/ stage " +
                    "(installed features: ${capabilities.availableFeatures})",
            )
        val inferenceDir = inferencePackagePath
            ?: PackagePaths.forCache(modelDir.parentFile, modelDir.name).inference.absolutePath
        val handoffFile = File(inferenceDir, WeightHandoffMap.FILENAME)
        if (!handoffFile.isFile) {
            throw MissingArtifactException(
                "federated rounds are keyed by ${WeightHandoffMap.FILENAME}, which is absent from " +
                    "$inferenceDir — it is the authority on adapter tensor names and shapes for both " +
                    "the device and the gateway, so a round without it would upload tensors neither " +
                    "side can identify",
            )
        }
        val handoff = WeightHandoffMap.load(handoffFile)

        return FederatedTrainingRepository.forSession(
            config = config,
            handoff = handoff,
            trainer = trainer,
            localTraining = localTraining,
            baseModelId = modelDir.name,
            packageRevision = handoff.schemaVersion,
        ).runRound(
            globalRecord = globalRecord,
            roundNumber = roundNumber,
            metrics = metrics,
            train = train,
        )
    }

    override fun close() {
        repo.resetInference()
        repo.resetTraining()
        // A classification session is a second native ORT session over the same package; leaking it
        // would hold the graph's memory for the whole process, which on a phone is the difference
        // between unloading a model and appearing to.
        classifier?.close()
        classifier = null
    }
}

private fun TrainingProgress.toPublic(): TrainProgress =
    TrainProgress(
        currentStep = currentStep,
        currentEpoch = currentEpoch,
        totalLoss = totalLoss,
        epochLoss = epochLoss,
        stepLoss = stepLoss,
        learningRate = learningRate,
        stepDurationMs = stepDurationMs,
        epochDurationMs = epochDurationMs,
        totalDurationMs = totalDurationMs,
        isCompleted = isCompleted,
    )

private fun InferenceProgress.toPublic(): GenerateProgress =
    GenerateProgress(
        token = token,
        tokenId = tokenId,
        totalDecodedTokens = totalDecodedTokens,
        prefillTimeMs = prefillTimeMs,
        timeToLoadModelMs = timeToLoadModelMs,
        generationTimeMs = generationTimeMs,
        avgTokensPerSecond = avgTokensPerSecond,
        isCompleted = isCompleted,
        promptTokenCount = promptTokenCount,
        contextLimit = contextLimit,
    )

private fun RagResult.toPublic(): RetrievalResult =
    RetrievalResult(
        matches = documents?.map { RetrievalMatch(it.document.text, it.score) } ?: emptyList(),
        queryTimeMs = queryTimeMs,
    )

package com.martinkorelic.mobiletransformers.repository

import android.content.Context
import android.util.Log
import com.martinkorelic.mobiletransformers.InferenceProgress
import com.martinkorelic.mobiletransformers.MobileTransformersException
import com.martinkorelic.mobiletransformers.ORTGenerationConfig
import com.martinkorelic.mobiletransformers.ORTRagArguments
import com.martinkorelic.mobiletransformers.runtime.MemoryHeadroom
import com.martinkorelic.mobiletransformers.runtime.MemoryProbe
import com.martinkorelic.mobiletransformers.runtime.ModelRuntime
import com.martinkorelic.mobiletransformers.runtime.ModelRuntimeFactory
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.ORTRetriever
import com.martinkorelic.mobiletransformers.ORTTokenizerNative
import com.martinkorelic.mobiletransformers.ORTTrainerNative
import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.RagResult
import com.martinkorelic.mobiletransformers.TaskPreprocessor
import com.martinkorelic.mobiletransformers.TrainingProgress
import com.martinkorelic.mobiletransformers.parseGenerationArguments
import com.martinkorelic.mobiletransformers.parseRagArguments
import com.martinkorelic.mobiletransformers.parseTrainingArguments

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.PackagePaths

enum class LLMState {
    NotInitialized,
    ReadyTrain,
    Training,
    ReadyGenerate,
    Generating,
    Querying,
    SavingModel
}

interface GenerationCallback {
    fun onModelLoadStart() {}
    fun onModelLoadEnd() {}
    fun onStartGeneration(inferenceProgress: InferenceProgress) {}
    fun onPartialResult(inferenceProgress: InferenceProgress) {}
    fun onCompletion(inferenceProgress: InferenceProgress) {}
    fun onError(error: Throwable) {}
}

interface TrainingCallback {
    fun onModelLoadStart() {}
    fun onModelLoadEnd() {}
    fun onDataLoadStart() {}
    fun onDataLoadEnd(totalSteps: Int, stepsPerEpoch : Int) {}
    fun onSaveModelStart(trainingProgress: TrainingProgress) {}
    fun onSaveModelEnd(trainingProgress: TrainingProgress) {}
    fun onOptimizerStep(trainingProgress: TrainingProgress) {}
    fun onStepStart(trainingProgress: TrainingProgress) {}
    fun onStepEnd(trainingProgress: TrainingProgress) {}
    fun onEpochStart(trainingProgress: TrainingProgress) {}
    fun onEpochEnd(trainingProgress: TrainingProgress) {}
    fun onMergeStart(trainingProgress: TrainingProgress) {}
    fun onMergeEnd(trainingProgress: TrainingProgress) {}
    fun onCompletion(trainingProgress: TrainingProgress) {}
    fun onError(error: Throwable) {}
}

interface RagCallback {
    fun onModelLoadStart() {}
    fun onModelLoadEnd() {}
    fun onQueryStart() {}
    fun onQueryResults(queryResult: RagResult) {}
    fun onQueryEnd() {}
    fun onError(error: Throwable) {}
}

class LLMRepository(val applicationContext: Context, private val cacheDir : String, initialModel : String? = null) {

    private val LOG_TAG = "LLMRepository"

    // TODO: Should rename into something else as it doesn't refer to only just one model, but rather a set of different models for training/inference/embedding
    private var _modelName: String = ""

    var modelName: String
        get() = _modelName
        set(value) {
            if (value in availableModels) {
                _modelName = value
                updatePaths()
            } else {
                Log.w(LOG_TAG, "Model '$value' not found in available models: $availableModels. Keeping modelName as '$_modelName'.")
            }
        }

    /**
     * Returns all LLM models that are present on device
     */
    val availableModels: List<String>
        get() {
            val dir = File(cacheDir)
            return dir.listFiles { file -> file.isDirectory }?.map { it.name } ?: emptyList()
        }

    // Configuration paths
    private var tokenizerConfigPath : String = PackagePaths.forCache(cacheDir, _modelName).tokenizer.absolutePath
    private var trainingConfigPath = File(PackagePaths.forCache(cacheDir, _modelName).train, "training_config.json").absolutePath
    private var generationConfigPath = File(PackagePaths.forCache(cacheDir, _modelName).inference, "generation_config.json").absolutePath
    private var embeddingConfigPath = File(PackagePaths.forCache(cacheDir, _modelName).inference, "rag_config.json").absolutePath

    // Training, generation and RAG config
    private var _trainingConfig = ORTTrainingConfig()
    private var _generationConfig = ORTGenerationConfig()
    private var _ragConfig = ORTRagConfig()

    var trainingConfig: ORTTrainingConfig
        get() = _trainingConfig
        set(value) {
            _trainingConfig = value
        }

    var generationConfig: ORTGenerationConfig
        get() = _generationConfig
        set(value) {
            _generationConfig = value
        }

    var ragConfig: ORTRagConfig
        get() = _ragConfig
        set(value) {
            _ragConfig = value
            ortRetriever?.ragConfig = _ragConfig
        }

    // Availability
    var isTrainingAvailable : Boolean = false
    var isGenerationAvailable : Boolean = false
    var isRagAvailable : Boolean = false

    // Callback properties
    var generationCallback: GenerationCallback? = null
    var trainingCallback: TrainingCallback? = null
    var ragCallback : RagCallback? = null

    // Training capabilities
    var ortTrainerNative : ORTTrainerNative? = null

    // Tokenizer capabilities
    var ortTokenizerNative : ORTTokenizerNative? = null

    // Inference capabilities (#11): the selected engine (Native floor or GenAI) behind ModelRuntime.
    var modelRuntime : ModelRuntime? = null

    /**
     * Why the last [prepareGeneration] failed, so the failure survives to the caller.
     *
     * `prepareGeneration` runs inside a `launch` and cannot throw at its caller, so it used to catch,
     * log, and then set `llmState = ReadyGenerate` with [modelRuntime] still null. `runGenerationStream`
     * would log "Model has not been initialized" and return — a generate() that produces nothing and
     * reports no error. The real reason (a rejected genai_config, a missing artifact) only existed in
     * logcat.
     *
     * Retained here and re-raised when work is actually requested, so the cause reaches the caller.
     */
    @Volatile
    var lastGenerationSessionFailure : Throwable? = null
        private set

    /**
     * Why the last training-session setup failed, or null.
     *
     * The generation path has had this since #11; the training path had not, and the difference was
     * a crash. `prepareTraining` builds the trainer inside `coroutineScope.launch`, and a `launch`
     * that throws does **not** deliver the exception to whoever `join()`s it — it goes to the scope's
     * parent job, and this scope's parent is a bare `Job()` with no handler, i.e. the thread's
     * default handler, i.e. process death. A mistyped `DatasetConfig.task` therefore killed the app:
     *
     *     FATAL EXCEPTION: main
     *     java.lang.IllegalArgumentException: Unsupported task: none.
     *         at ORTTrainerNative.<init>(ORTTrainerNative.kt:42)
     *         at LLMRepository$prepareTraining$3$1$1.invokeSuspend(LLMRepository.kt:546)
     *
     * `Tasks.resolve` now rejects that particular input before the launch, but the shape of the
     * hazard is not specific to it — any failure opening the training session (a missing checkpoint,
     * an unreadable graph, an OOM) had the same fate. Captured here and re-raised by
     * [TrainingRepository.performTraining], it becomes an error the caller can show.
     */
    @Volatile
    var lastTrainingSessionFailure : Throwable? = null
        private set

    /** Take the recorded training-setup failure, clearing it. */
    fun consumeTrainingSessionFailure(): Throwable? {
        val failure = lastTrainingSessionFailure
        lastTrainingSessionFailure = null
        return failure
    }

    // Retriever capabilities
    var ortRetriever : ORTRetriever? = null

    // LLM state. @Volatile because the prepare*/run* paths mutate it from Dispatchers.Default while
    // callers observe it from other threads.
    @Volatile
    var llmState : LLMState = LLMState.NotInitialized

    /**
     * #18/#34 session lock: ONE native session at a time.
     *
     * `prepareTraining`/`prepareGeneration`/`prepareRetriever` each destroy and create native handles
     * and reassign [llmState]. Nothing serialized them, so two concurrent calls (a train kicked off
     * while a generate was still loading, say) could race on the same handles — a use-after-free in
     * native code, not a Kotlin exception. This lock was once recorded as done and
     * #34's scheduler is specified against it, but no lock existed anywhere in the library.
     *
     * Held only across session *setup/teardown*, never across a full training run or generation loop,
     * so a long job does not block a subsequent `release`.
     */
    private val sessionLock = Mutex()

    /** Run [block] with exclusive access to the native sessions. */
    suspend fun <T> withSessionLock(block: suspend () -> T): T = sessionLock.withLock { block() }

    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())

    init {
        llmState = LLMState.NotInitialized

        if (initialModel != null) {
            _modelName = initialModel
            updatePaths()
            Log.i(LOG_TAG, "Model set to '$_modelName'.")
        }

        if (_modelName.isEmpty()) {
            val firstAvailable = availableModels.firstOrNull()
            if (firstAvailable != null) {
                _modelName = firstAvailable
                updatePaths()
                Log.i(LOG_TAG, "Default model set to first available: $_modelName")
            }
        }
    }

    /**
     * The inference graph filename actually present in [inferenceDir].
     *
     * `model.onnx` is what the exporter's normalization step always writes (and what the manifest
     * records), so it is preferred; a single other `.onnx` is accepted for hand-assembled packages.
     * Falls back to `model.onnx` so the failure, if any, names a real file rather than `.onnx`.
     */
    private fun resolveInferenceGraphName(inferenceDir: String): String {
        val dir = File(inferenceDir)
        val canonical = File(dir, "model.onnx")
        if (canonical.isFile) return canonical.name
        val candidates = dir.listFiles { f: File -> f.isFile && f.name.endsWith(".onnx") }.orEmpty()
        return candidates.singleOrNull()?.name ?: "model.onnx"
    }

    private fun updatePaths() {
        tokenizerConfigPath = PackagePaths.forCache(cacheDir, _modelName).tokenizer.absolutePath
        trainingConfigPath = File(PackagePaths.forCache(cacheDir, _modelName).train, "training_config.json").absolutePath
        generationConfigPath = File(PackagePaths.forCache(cacheDir, _modelName).inference, "generation_config.json").absolutePath
        embeddingConfigPath = File(PackagePaths.forCache(cacheDir, _modelName).embedding, "rag_config.json").absolutePath

        // Check if training config exists before parsing
        if (File(trainingConfigPath).exists()) {
            // Installed directory name is authoritative, as for the generation/RAG configs: the
            // trainer resolves its dataset under `<cacheDir>/<repoName>/train/`.
            trainingConfig = parseTrainingArguments(trainingConfigPath).copy(repoName = _modelName)
            Log.d(LOG_TAG, "Training config loaded from: $trainingConfigPath")
            isTrainingAvailable = true
        } else {
            Log.w(LOG_TAG, "Training config not found at: $trainingConfigPath")
            isTrainingAvailable = false
        }

        // Check if generation config exists before parsing
        if (File(generationConfigPath).exists()) {
            // The installed package is authoritative for model *identity*. `generation_config.json` is
            // the model's own HF generation config and carries no `repoName`/`onnxName`, so the parser
            // fell back to "model" and ".onnx" — the session then tried to open
            // `<cacheDir>/model/inference/.onnx`, which cannot exist. Pin both to what is on disk.
            generationConfig = parseGenerationArguments(generationConfigPath).copy(
                repoName = _modelName,
                onnxName = resolveInferenceGraphName(PackagePaths.forCache(cacheDir, _modelName).inference.absolutePath),
            )
            Log.d(LOG_TAG, "Generation config loaded from: $generationConfigPath")
            isGenerationAvailable = true
        } else {
            Log.w(LOG_TAG, "Generation config not found at: $generationConfigPath")
            isGenerationAvailable = false
        }

        // Check if embedding config exists before parsing
        if (File(embeddingConfigPath).exists()) {
            // The installed directory name is authoritative for `repoName` — the retriever resolves
            // `<cacheDir>/<repoName>/embedding/`, so a config carrying a differently-sanitized id
            // (exporter vs installer) would send it to a path that does not exist.
            ragConfig = parseRagArguments(embeddingConfigPath).copy(repoName = _modelName)
            Log.d(LOG_TAG, "RAG config loaded from: $embeddingConfigPath")
            isRagAvailable = true
        } else {
            Log.w(LOG_TAG, "RAG config not found at: $embeddingConfigPath")
            isRagAvailable = false
        }
    }

    fun resetInference() {
        // Destroy previous tokenizer session
        ortTokenizerNative?.destroySession()
        // Destroy previous inference session
        modelRuntime?.release()

        ortTokenizerNative = null
        modelRuntime = null

        llmState = LLMState.NotInitialized
    }

    /**
     * Drop the inference session if one is open. Idempotent, and **not conditional on [llmState]**.
     *
     * ### Why this is a resource check and not a state check
     *
     * `prepareTraining` used to release only `if (llmState == LLMState.ReadyGenerate)` — a *state*
     * standing in for a *resource*. Whether a native session is open is knowable directly
     * (`modelRuntime != null`), and the state enum has seven values of which four can hold a live
     * inference session: `Generating` and `Querying` (work in flight), `NotInitialized` (set by
     * `prepareGeneration`'s own catch path, which leaves a partially-built runtime behind), and
     * `ReadyTrain`. Any of those and the training session was opened **on top of** the inference one.
     *
     * That is not a leak of a handle, it is a leak of a graph: this package's `inference/` stage is
     * 3.5 GB of fp32 (`model.onnx_data` 1.74 GB + `frozen_base.onnx.data` 1.68 GB). Holding it while
     * ORT builds the training graph is what took the app to 2.1 GB RSS + 1.1 GB swap on a 5.5 GB
     * device and got it SIGKILLed by `lmkd` mid-run, after the killer had already reclaimed five
     * other processes:
     *
     *     lmkd: Reclaim 'com.martinkorelic.mobiletransformers.app' (31489), oom_score_adj 0,
     *           to free 2175440kB rss, 1091904kB swap; reason: min2x watermark is breached
     *     Zygote: Process 31489 exited due to signal 9 (Killed)
     *
     * There is no Java exception for that and nothing to catch — the only defence is not holding both.
     *
     * The log line is deliberate. The previous code left no trace either way, so "was the inference
     * session still resident during training?" could not be answered from a logcat capture; it had to
     * be re-derived from the source. Now the answer is in the log next to the RSS it freed.
     */
    private fun releaseInferenceRuntime(reason: String) {
        val runtime = modelRuntime ?: return
        val before = MemoryProbe.currentRssKb()
        // #11: release whichever engine is loaded. The old `when (type)` released only Native, so a
        // GenAI session leaked its native handle across a train switch.
        runtime.release()
        modelRuntime = null
        val after = MemoryProbe.currentRssKb()
        Log.i(
            LOG_TAG,
            "Released the inference session ($reason): RSS ${before} kB -> ${after} kB",
        )
        if (llmState == LLMState.ReadyGenerate || llmState == LLMState.Generating || llmState == LLMState.Querying) {
            llmState = LLMState.NotInitialized
        }
    }

    /**
     * Drop the training session if one is open. The mirror of [releaseInferenceRuntime].
     *
     * `destroySession` is already idempotent (it no-ops on a zero handle), but the reference was left
     * dangling and nothing recorded that the swap had happened — the same blind spot in the other
     * direction.
     */
    private fun releaseTrainingSessionForSwap(reason: String, saveCheckpoint: Boolean = false) {
        val trainer = ortTrainerNative ?: return
        val before = MemoryProbe.currentRssKb()
        trainer.destroySession(saveCheckpoint)
        ortTrainerNative = null
        val after = MemoryProbe.currentRssKb()
        Log.i(
            LOG_TAG,
            "Released the training session ($reason, saveCheckpoint=$saveCheckpoint): " +
                "RSS ${before} kB -> ${after} kB",
        )
        if (llmState == LLMState.ReadyTrain || llmState == LLMState.Training) {
            llmState = LLMState.NotInitialized
        }
    }

    fun resetTraining() {
        ortTokenizerNative?.destroySession()
        ortTrainerNative?.destroySession(false)
        ortTokenizerNative = null
        ortTrainerNative = null

        llmState = LLMState.NotInitialized
    }

    suspend private fun makeOrtTrainer(trainingArguments: ORTTrainingConfig? = null, dataPreprocessFunction: TaskPreprocessor? = null) : ORTTrainerNative {
        if (ortTokenizerNative == null) {
            Log.d(LOG_TAG, "Could not find the tokenizer. Initializing tokenizer...")
            ortTokenizerNative = ORTTokenizerNative(tokenizerConfigPath)
            ortTokenizerNative?.createTokenizerModel()
        }

        val trainArgs = trainingConfig.overrideConfig(trainingArguments)

        val finalConfig = if (dataPreprocessFunction != null)
            trainArgs.copy(customPreprocess = dataPreprocessFunction)
        else
            trainArgs

        return ORTTrainerNative(
            applicationContext,
            cacheDir,
            ortTokenizerNative!!,
            finalConfig
        )
    }

    // #11: select + load the inference engine (Native floor, or GenAI when requested & available) over the
    // one shared inference/ package via ModelRuntimeFactory, with transparent fallback to Native.
    private suspend fun makeModelRuntime(generationArgs : ORTGenerationConfig) : ModelRuntime {
        if (ortTokenizerNative == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer. Initializing tokenizer...")
            ortTokenizerNative = ORTTokenizerNative(tokenizerConfigPath)
            ortTokenizerNative?.createTokenizerModel()
        }

        // Drop the training session before opening a generation one. Symmetric with
        // prepareTraining's release, and for the same memory reason — the two graphs must never be
        // resident together. `saveCheckpoint = false` is unchanged: the training path is responsible
        // for persisting its own checkpoint before handing over.
        releaseTrainingSessionForSwap("a generation session is being opened")

        // #13: supportedEngines comes from the installed variant's manifest declaration. A package that
        // declares none (an older export, or a manifest-less legacy dir) keeps the permissive default —
        // narrowing an unknown declaration would break packages that work today.
        val declaredEngines = installedSupportedEngines()
        return if (declaredEngines != null) {
            ModelRuntimeFactory.create(cacheDir, ortTokenizerNative!!, generationArgs, declaredEngines)
        } else {
            ModelRuntimeFactory.create(cacheDir, ortTokenizerNative!!, generationArgs)
        }
    }

    /**
     * What the last [prepareTraining] concluded about memory headroom, or null when it was fine.
     *
     * Surfaced so an app can warn the user; it never blocks the run.
     */
    @Volatile
    var lastTrainingHeadroomWarning : String? = null
        private set

    /** The full parameter count the installed package's training graph materialises, or 0. */
    private fun installedTrainingParameterCount(): Long {
        val manifestFile = File(
            PackagePaths.forCache(cacheDir, _modelName).root,
            PackageFormat.MANIFEST_FILENAME,
        )
        if (!manifestFile.isFile) return 0L
        return runCatching { MobileTransformersManifest.load(manifestFile).trainingParameterCount }
            .getOrDefault(0L)
    }

    /** The installed package's declared engines, or null when it declares none (see [makeModelRuntime]). */
    private fun installedSupportedEngines(): Set<String>? {
        val manifestFile = File(
            PackagePaths.forCache(cacheDir, _modelName).root,
            PackageFormat.MANIFEST_FILENAME,
        )
        if (!manifestFile.isFile) return null
        return runCatching { MobileTransformersManifest.load(manifestFile).supportedEnginesFor() }
            .onFailure { Log.w(LOG_TAG, "unreadable manifest at ${manifestFile.path}: ${it.message}") }
            .getOrNull()
    }

    private suspend fun makeOrtRag(ortArgs : ORTRagConfig) : ORTRetriever {

        // Same swap rule as the generation path: retrieval opens an embedding session, and the
        // training graph must not still be resident behind it.
        releaseTrainingSessionForSwap("a retrieval session is being opened")

        // #27: honor the override config actually passed in (was previously ignoring ortArgs).
        val retriever = ORTRetriever(cacheDir, applicationContext, ortArgs)
        retriever.createEmbeddingModel()

        return retriever
    }

    /* Inference methods */

    suspend fun prepareRetriever(ragArgs : ORTRagConfig? = null): Job {
        // Clean up the tokenizer and destroy session if there was previous training
        // Takes less memory if we initialize the training session again with the checkpoint state

        if (llmState == LLMState.ReadyTrain) {
            ortTokenizerNative = null
        }

        // #27: apply the caller's RAG config override (falls back to the loaded field config).
        val finalRagConfig = ragArgs ?: ragConfig
        ragConfig = finalRagConfig

        // If the model was in training state
        if (llmState == LLMState.Training) {

            coroutineScope.launch {
                withContext(Dispatchers.Default) {

                    // Release training session if there was any (no saving)
                    releaseTrainingSessionForSwap("switching to generation")

                    llmState = LLMState.ReadyGenerate
                }
            }.join()
        }

        return coroutineScope.launch {
            // #18/#34 session lock: serialize native session creation/teardown (see `sessionLock`).
            sessionLock.withLock {
                try {
                    withContext(Dispatchers.Default) {
                        ortRetriever = makeOrtRag(finalRagConfig)
                    }
                } catch (e: Exception) {
                    Log.e(LOG_TAG, "Retriever session failed to create: ${e.message}")
                }
            }
        }
    }

    suspend fun prepareGeneration(generationArgs : ORTGenerationConfig? = null): Job {
        // Clean up the tokenizer and destroy session if there was previous training
        // Takes less memory if we initialize the training session again with the checkpoint state

        if (llmState == LLMState.ReadyTrain) {
            ortTokenizerNative = null
        }

        val finalGenConfig = generationConfig.overrideConfig(generationArgs)

        // If the model was in training state
        if (llmState == LLMState.Training) {

            coroutineScope.launch {
                withContext(Dispatchers.Default) {

                    // Release training session if there was any (no saving)
                    releaseTrainingSessionForSwap("switching to generation")

                    llmState = LLMState.ReadyGenerate
                }
            }.join()
        }

        return coroutineScope.launch {
            // #18/#34 session lock: serialize native session creation/teardown (see `sessionLock`).
            sessionLock.withLock {
                try {
                    withContext(Dispatchers.Default) {
                        // #11: engine selection belongs to ModelRuntimeFactory (which owns the GenAI
                        // availability probe and the transparent fallback to Native), not to a string
                        // `when` here. The old `when (type) { "native" -> …; else -> Log.e }` dropped
                        // every GenAI config on the floor, leaving the runtime null and generate() hanging.
                        modelRuntime = makeModelRuntime(finalGenConfig)
                        lastGenerationSessionFailure = null
                    }
                } catch (e: Exception) {
                    // Keep the cause: this coroutine cannot throw at prepareGeneration's caller, and
                    // a log line is not an error report. runGenerationStream re-raises it.
                    lastGenerationSessionFailure = e
                    Log.e(LOG_TAG, "Generation session failed to create: ${e.message}", e)
                } finally {
                    llmState = LLMState.ReadyGenerate
                }
            }
        }
    }

    suspend fun runGenerationStream(prompt: String, generationArgs: ORTGenerationConfig? = null) {
        if (modelRuntime == null) {
            // Surface why. Returning quietly here is what let a rejected genai_config.json read as
            // "generation produced nothing" instead of "the engine you asked for never loaded".
            lastGenerationSessionFailure?.let { cause ->
                throw MobileTransformersException(
                    "Generation session was never created: ${cause.message}",
                    cause,
                )
            }
            throw MobileTransformersException(
                "Model has not been initialized: call prepareGeneration() before runGenerationStream().",
            )
        }

        val finalGenConfig = generationConfig.overrideConfig(generationArgs)

        llmState = LLMState.Generating

        coroutineScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    // #11: the loaded ModelRuntime already *is* the selected engine — dispatching on
                    // `type` again here would re-open the GenAI hole closed in prepareGeneration.
                    modelRuntime!!.generate(prompt, finalGenConfig, generationCallback)
                }
            } catch (e : Exception) {
                Log.e(LOG_TAG, "Generation failed: ${e.message}")
            } finally {
                llmState = LLMState.ReadyGenerate
            }
        }
    }

    suspend fun runRetriever(prompt: String, ragArgs: ORTRagArguments? = null): Job {

        val finalRagConfig = ragConfig.overwriteWith(ragArgs)

        llmState = LLMState.Querying

        return coroutineScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    ortRetriever?.query(prompt, finalRagConfig, ragCallback)
                }
            } catch (e : Exception) {
                Log.e(LOG_TAG, "Query failed: ${e.message}")
            } finally {
                llmState = LLMState.ReadyGenerate
            }
        }
    }

    /* Training methods */

    suspend fun prepareTraining(trainingArguments: ORTTrainingConfig? = null,  dataPreprocessFunction: TaskPreprocessor? = null) : Job {

        // #18/#34 session lock: the inference teardown must not interleave with another prepare*
        // creating a session on the same handles.
        coroutineScope.launch {
            sessionLock.withLock {
                withContext(Dispatchers.Default) {
                    releaseInferenceRuntime("a training session is being opened")
                }
            }
        }.join()

        val finalTrainConfig = trainingConfig.overrideConfig(trainingArguments);

        // Advisory only — see MemoryHeadroom for why this warns instead of refusing. Logged before
        // the session opens because if the estimate is right there will be no `after`: the process
        // is SIGKILLed and this line is the last thing in the capture that explains why.
        lastTrainingHeadroomWarning = when (
            val verdict = MemoryHeadroom.verdict(
                trainingParameterCount = installedTrainingParameterCount(),
                availableKb = MemoryHeadroom.availableKb(),
            )
        ) {
            is MemoryHeadroom.Verdict.Tight -> {
                Log.w(LOG_TAG, "Memory headroom: ${verdict.message}")
                verdict.message
            }
            else -> null
        }
        Log.i(LOG_TAG, "Opening a training session at RSS ${MemoryProbe.currentRssKb()} kB")

        lastTrainingSessionFailure = null

        return coroutineScope.launch {
            // #18/#34 session lock: serialize native session creation/teardown (see `sessionLock`).
            sessionLock.withLock {
                try {
                    withContext(Dispatchers.Default) {
                        ortTrainerNative = makeOrtTrainer(
                            finalTrainConfig,
                            dataPreprocessFunction
                        )
                        llmState = LLMState.ReadyTrain
                    }
                } catch (e: Throwable) {
                    // Must not escape: this coroutine's parent has no handler, so an escaping throw
                    // is a FATAL EXCEPTION rather than a failed call. See lastTrainingSessionFailure.
                    lastTrainingSessionFailure = e
                    llmState = LLMState.NotInitialized
                    Log.e(LOG_TAG, "Training session failed to create: ${e.message}", e)
                }
            }
        }
    }

    suspend fun runTraining() : Job? {
        if (llmState != LLMState.ReadyTrain && llmState != LLMState.Training) {
            Log.e(LOG_TAG, "Model is not ready to train.")
            return null
        }

        llmState = LLMState.Training

        // Here we mark that there was training done on this model
        return coroutineScope.launch {
            withContext(Dispatchers.IO) {
                ortTrainerNative?.startTraining(trainingCallback)
            }
            llmState = LLMState.ReadyTrain
        }
    }

    suspend fun saveTraining(saveModel : Boolean) : Job? {
        if (llmState != LLMState.ReadyTrain && llmState != LLMState.Training) {
            Log.e(LOG_TAG, "Model is not ready to save.")
            return null
        }

        llmState = LLMState.SavingModel

        return coroutineScope.launch {
            withContext(Dispatchers.IO) {
                ortTrainerNative?.destroySession(saveModel)
            }
            llmState = LLMState.NotInitialized
        }
    }
}
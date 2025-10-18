package com.martinkorelic.ortmobile.repository

import android.content.Context
import android.util.Log
import com.martinkorelic.ortmobile.InferenceProgress
import com.martinkorelic.ortmobile.ORTGenAINative
import com.martinkorelic.ortmobile.ORTGenAITokenizer
import com.martinkorelic.ortmobile.ORTGenerationConfig
import com.martinkorelic.ortmobile.ORTGeneratorNative
import com.martinkorelic.ortmobile.ORTRagArguments
import com.martinkorelic.ortmobile.ORTRagConfig
import com.martinkorelic.ortmobile.ORTRetriever
import com.martinkorelic.ortmobile.ORTTokenizerNative
import com.martinkorelic.ortmobile.ORTTrainerNative
import com.martinkorelic.ortmobile.ORTTrainingConfig
import com.martinkorelic.ortmobile.RagResult
import com.martinkorelic.ortmobile.TaskPreprocessor
import com.martinkorelic.ortmobile.TrainingProgress
import com.martinkorelic.ortmobile.parseGenerationArguments
import com.martinkorelic.ortmobile.parseRagArguments
import com.martinkorelic.ortmobile.parseTrainingArguments

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

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
    private var tokenizerConfigPath : String = "$cacheDir/$_modelName/tokenizer"
    private var trainingConfigPath = "$cacheDir/$_modelName/train/training_config.json"
    private var generationConfigPath = "$cacheDir/$_modelName/inference/generation_config.json"
    private var embeddingConfigPath = "$cacheDir/$_modelName/inference/rag_config.json"

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
            ortNativeInference?.generationConfig = _generationConfig
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
    private var ortGenAITokenizer : ORTGenAITokenizer? = null
    var ortTokenizerNative : ORTTokenizerNative? = null

    // Inference capabilities
    private var ortGenAiNative : ORTGenAINative? = null
    var ortNativeInference : ORTGeneratorNative? = null

    // Retriever capabilities
    var ortRetriever : ORTRetriever? = null

    // LLM state
    var llmState : LLMState = LLMState.NotInitialized

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

    private fun updatePaths() {
        tokenizerConfigPath = "$cacheDir/$_modelName/tokenizer"
        trainingConfigPath = "$cacheDir/$_modelName/train/training_config.json"
        generationConfigPath = "$cacheDir/$_modelName/inference/generation_config.json"
        embeddingConfigPath = "$cacheDir/$_modelName/embedding/rag_config.json"

        // Check if training config exists before parsing
        if (File(trainingConfigPath).exists()) {
            trainingConfig = parseTrainingArguments(trainingConfigPath)
            Log.d(LOG_TAG, "Training config loaded from: $trainingConfigPath")
            isTrainingAvailable = true
        } else {
            Log.w(LOG_TAG, "Training config not found at: $trainingConfigPath")
            isTrainingAvailable = false
        }

        // Check if generation config exists before parsing
        if (File(generationConfigPath).exists()) {
            generationConfig = parseGenerationArguments(generationConfigPath)
            Log.d(LOG_TAG, "Generation config loaded from: $generationConfigPath")
            isGenerationAvailable = true
        } else {
            Log.w(LOG_TAG, "Generation config not found at: $generationConfigPath")
            isGenerationAvailable = false
        }

        // Check if embedding config exists before parsing
        if (File(embeddingConfigPath).exists()) {
            ragConfig = parseRagArguments(embeddingConfigPath)
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
        ortNativeInference?.destroySession()

        ortTokenizerNative = null
        ortNativeInference = null

        llmState = LLMState.NotInitialized
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

    private suspend fun makeOrtNativeInference(generationArgs : ORTGenerationConfig) : ORTGeneratorNative {
        //if (ortTrainerNative == null) {
        //    Log.e(LOG_TAG, "Could not find the train model. Make sure it is initialized before GenAI inference.")
        //    return null
        //}

        if (ortTokenizerNative == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer. Initializing tokenizer...")
            ortTokenizerNative = ORTTokenizerNative(tokenizerConfigPath)
            ortTokenizerNative?.createTokenizerModel()
        }

        // We destroy trainer session before loading generation session, if it was active before
        // Assuming the training session has been saved prior to this
        ortTrainerNative?.destroySession(false)

        val nativeInference = ORTGeneratorNative(cacheDir, ortTokenizerNative!!, generationConfig)
        nativeInference.createInferenceModel()

        return nativeInference
    }

    private suspend fun makeOrtRag(ortArgs : ORTRagConfig) : ORTRetriever {

        // We destroy trainer session before loading generation session, if it was active before
        // Assuming the training session has been saved prior to this
        ortTrainerNative?.destroySession(false)

        val retriever = ORTRetriever(cacheDir, applicationContext, ragConfig)
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

        // TODO: Override RAG config if needed
        //val finalGenConfig = generationConfig.overrideConfig(generationArgs)

        // If the model was in training state
        if (llmState == LLMState.Training) {

            coroutineScope.launch {
                withContext(Dispatchers.Default) {

                    // Release training session if there was any (no saving)
                    ortTrainerNative?.destroySession(false)

                    llmState = LLMState.ReadyGenerate
                }
            }.join()
        }

        return coroutineScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    ortRetriever = makeOrtRag(ragConfig)
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Retriever session failed to create: ${e.message}")
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
                    ortTrainerNative?.destroySession(false)

                    llmState = LLMState.ReadyGenerate
                }
            }.join()
        }

        return coroutineScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    when (finalGenConfig.type) {
                        // Deprecated
                        //"gen_ai" -> {
                        //    ortGenAiNative = makeOrtGenAI()
                        //}
                        "native" -> {
                            ortNativeInference = makeOrtNativeInference(finalGenConfig)
                        }
                        else -> {
                            Log.e(LOG_TAG, "Unknown generation type - ${finalGenConfig.type}")
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Generation session failed to create: ${e.message}")
            } finally {
                llmState = LLMState.ReadyGenerate
            }
        }
    }

    suspend fun runGenerationStream(prompt: String, generationArgs: ORTGenerationConfig? = null) {
        if (ortNativeInference == null) {
            Log.e(LOG_TAG, "Model has not been initialized and cached yet.")
            return
        }

        val finalGenConfig = generationConfig.overrideConfig(generationArgs)

        llmState = LLMState.Generating

        coroutineScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    when (finalGenConfig.type) {
                        // Deprecated
                        //"genai" -> ortGenAiNative!!.generateStream()
                        "native" -> {
                            ortNativeInference!!.generate(prompt, finalGenConfig, generationCallback)
                        }

                        else -> {
                            Log.e(LOG_TAG, "Unknown generation type - ${finalGenConfig.type}")
                        }
                    }
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

        if (llmState == LLMState.ReadyGenerate) {
            coroutineScope.launch {
                withContext(Dispatchers.Default) {

                    when (generationConfig.type) {
                        // Deprecated
                        //"genai" -> ortGenAiNative?.destroySession()
                        "native" -> ortNativeInference?.destroySession()
                    }

                    llmState = LLMState.NotInitialized
                }
            }.join()

        }

        val finalTrainConfig = trainingConfig.overrideConfig(trainingArguments);

        return coroutineScope.launch {
            withContext(Dispatchers.Default) {
                ortTrainerNative = makeOrtTrainer(
                    finalTrainConfig,
                    dataPreprocessFunction
                )
                llmState = LLMState.ReadyTrain
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
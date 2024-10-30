package com.example.orttransformer.repository

import android.util.Log
import com.example.orttransformer.ORTGenAINative
import com.example.orttransformer.ORTGeneratorNative
import com.example.orttransformer.ORTTokenizer
import com.example.orttransformer.ORTTrainerNative
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class LLMState {
    NotInitialized,
    ReadyTrain,
    Training,
    ReadyGenerate,
    Generating,
    SavingModel
}

class LLMRepository(modelArtifactPath : String, private val cacheDir : String) {

    private val LOG_TAG = "LLMRepository"

    private var generationConfig : MutableMap<String, String> = mutableMapOf(
        "type" to "native",
        "sampling" to "greedy",
        "max_sequence_length" to "100"
    )

    // Configuration paths
    private var artifactTrainDir : String = "/data/local/tmp/tinyllama_int16/train"
    private var genAiConfigPath : String = "/data/local/tmp/tinyllama_int16/inference"
    private var artifactInferenceModelPath : String = "/data/local/tmp/tinyllama_int16/inference/"
    private var artifactInferenceModelName : String = "genai_inference.onnx"
    private var tokenizerConfigPath : String = "/data/local/tmp/genaitest"

    // Training capabilities
    private var ortTrainerNative : ORTTrainerNative? = null
    private var ortTokenizer : ORTTokenizer? = null

    // Inference capabilities
    private var ortGenAiNative : ORTGenAINative? = null
    private var ortNativeInference : ORTGeneratorNative? = null

    // LLM state
    var llmState : LLMState = LLMState.NotInitialized

    private val _tokenFlow = MutableSharedFlow<String>(replay = 1)
    val tokenFlow: SharedFlow<String> = _tokenFlow

    private val _lossFlow = MutableSharedFlow<Float>(replay = 1)
    val lossFlow: SharedFlow<Float> = _lossFlow

    //private val executorService = Executors.newSingleThreadExecutor()
    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())

    init {
        // In any case, first initialize the training session
        // TODO: do not initialize training session beforehand, but only when needed
        ortTokenizer = ORTTokenizer(tokenizerConfigPath)

        ortTrainerNative = makeOrtTrainer()
        llmState = LLMState.ReadyTrain
    }

    private fun makeOrtTrainer() : ORTTrainerNative {

        if (ortTokenizer == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer. Initializing tokenizer...")
            ortTokenizer = ORTTokenizer(tokenizerConfigPath)
        }

        return ORTTrainerNative(artifactTrainDir, ortTokenizer!!, cacheDir)
    }

    private fun makeOrtGenAI() : ORTGenAINative? {
        if (ortTrainerNative == null) {
            Log.e(LOG_TAG, "Could not find the train model. Make sure it is initialized before GenAI inference.")
            return null
        }

        val genAiNative = ORTGenAINative(artifactTrainDir, genAiConfigPath)
        genAiNative.createGenAISessionFromTraining(ortTrainerNative!!.model)

        return genAiNative
    }

    private fun makeOrtNativeInference() : ORTGeneratorNative? {
        if (ortTrainerNative == null) {
            Log.e(LOG_TAG, "Could not find the train model. Make sure it is initialized before GenAI inference.")
            return null
        }

        if (ortTokenizer == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer. Initializing tokenizer...")
            ortTokenizer = ORTTokenizer(tokenizerConfigPath)
        }

        // Either we destroy or move session weights, depending on if the model was trained beforehand or not
        // TODO: for now we destroy training session
        ortTrainerNative?.destroySession(false)

        val nativeInference = ORTGeneratorNative(ortTokenizer!!)
        nativeInference.createInferenceModel(artifactInferenceModelPath, artifactInferenceModelName)
        return nativeInference
    }

    /**
     * Loads the native inference generation configuration from file.
     */
    fun loadGenerationConfig() {
        // TODO
    }

    /* Inference methods */

    suspend fun prepareGeneration(inferenceConfig : Map<String, String>?): Job {
        // Clean up the tokenizer and destroy session if there was previous training
        // Takes less memory if we initialize the training session again with the checkpoint state
        if (llmState == LLMState.ReadyTrain) {
            ortTokenizer = null
        }

        // If the model was in training state
        if (llmState == LLMState.Training) {

            // Overwrite generation configuration
            if (inferenceConfig != null) {
                for ((key, value) in inferenceConfig) {
                    generationConfig[key] = value
                }
            }

            coroutineScope.launch {
                withContext(Dispatchers.Default) {

                    // Either we destroy or move session weights, depending on if the model was trained beforehand or not
                    // TODO: for now we destroy training session with no saving of weights
                    ortTrainerNative?.destroySession(false)
                    //ortTrainerNative = makeOrtTrainer()

                    when (generationConfig["type"]) {
                        "gen_ai" -> {
                            ortGenAiNative = makeOrtGenAI()
                            ortTokenizer = null
                        }
                        "native" -> {
                            // TODO: Create native inference model from weight transfer
                            //ortNativeInference = makeOrtNativeInference()
                        }
                    }

                    llmState = LLMState.ReadyGenerate
                }
            }.join()

        }

        return coroutineScope.launch {
            withContext(Dispatchers.Default) {
                when (generationConfig["type"]) {
                    "gen_ai" -> {
                        ortGenAiNative = makeOrtGenAI()
                    }
                    "native" -> {
                        ortNativeInference = makeOrtNativeInference()
                    }
                }

                llmState = LLMState.ReadyGenerate
            }
        }
    }

    suspend fun runGenerationStream(prompt: String) {
        if (llmState != LLMState.ReadyGenerate) {
            Log.e(LOG_TAG, "Model has not been initialized and cached yet.")
            return
        }

        when (generationConfig["type"]) {
            "genai" -> {
                ortGenAiNative?.initializeGenerateStream(prompt)
            }
            "native" -> {
                Log.d(LOG_TAG, "TODO: Initialize and prefill stage...")
            }
        }

        coroutineScope.launch {

            withContext(Dispatchers.IO) {
                when (generationConfig["type"]) {
                    "genai" -> ortGenAiNative!!.generateStream()
                    "native" -> {
                        ortNativeInference!!.generate(prompt, generationConfig, _tokenFlow)
                    }
                }
            }
        }
    }

    fun runFullGenAi(prompt : String) : String {
        if (llmState != LLMState.ReadyGenerate) {
            Log.e(LOG_TAG, "Model has not been initialized and cached.")
            return "[ERROR]"
        }

        return ortGenAiNative?.generate(prompt) ?: "[NULL]"
    }

    /* Training methods */

    suspend fun prepareTraining() : Job {

        if (llmState == LLMState.ReadyGenerate) {
            coroutineScope.launch {
                withContext(Dispatchers.Default) {
                    ortGenAiNative?.destroySession()
                    llmState = LLMState.NotInitialized
                }
            }.join()

        }

        return coroutineScope.launch {
            withContext(Dispatchers.Default) {
                ortTrainerNative = makeOrtTrainer()
                llmState = LLMState.ReadyTrain
            }
        }
    }

    suspend fun runTraining(trainData: List<String>) : Job? {
        if (llmState != LLMState.ReadyTrain && llmState != LLMState.Training) {
            Log.e(LOG_TAG, "Model is not ready to train.")
            return null
        }
        var loss = -1.0F

        llmState = LLMState.Training

        // Here we mark that there was training done on this model
        return coroutineScope.launch {
            loss = withContext(Dispatchers.IO) {
                ortTrainerNative?.performTrainStep(trainData) ?: -1.0F
            }
            _lossFlow.emit(loss)
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
        }
    }


}
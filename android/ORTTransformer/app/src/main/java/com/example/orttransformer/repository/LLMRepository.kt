package com.example.orttransformer.repository

import android.util.Log
import com.example.orttransformer.ORTGenAINative
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
    Generating
}

class LLMRepository(modelArtifactPath : String, private val cacheDir : String) {

    private val LOG_TAG = "LLMRepository"

    // Configuration paths
    private var artifactTrainDir : String = "/data/local/tmp/tinyllama_int16/train"
    private var genAiConfigPath : String = "/data/local/tmp/tinyllama_int16/inference"
    private var tokenizerConfigPath : String = "/data/local/tmp/genaitest"

    // Training capabilities
    private var ortTrainerNative : ORTTrainerNative? = null
    private var ortTokenizer : ORTTokenizer? = null

    // Inference capabilities
    private var ortGenAiNative : ORTGenAINative? = null

    // LLM state
    private var llmState : LLMState = LLMState.NotInitialized

    private val _tokenFlow = MutableSharedFlow<String>(replay = 1)
    val tokenFlow: SharedFlow<String> = _tokenFlow

    private val _lossFlow = MutableSharedFlow<Float>(replay = 1)
    val lossFlow: SharedFlow<Float> = _lossFlow

    //private val executorService = Executors.newSingleThreadExecutor()
    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())

    init {
        // In any case, first initialize the training session
        ortTokenizer = ORTTokenizer(tokenizerConfigPath)

        ortTrainerNative = makeOrtTrainer()
        llmState = LLMState.ReadyTrain
    }

    private fun makeOrtTrainer() : ORTTrainerNative? {

        if (ortTokenizer == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer.")
            return null
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

    /* Inference methods */

    fun prepareGeneration() {
        if (llmState == LLMState.ReadyGenerate) {
            return
        }

        // Clean up the tokenizer and destroy session if there was previous training
        // Takes less memory if we initialize the training session again with the checkpoint state
        if (llmState == LLMState.ReadyTrain) {
            ortTokenizer = null
        }
        if (llmState == LLMState.Training) {
            ortTrainerNative?.destroySession()
            ortTrainerNative = makeOrtTrainer()
            llmState = LLMState.ReadyTrain
            ortTokenizer = null
        }

        ortGenAiNative = makeOrtGenAI()
        llmState = LLMState.ReadyGenerate
    }

    fun runInferenceStream(prompt: String) {
        if (llmState != LLMState.ReadyGenerate) {
            Log.e(LOG_TAG, "Model has not been initialized and cached.")
            return
        }

        ortGenAiNative?.initializeGenerateStream(prompt)

        coroutineScope.launch {
            var newToken = ""

            ortGenAiNative

            while (true) {

                // Thread pool?
                newToken = withContext(Dispatchers.IO) {
                    ortGenAiNative!!.generateStream()
                }

                _tokenFlow.emit(newToken)

                // Check if the stream has ended
                if (newToken == "[STOP]") {
                    break
                }
            }
        }


    }

    fun runFullInference(prompt : String) : String {
        if (llmState != LLMState.ReadyGenerate) {
            Log.e(LOG_TAG, "Model has not been initialized and cached.")
            return "[ERROR]"
        }

        return ortGenAiNative?.generate(prompt) ?: "[NULL]"
    }

    /* Training methods */

    fun prepareTraining() {
        if (llmState == LLMState.Training) {
            return
        }

        if (llmState != LLMState.ReadyTrain) {
            Log.d(LOG_TAG, "Model has not been initialized for training. Initializing training model...")
            ortTrainerNative = makeOrtTrainer()
            llmState = LLMState.ReadyTrain
        }

        if (llmState == LLMState.ReadyGenerate) {
            ortGenAiNative?.destroySession()
            llmState = LLMState.NotInitialized
        }
    }

    fun runTraining(trainData: List<String>) : Float {
        if (llmState != LLMState.ReadyTrain && llmState != LLMState.Training) {
            Log.e(LOG_TAG, "Model is not ready to train.")
            return 0F;
        }
        var loss = -1.0F

        coroutineScope.launch {
            loss = withContext(Dispatchers.IO) {
                ortTrainerNative?.performTrainStep(trainData) ?: -1.0F
            }
            _lossFlow.emit(loss)
        }

        // Here we mark that there was training done on this model
        llmState = LLMState.Training
        return loss
    }


}
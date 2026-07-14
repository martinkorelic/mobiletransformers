package com.martinkorelic.mobiletransformers

import android.util.Log

/**
 * ONNX Runtime GenAI class
 */
@Deprecated("Unfinished and abandoned class, use ORTGeneratorNative")
class ORTGenAINative(artifactDir : String, val genAIPath: String) {

    private val LOG_TAG = "ORTGenAINative"

    var genAIModel : Long = 0
    var weightCache : Long = 0

    var trainConfigPath : String = "${artifactDir}/training_config.json"

    fun createGenAISessionFromTraining(trainModel: Long) {
        Log.d(LOG_TAG, trainConfigPath)
        val requiresGrad = loadTrainableLayerNamesJSON(trainConfigPath)

        if (requiresGrad == null) {
            Log.e(LOG_TAG, "No training config provided. GenAI model cannot be initialized.")
        } else {
            Log.d(LOG_TAG, "Loading the training model...")
            weightCache = cacheSessionWeights(trainModel, requiresGrad)
            Log.d(LOG_TAG, "Cached trainable weights, now loading GenAI model...")
            genAIModel = createGenAISession(weightCache, genAIPath)
            Log.d(LOG_TAG, "Successfully created GenAI model")
        }

    }

    fun initializeGenerateStream(prompt: String) {
        initializeGenAIInference(genAIModel, prompt)
    }

    fun generateStream() : String {
        return performGenAIInferenceStep(genAIModel)
    }

    fun generate(prompt: String) : String {

        if (genAIModel == 0L) {
            Log.d(LOG_TAG, "GenAI model is not initialized yet. Have you initialized the model?")
        }
        Log.d(LOG_TAG, "Starting a new inference...")
        // Initialize the GenAI inference session with the prompt
        initializeGenAIInference(genAIModel, prompt)

        var newToken = ""
        var fullResults = ""

        // Decode the tokens as you go
        while (newToken != "[STOP]") {
            newToken = performGenAIInferenceStep(genAIModel)
            if (newToken != "[STOP]") {
                fullResults += newToken
            }
            Log.d(LOG_TAG, fullResults)
        }

        return fullResults
    }

    fun destroySession() {
        releaseGenAISession(genAIModel)
        releaseWeightSession(weightCache)
    }

    // NOTE: This should have been external functions, but were deprecated due to incompatibility with ONNX GenAI framework

    fun releaseWeightSession(weightCache: Long) {
        throw NotImplementedError("releaseWeightSession not yet implemented")
    }

    fun releaseGenAISession(genModel: Long) {
        throw NotImplementedError("releaseGenAISession not yet implemented")
    }

    fun initializeGenAIInference(genModel: Long, prompt: String) {
        throw NotImplementedError("initializeGenAIInference not yet implemented")
    }

    fun performGenAIInferenceStep(genModel: Long): String {
        throw NotImplementedError("performGenAIInferenceStep not yet implemented")
    }

    fun cacheSessionWeights(trainModel: Long, requiresGrad: Array<String>): Long {
        throw NotImplementedError("cacheSessionWeights not yet implemented")
    }

    fun createGenAISession(weightCache: Long, genAIPath: String): Long {
        throw NotImplementedError("createGenAISession not yet implemented")
    }
}
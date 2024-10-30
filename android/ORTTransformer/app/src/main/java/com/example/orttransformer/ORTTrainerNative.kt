package com.example.orttransformer

import android.util.Log

class ORTTrainerNative(artifactDir : String, private var tokenizer: ORTTokenizer, cacheDirPath: String) {

    private val LOG_TAG = "ORTTrainerNative"

    // Pointer to the model
    var model : Long = 0;

    val trainingModelPath = "$artifactDir/training_model.onnx"
    val evalModelPath = "$artifactDir/eval_model.onnx"
    val checkpointPath = "$artifactDir/checkpoint"
    val optimizerPath = "$artifactDir/optimizer_model.onnx"
    val trainingConfigPath = "$artifactDir/training_config.json"

    init {
        val requiresGrad = loadTrainableLayerNamesJSON("${artifactDir}/training_config.json")

        if (requiresGrad == null) {
            Log.e(LOG_TAG, "No training config provided. Model cannot be initialized.")
        } else {
            Log.d(LOG_TAG, "Loading the training model...")
            model = createTrainingSession(checkpointPath, trainingModelPath, evalModelPath, optimizerPath, cacheDirPath, requiresGrad)
            Log.d(LOG_TAG, "Successfully created the training model. Native handle at $model")

        }

    }

    fun performTrainStep(trainData : List<String>) : Float {
        if (model == 0L) {
            Log.e(LOG_TAG, "No native training model has been created.")
            return 0F
        }

        val batchSize = trainData.size

        val tokenizedExamples: List<List<Int>> = trainData.map { tokenizer.tokenize(it).toMutableList() }
        //tokenizer.tokenizeBatch(trainData)

        val maxSequenceLength = tokenizedExamples.maxOf { it.size }

        val paddedExamples: List<List<Int>> = tokenizedExamples.map {
            it + List(maxSequenceLength - it.size) { tokenizer.padToken }
        }

        val flatInputIds: LongArray = paddedExamples.flatten().map { it.toLong() }.toLongArray()
        val loss = performTraining(model, flatInputIds, batchSize, maxSequenceLength)
        Log.d(LOG_TAG, loss.toString())
        return loss
    }

    fun destroySession(saveCheckpoint: Boolean) {
        Log.d(LOG_TAG, "Destroying training session and saving checkpoint...")
        releaseTrainingSession(model, saveCheckpoint = saveCheckpoint)
    }

    /**
     * Performs test training with the hardcoded input ids. This will vary based on model.
     */
    private fun performTestTraining() : Float {

        val batchSize = 2
        val sequenceLength = 10

        // TinyLlama batched input token ids
        // "This is a test, hello from world."
        // "This is a test, hello to world."
        val inputIds = longArrayOf(1, 910, 338, 263, 1243, 29892, 22172, 515, 3186, 29889, 1, 910, 338, 263, 1243, 29892, 22172, 304, 3186, 29889)

        return performTraining(model, inputIds, batchSize, sequenceLength)
    }

    external fun releaseTrainingSession(session: Long, saveCheckpoint : Boolean)

    external fun createTrainingSession(checkpointPath:String, trainModelPath: String, evalModelPath: String,
                                       optimizerModelPath: String, cacheDirPath: String, requiresGrad: Array<String>) : Long

    external fun performTraining(session: Long, input_ids: LongArray, batchSize: Int,
                                 sequenceLength: Int) : Float

}
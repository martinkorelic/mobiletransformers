package com.example.orttransformer

import android.util.Log
import kotlinx.coroutines.flow.MutableSharedFlow

class ORTGeneratorNative(private var tokenizer: ORTGenAiTokenizer) {

    private var LOG_TAG = "ORTGeneratorNative"

    private var inferenceModel : Long = 0

    var decodedText = ""
    var prefillPhaseTime : Double = 0.0
    var generationTime : Double = 0.0

    fun createInferenceModel(inferenceModelPath : String, inferenceModelName : String) {
        inferenceModel = createInferenceSession(inferenceModelPath, inferenceModelName)
    }

    fun createInferenceModelFromTraining(trainModel : Long, inferenceModelPath : String, inferenceModelName : String) {
        inferenceModel = createInferenceSessionFromTraining(inferenceModelPath, inferenceModelName, trainModel)
    }

    fun createInferenceModelFromCheckpoint(inferenceModelPath : String, inferenceModelName : String, checkpointPath: String, trainingConfigPath : String) {

        val requiresGradLayers = loadTrainableLayerNamesJSON(trainingConfigPath)

        if (requiresGradLayers == null) {
            Log.d(LOG_TAG, "Failed to load trainable layer names from training config.")
            return
        }

        inferenceModel = createInferenceSessionFromCheckpoint(inferenceModelPath, inferenceModelName, checkpointPath, requiresGradLayers)
    }

    /**
     * Generation loop with KV caching enabled in native inference model.
     * This generation loop generates token by token.
     * TODO: Does not support batched inputs as of now.
     *
     */
    suspend fun generate(promptText: String, generationConfig : Map<String, String>, sharedFlow: MutableSharedFlow<String>) : String {

        // Assert that the key "max_sequence_length" exists
        assert(generationConfig.containsKey("max_sequence_length")) {
            "Key 'max_sequence_length' does not exist in generationConfig"
        }

        val inputTokens = tokenizer.tokenize(promptText)

        if (inputTokens == null) {
            Log.d(LOG_TAG, "Failed to generate tokens for the prompt.")
            return ""
        }

        decodedText = ""

        var (inputIds, attentionMask, positionIds) = createModelInputs(inputTokens)
        val generatedIds = inputIds

        var decoded = 0
        var start = System.nanoTime()

        while (decoded < generationConfig["max_sequence_length"]!!.toInt()) {

            Log.d(LOG_TAG, inputIds.toString())

            val nextTokenId = performInferenceStep(
                inferenceModel,
                inputIds.toLongArray(),
                attentionMask.toLongArray(),
                positionIds.toLongArray(),
                1,
                attentionMask.size,
                inputIds.size,
                this.tokenizer.vocabSize
            )

            if (decoded == 0) {
                val end = System.nanoTime() // End time
                prefillPhaseTime = (end - start) / 1_000_000_000.0
                start = System.nanoTime()
            }

            // Append the next token ID to generated ids
            generatedIds.add(nextTokenId.toLong())

            // Replace the next input id (since we are generating token by token)
            inputIds = mutableListOf(nextTokenId.toLong())

            // Update the attention mask to reflect the new token
            attentionMask.add(1L)

            // Update position IDs by appending the next position index
            val nextPositionId = positionIds.last() + 1
            positionIds = mutableListOf(nextPositionId)

            var decodedToken = tokenizer.decode(intArrayOf(nextTokenId)).toString()
            sharedFlow.emit(decodedToken)
            decodedText += decodedToken
            decoded++

            Log.d(LOG_TAG, decodedText)

            if (inputIds.last().toInt() == this.tokenizer.eosToken)
                break

        }

        var end_time = System.nanoTime()

        generationTime = decoded / ((end_time - start) / 1_000_000_000.0)

        return decodedText
    }

    fun destroySession() {
        releaseInferenceSession(inferenceModel)
    }

    fun createModelInputs(inputIds: IntArray): Triple<MutableList<Long>, MutableList<Long>, MutableList<Long>> {
        val inputIdsList = inputIds.map { it.toLong() }.toMutableList()
        val attentionMaskList = MutableList(inputIds.size) { 1L }
        val positionIdsList = MutableList(inputIds.size) { it.toLong() }
        return Triple(inputIdsList, attentionMaskList, positionIdsList)
    }

    external fun performInferenceStep(session: Long, input_ids: LongArray, attention_mask: LongArray, position_ids : LongArray, batchSize: Int, sequenceLength: Int, pastSequenceLength : Int, vocabSize : Int) : Int

    external fun createInferenceSession(inferenceModelPath : String,inferenceModelName : String) : Long

    external fun createInferenceSessionFromTraining(inferenceModelPath : String, inferenceModelName : String, trainModel : Long) : Long

    external fun createInferenceSessionFromCheckpoint(inferenceModelPath : String, inferenceModelName : String, checkpointPath: String, requiresGrad: Array<String>) : Long

    external fun releaseInferenceSession(session: Long)

}
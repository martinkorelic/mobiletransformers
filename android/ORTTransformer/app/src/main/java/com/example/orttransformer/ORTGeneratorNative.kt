package com.example.orttransformer

import android.util.Log

class ORTGeneratorNative(private var tokenizer: ORTTokenizer) {

    private var LOG_TAG = "ORTGenerator"

    private var inferenceModel : Long = 0

    fun createInferenceModelFromTraining(trainModel : Long, inferenceModelPath : String) {
        inferenceModel = createInferenceSessionFromTraining(inferenceModelPath, trainModel)
    }

    fun createInferenceModelFromCheckpoint(inferenceModelPath : String, checkpointPath: String, trainingConfigPath : String) {

        val requiresGradLayers = loadTrainableLayerNamesJSON(trainingConfigPath)

        if (requiresGradLayers == null) {
            Log.d(LOG_TAG, "Failed to load trainable layer names from training config.")
            return
        }

        inferenceModel = createInferenceSessionFromCheckpoint(inferenceModelPath, checkpointPath, requiresGradLayers)
    }

    fun generate(promptText: String, maxSequenceLength : Int) : String {

        val inputTokens = tokenizer.tokenize(promptText)

        if (inputTokens == null) {
            Log.d(LOG_TAG, "Failed to generate tokens for the prompt.")
            return ""
        }

        var decodedText = ""

        val (inputIds, attentionMask, positionIds) = createModelInputs(inputTokens)

        while (inputIds.size < maxSequenceLength) {

            val nextTokenId = performInferenceStep(
                inferenceModel,
                inputIds.toLongArray(),
                attentionMask.toLongArray(),
                positionIds.toLongArray(),
                1,
                inputIds.size,
                this.tokenizer.vocabSize
            )

            // Append the next token ID to inputIds
            inputIds.add(nextTokenId.toLong())

            // Update the attention mask to reflect the new token
            attentionMask.add(1L)

            // Update position IDs by appending the next position index
            val nextPositionId = positionIds.last() + 1
            positionIds.add(nextPositionId)

            decodedText = tokenizer.decode(inputIds.map { it.toInt() }.toIntArray()).toString()

            Log.d(LOG_TAG, decodedText)

            if (inputIds.last().toInt() == this.tokenizer.eosToken)
                break

        }

        return decodedText
    }

    fun createModelInputs(inputIds: IntArray): Triple<MutableList<Long>, MutableList<Long>, MutableList<Long>> {
        val inputIdsList = inputIds.map { it.toLong() }.toMutableList()
        val attentionMaskList = MutableList(inputIds.size) { 1L }
        val positionIdsList = MutableList(inputIds.size) { it.toLong() }
        return Triple(inputIdsList, attentionMaskList, positionIdsList)
    }

    external fun performInferenceStep(session: Long, input_ids: LongArray, attention_mask: LongArray, position_ids : LongArray, batchSize: Int, sequenceLength: Int, vocabSize : Int) : Int

    external fun createInferenceSessionFromTraining(inferenceModelPath : String, trainModel : Long) : Long

    external fun createInferenceSessionFromCheckpoint(inferenceModelPath : String, checkpointPath: String, requiresGrad: Array<String>) : Long

}
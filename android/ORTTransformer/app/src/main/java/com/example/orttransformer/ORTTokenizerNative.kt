package com.example.orttransformer

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Native tokenizer based on Huggingface and Sentencepiece tokenizers.
 *
 *
 */
class ORTTokenizerNative (private val tokenizerDir : String) {

    val LOG_TAG = "ORTTokenizerNative"

    var vocabSize: Int = 0
    var eosToken: Int = 0
    var padToken : Int = 0
    var bosToken : Int = 1
    private var modelType: String = ""

    // TODO: Should be adjustable
    private val tokenizerFileName = "tokenizer.json"
    private var tokenizerModel : Long = 0

    init {

        val modelFields = readModelFieldsFromJson("${tokenizerDir}/genai_config.json")
        if (!modelFields) {
            Log.e(LOG_TAG, "Error reading training_config.json fields for tokenizer.")
        }
    }

    fun readModelFieldsFromJson(filePath: String): Boolean {
        try {
            // Read the JSON file from internal storage
            val jsonFile = File(filePath)
            if (!jsonFile.exists()) {
                println("File does not exist: $filePath")
                return false
            }

            val jsonString = jsonFile.readText()

            // Parse the JSON object
            val jsonObject = JSONObject(jsonString)

            // Navigate to the 'model' object
            val modelObject = jsonObject.getJSONObject("model")

            // Extract the fields 'type' and 'vocab_size'
            this.modelType = modelObject.getString("type")
            this.vocabSize = modelObject.getInt("vocab_size")
            this.padToken = modelObject.getInt("pad_token_id")
            this.bosToken = modelObject.getInt("bos_token_id")

            // Handle 'eos_token_id' which might be an int or an array
            val eosTokenField = modelObject.get("eos_token_id")
            this.eosToken = when (eosTokenField) {
                is Int -> eosTokenField
                // TODO: What to do with multiple eos tokens?
                is JSONArray -> eosTokenField.getInt(0)
                else -> {
                    println("Invalid eos_token_id format")
                    return false
                }
            }

            return true

        } catch (e: Exception) {
            e.printStackTrace()
        }
        return false
    }

    suspend fun createTokenizerModel() {
        if (tokenizerModel != 0L) {
            Log.d(LOG_TAG, "There is already a tokenizer active, please destroy the previous session to create a new tokenizer model.")
            return
        }
        // TODO: Track metrics?
        tokenizerModel = createTokenizerSession("${tokenizerDir}/${tokenizerFileName}")
    }

    suspend fun destroySession() {
        if (tokenizerModel == 0L) {
            Log.d(LOG_TAG, "No tokenizer is active.")
            return
        }
        releaseTokenizerSession(tokenizerModel)
        tokenizerModel = 0L
    }

    fun tokenize(sequence: String) : IntArray {
        val tokens = tokenizeString(tokenizerModel, sequence)
        return tokens
    }

    fun prependBosToken(originalArray: IntArray) : IntArray {
        val newArray = IntArray(originalArray.size + 1)

        newArray[0] = this.bosToken

        System.arraycopy(originalArray, 0, newArray, 1, originalArray.size)

        return newArray
    }

    fun decode(sequence: IntArray) : String {
        val stringSequence = decodeString(tokenizerModel, sequence)
        return stringSequence
    }

    fun decodeToken(tokenId : Int) : String {
        var tokenString = decodeToken(tokenizerModel, tokenId)

        // Handle subword markers
        tokenString = tokenString.replace("▁", " ")

        // Handle newline tokens
        tokenString = tokenString.replace("<0x0A>", "\n")

        // Handle padding and end-of-sequence tokens
        tokenString = tokenString.replace("<PAD>", "").replace("<EOS>", "")

        // Handle unknown tokens
        tokenString = tokenString.replace("<UNK>", "?")

        // Trim leading/trailing whitespace (optional)
        tokenString = tokenString.trim()
        return tokenString
    }

    external fun decodeToken(tokenizerModel: Long, tokenId: Int) : String

    external fun decodeString(tokenizerModel: Long, sequence: IntArray) : String

    external fun tokenizeString(tokenizerModel : Long, sequence : String) : IntArray

    external fun createTokenizerSession(tokenizerFilePath : String) : Long

    external fun releaseTokenizerSession(tokenizerModel: Long)
}
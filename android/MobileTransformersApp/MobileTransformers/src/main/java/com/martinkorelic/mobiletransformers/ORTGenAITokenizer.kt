package com.martinkorelic.mobiletransformers

import ai.onnxruntime.genai.Generator
import ai.onnxruntime.genai.Model
import ai.onnxruntime.genai.Tokenizer
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * ONNX Runtime GenAI tokenizer class
 */
@Deprecated("Unfinished and abandoned class, use ORTTokenizerNative")
class ORTGenAITokenizer(folderPath: String) {

    private var LOG_TAG = "ORTTokenizer"

    private var model: Model = Model(folderPath)
    private var tokenizer: Tokenizer = model.createTokenizer()
    //private var generatorParams : GeneratorParams = model.createGeneratorParams()

    var vocabSize: Int = 0
    var eosToken: Int = 0
    var padToken : Int = 0
    private var modelType: String = ""

    init {

        val modelFields = readModelFieldsFromJson("${folderPath}/genai_config.json")
        if (!modelFields) {
            Log.e(LOG_TAG, "Error reading training_config.json fields for tokenizer.")
        }
    }

    fun tokenize(promptText: String): IntArray {
        return tokenizer.encode(promptText).getSequence(0)
    }

    fun tokenizeBatch(textInputs: Array<String>): List<List<Int>> {
        val sequences = tokenizer.encodeBatch(textInputs)

        val tokenizedInputs : MutableList<MutableList<Int>> = mutableListOf()
        for (i in 0..<sequences.numSequences()) {
            tokenizedInputs.add(sequences.getSequence(i).toMutableList())
        }

        return tokenizedInputs
    }

    fun decode(inputIds: IntArray): String? {
        return tokenizer.decode(inputIds)
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
            this.eosToken = modelObject.getInt("eos_token_id")
            this.padToken = modelObject.getInt("pad_token_id")

            return true

        } catch (e: Exception) {
            e.printStackTrace()
        }
        return false
    }

    fun generate(promptText: String) {

        val tokens = tokenizer.encode(promptText)
        val generatorParams = model.createGeneratorParams()
        generatorParams.setInput(tokens)
        //Log.d("ORTGenerator", "${tokens.getSequence(0)}")
        val generator = Generator(model, generatorParams)
        val maxlen = 100
        var i = 0;
        while (i < maxlen) {
            Log.d("ORTGenerator", "Computing logits...")
            generator.computeLogits()
            Log.d("ORTGenerator", "token...")
            generator.generateNextToken()
            val seq = generator.getSequence(0)
            val decoded_token = tokenizer.decode(seq)
            Log.d("ORTGenerator", "${decoded_token}")
            i+=1
        }

    }
}
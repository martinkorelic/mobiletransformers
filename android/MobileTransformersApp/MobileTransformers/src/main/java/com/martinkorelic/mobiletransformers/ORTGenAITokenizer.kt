package com.martinkorelic.mobiletransformers

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * ONNX Runtime GenAI tokenizer class — RETIRED.
 *
 * DECOMPOSE(#11): this class wrapped the old onnxruntime-genai Java API (`model.createTokenizer()`,
 * `createGeneratorParams()`, `generator.computeLogits()`), which the real 0.14 AAR replaced with
 * constructor-based `Tokenizer(model)` / `GeneratorParams(model)` and a merged `generateNextToken()`. It is
 * dead code (never instantiated — `ORTTokenizerNative` is the real tokenizer) and #11 deletes it. Kept only
 * as a compiling stub so the module builds against the real AAR; the JSON field reader is preserved because
 * it is pure. The GenAI engine is now the C-API path (`genai_spike.cpp` → #11's `ModelRuntime`).
 */
@Deprecated("Retired GenAI-Java-API wrapper; use ORTTokenizerNative. Removed by #11.")
class ORTGenAITokenizer(folderPath: String) {

    private var LOG_TAG = "ORTGenAITokenizer"

    var vocabSize: Int = 0
    var eosToken: Int = 0
    var padToken: Int = 0
    private var modelType: String = ""

    init {
        if (!readModelFieldsFromJson("$folderPath/genai_config.json")) {
            Log.e(LOG_TAG, "Error reading genai_config.json fields for tokenizer.")
        }
    }

    private fun retired(): Nothing =
        throw NotImplementedError("ORTGenAITokenizer is retired (see #11); use ORTTokenizerNative")

    fun tokenize(promptText: String): IntArray = retired()

    fun tokenizeBatch(textInputs: Array<String>): List<List<Int>> = retired()

    fun decode(inputIds: IntArray): String? = retired()

    fun generate(promptText: String): Unit = retired()

    fun readModelFieldsFromJson(filePath: String): Boolean {
        try {
            val jsonFile = File(filePath)
            if (!jsonFile.exists()) {
                println("File does not exist: $filePath")
                return false
            }
            val jsonObject = JSONObject(jsonFile.readText())
            val modelObject = jsonObject.getJSONObject("model")
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
}

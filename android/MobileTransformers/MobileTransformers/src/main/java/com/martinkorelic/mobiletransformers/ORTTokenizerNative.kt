package com.martinkorelic.mobiletransformers

import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.reflect.TypeToken
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Native tokenizer based on Huggingface and Sentencepiece tokenizers.
 */
class ORTTokenizerNative (private val tokenizerDir : String) {

    init {
        NativeLibrary.ensureLoaded()
    }


    val LOG_TAG = "ORTTokenizerNative"

    var vocabSize: Int = 0

    private var modelType: String = ""

    // TODO: Should be adjustable
    private val tokenizerFileName = "tokenizer.json"
    private var tokenizerModel : Long = 0

    // Map to store special tokens with attributes
    private val specialTokenFileName = "special_tokens_map.json"
    val specialTokensMap: MutableMap<String, TokenAttributes> = mutableMapOf()
    private val addedTokensMap: MutableMap<Int, String> = mutableMapOf()

    // Tokenizer config
    private val tokenizerConfigFileName = "tokenizer_config.json"

    // Important text control tokens
    // This tokens should be set
    val eosToken : Int?
        get() = specialTokensMap["eos_token"]?.tokenId
    val padToken : Int?
        get() = specialTokensMap["pad_token"]?.tokenId
    val bosToken : Int?
        get() = specialTokensMap["bos_token"]?.tokenId

    /**
     * Checks if the given token ID is any EOS (End of Sequence) token
     * This includes eos_token, eos_token_0, eos_token_1, etc.
     * @param tokenId The token ID to check
     * @return true if the token ID matches any EOS token, false otherwise
     */
    fun isEosToken(tokenId: Int): Boolean {
        return specialTokensMap.entries
            .filter { it.key.startsWith("eos_token") }
            .any { it.value.tokenId == tokenId }
    }

    // Can be either found in embedding models
    val clsToken : Int?
        get() = specialTokensMap["cls_token"]?.tokenId
    val sepToken : Int?
        get() = specialTokensMap["sep_token"]?.tokenId

    var maximumTokenLength = 0

    var chatTemplate : String? = null

    init {

        val modelFields = readModelFieldsFromJson("${tokenizerDir}/mobiletransformers_tokenizer_config.json")
        loadTokenizerConfiguration("${tokenizerDir}/${tokenizerConfigFileName}")
        // Load special token map
        loadSpecialTokensMapFromFile("$tokenizerDir/$specialTokenFileName")

        // Deliberately AFTER the special-token map is populated. The probe render feeds the same
        // context a real turn does, and templates routinely reference `bos_token`/`eos_token`; probing
        // before the map exists would fail templates that work perfectly in use.
        validateChatTemplate()

        if (!modelFields) {
            Log.e(LOG_TAG, "Error reading training_config.json fields for tokenizer.")
        }
    }

    /**
     * Checks if the given token ID matches any of the special tokens
     * @param tokenId The token ID to check
     * @return true if the token ID matches any special token, false otherwise
     */
    fun isSpecialToken(tokenId: Int): Boolean {
        return specialTokensMap.values.any { it.tokenId == tokenId }
    }

    fun trimModelInputs(inputIds: MutableList<Long>,
                        attentionMask: MutableList<Long>,
                        positionIds: MutableList<Long>) : Triple<MutableList<Long>, MutableList<Long>, MutableList<Long>> {
        if (inputIds.size > this.maximumTokenLength) {
            val excessLength = inputIds.size - this.maximumTokenLength
            inputIds.subList(0, excessLength).clear()
            attentionMask.subList(0, excessLength).clear()
            positionIds.subList(0, excessLength).clear()
        }
        return Triple(
            inputIds,
            attentionMask,
            positionIds
        )
    }

    fun loadTokenizerConfiguration(tokenizerConfigPath: String) {
        val gson = Gson()

        try {
            // Read the tokenizer_config.json file
            val tokenizerConfigFile = File(tokenizerConfigPath)
            if (!tokenizerConfigFile.exists()) {
                Log.w(LOG_TAG, "Tokenizer config file not found: $tokenizerConfigPath.")
                return
            }

            val tokenizerConfig = gson.fromJson(tokenizerConfigFile.reader(), JsonObject::class.java)

            // Load and map tokenizer added tokens
            if (tokenizerConfig["added_tokens_decoder"] != null) {
                val gson = Gson()
                val type = object : TypeToken<Map<String, TokenAttributes>>() {}.type
                val tokensMap: Map<String, TokenAttributes> = gson.fromJson(tokenizerConfig["added_tokens_decoder"], type)
                addedTokensMap.clear()

                // Step 1: Update specialTokensMap with token IDs if their content matches
                for ((idStr, attr) in tokensMap) {
                    val id = idStr.toIntOrNull() ?: continue
                    val content = attr.content

                    for ((_, specialAttr) in specialTokensMap) {
                        if (specialAttr.content == content && specialAttr.tokenId == null) {
                            specialAttr.tokenId = id
                        }
                    }
                }

                // Step 2: Add missing special tokens to specialTokensMap
                for ((idStr, attr) in tokensMap) {
                    val id = idStr.toIntOrNull() ?: continue
                    val content = attr.content ?: ""

                    // Check if this is a special token and not already in specialTokensMap
                    val existsInSpecialMap = specialTokensMap.values.any { it.content == content }

                    if (!existsInSpecialMap) {
                        // Determine the key name based on content
                        val keyName = when {
                            content.contains("PAD", ignoreCase = true) -> "pad_token"
                            content.contains("UNK", ignoreCase = true) -> "unk_token"
                            content.contains("CLS", ignoreCase = true) -> "cls_token"
                            content.contains("SEP", ignoreCase = true) -> "sep_token"
                            content.contains("BOS", ignoreCase = true) -> "bos_token"
                            content.contains("EOS", ignoreCase = true) -> "eos_token"
                            content.contains("MASK", ignoreCase = true) -> "mask_token"
                            else -> content.lowercase().replace("[", "").replace("]", "") + "_token"
                        }

                        specialTokensMap[keyName] = TokenAttributes(
                            tokenId = id,
                            content = content,
                            lstrip = attr.lstrip,
                            normalized = attr.normalized,
                            rstrip = attr.rstrip,
                            single_word = attr.single_word
                        )
                    }
                }

                // Step 3: Add only non-special tokens to addedTokensMap
                for ((idStr, attr) in tokensMap) {
                    val id = idStr.toIntOrNull() ?: continue
                    val content = attr.content ?: ""
                    if (specialTokensMap.values.none { it.tokenId == id }) {
                        addedTokensMap[id] = content
                    }
                }
            }

            // Load additional tokenizer information
            if (tokenizerConfig["model_max_length"] != null) {
                tokenizerConfig["model_max_length"]?.asInt?.let { maxLength -> maximumTokenLength = if (maxLength == 0) Int.MAX_VALUE else maxLength }
            }

            // Extract the chat template. Only a CANDIDATE at this point — [validateChatTemplate]
            // decides whether it survives, once the special-token map is loaded.
            chatTemplate = resolveChatTemplate(tokenizerConfig, tokenizerConfigFile.parentFile)

            if (chatTemplate == null)   {
                Log.w(LOG_TAG, "Chat template not found for $tokenizerDir. Prompts will NOT be turn-wrapped.")
                return
            }

        } catch (e: Exception) {
            e.printStackTrace()
            return
        }
    }

    /**
 * Drop the resolved template unless Pebble can actually evaluate it.
     *
 * Delegates to [validatedChatTemplate]; the logic lives in the companion because this class
 * cannot be constructed off-device — its `init` loads the native library — and the resolution
 * rules are exactly the part worth testing on the JVM.
 */
    private fun validateChatTemplate() {
        chatTemplate = validatedChatTemplate(chatTemplate, getSpecialTokensWithContent())
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

            // Load special tokens if they are missing from specialTokensMap
            val tokenConfigs = mapOf(
                // Required tokens
                "eos_token" to Pair("eos_token_id", true),
                "pad_token" to Pair("pad_token_id", true),
                "bos_token" to Pair("bos_token_id", true),

                // Optional tokens
                "cls_token" to Pair("cls_token_id", false),
                "sep_token" to Pair("sep_token_id", false),
                "unk_token" to Pair("unk_token_id", false),
                "mask_token" to Pair("mask_token_id", false)
            )

            tokenConfigs.forEach { (tokenKey, config) ->
                val (configKey, requiresExisting) = config
                processSpecialToken(specialTokensMap, modelObject, tokenKey, configKey, requiresExisting)
            }

            return true

        } catch (e: Exception) {
            e.printStackTrace()
        }
        return false
    }

    // Helper function to get token ID from JSON (handles both int and array)
    private fun getTokenId(jsonObject: JSONObject, key: String): List<Int>? {
        return try {
            when {
                jsonObject.has(key) -> {
                    when (val tokenValue = jsonObject.get(key)) {
                        is Int -> listOf(tokenValue)
                        is JSONArray -> {
                            // Convert JSONArray to List<Int>
                            val result = mutableListOf<Int>()
                            for (i in 0 until tokenValue.length()) {
                                val item = tokenValue.get(i)
                                if (item is Int) {
                                    result.add(item)
                                }
                            }
                            result.ifEmpty { null }
                        }
                        else -> null
                    }
                }
                else -> null
            }
        } catch (e: Exception) {
            null
        }
    }

    // Helper function to update or create token attributes
    private fun updateTokenAttributes(
        specialTokensMap: MutableMap<String, TokenAttributes>,
        tokenKey: String,
        tokenIds: List<Int>
    ) {
        when (tokenIds.size) {
            0 -> return // No tokens to add
            1 -> {
                // Single token - use the original key
                val existingAttr = specialTokensMap[tokenKey]
                if (existingAttr == null) {
                    specialTokensMap[tokenKey] = TokenAttributes(tokenId = tokenIds[0])
                } else if (existingAttr.tokenId == null) {
                    existingAttr.tokenId = tokenIds[0]
                }
            }
            else -> {
                // Multiple tokens - create indexed keys
                tokenIds.forEachIndexed { index, tokenId ->
                    val indexedKey = if (index == 0) tokenKey else "${tokenKey}_$index"
                    val existingAttr = specialTokensMap[indexedKey]
                    if (existingAttr == null) {
                        specialTokensMap[indexedKey] = TokenAttributes(tokenId = tokenId)
                    } else if (existingAttr.tokenId == null) {
                        existingAttr.tokenId = tokenId
                    }
                }
            }
        }
    }

    // Simplified token processing function
    private fun processSpecialToken(
        specialTokensMap: MutableMap<String, TokenAttributes>,
        modelObject: JSONObject,
        tokenKey: String,
        configKey: String,
        requiresExisting: Boolean = false
    ) {
        val shouldProcess = if (requiresExisting) {
            // Only process if key exists but tokenId is missing (for required tokens like EOS, PAD, BOS)
            !specialTokensMap.containsKey(tokenKey) || specialTokensMap[tokenKey]?.tokenId == null
        } else {
            // Always try to process (for optional tokens like CLS, SEP)
            true
        }

        if (shouldProcess) {
            getTokenId(modelObject, configKey)?.let { tokenIds ->
                updateTokenAttributes(specialTokensMap, tokenKey, tokenIds)
            }
        }
    }

    // Function to read special_tokens_map.json and merge attributes with existing tokens
    fun loadSpecialTokensMapFromFile(filePath: String) {
        try {
            val jsonContent = File(filePath).readText()
            val specialTokensJson = JSONObject(jsonContent)

            // Iterate through all keys in the special_tokens_map.json
            specialTokensJson.keys().forEach { tokenKey ->
                val tokenValue = specialTokensJson.get(tokenKey)

                when (tokenValue) {
                    is JSONObject -> {
                        // Extract attributes from the JSON object
                        val attributesFromFile = TokenAttributes(
                            tokenId = null, // Keep existing tokenId, don't overwrite
                            content = tokenValue.optString("content", ""),
                            lstrip = tokenValue.optBoolean("lstrip", false),
                            normalized = tokenValue.optBoolean("normalized", false),
                            rstrip = tokenValue.optBoolean("rstrip", false),
                            single_word = tokenValue.optBoolean("single_word", false)
                        )

                        // Check if token already exists in specialTokensMap
                        val existingToken = specialTokensMap[tokenKey]

                        if (existingToken != null) {
                            // Merge with existing token (keep existing tokenId, update other attributes)
                            existingToken.apply {
                                // Only update attributes that have meaningful values from file
                                if (attributesFromFile.content.isNotEmpty()) {
                                    // Note: content is immutable in your data class, so you might need to recreate
                                    specialTokensMap[tokenKey] = TokenAttributes(
                                        tokenId = this.tokenId,
                                        content = attributesFromFile.content,
                                        lstrip = attributesFromFile.lstrip,
                                        normalized = attributesFromFile.normalized,
                                        rstrip = attributesFromFile.rstrip,
                                        single_word = attributesFromFile.single_word
                                    )
                                }
                            }
                        } else {
                            // Add new token (without tokenId, which should be set elsewhere)
                            specialTokensMap[tokenKey] = attributesFromFile
                        }
                    }
                    is JSONArray -> {
                        // Handle arrays like "additional_special_tokens"
                        // You might want to handle these differently based on your needs
                        // For now, skip or handle as needed
                        println("Skipping array token: $tokenKey")
                    }
                    else -> {
                        // Handle other types if needed
                        println("Skipping unsupported token type for: $tokenKey")
                    }
                }
            }

        } catch (e: Exception) {
            println("Error loading special tokens map: ${e.message}")
        }
    }

    /**
     * Retrieves a specific special token's content by its key.
     *
     * @param tokenName The key of the special token.
     * @return The content of the special token or null if not found.
     */
    fun getSpecialTokenContent(tokenName: String): String? {
        return specialTokensMap[tokenName]?.content
    }

    fun getSpecialTokensWithContent(): Map<String, String> {
        return specialTokensMap.mapValues {
            it.value.content
        }
    }

    /**
     * Retrieves all special tokens with their attributes.
     *
     * @return A map containing all special tokens with attributes.
     */
    fun getAllSpecialTokens(): Map<String, TokenAttributes> {
        return specialTokensMap.toMap()
    }

    fun loadTokens(filePath: String, tokenMap: MutableMap<String, TokenAttributes>): Boolean {
        try {
            val file = File(filePath)
            if (!file.exists()) {
                Log.w(LOG_TAG, "Special tokens map file not found at $filePath.")
            }
            val fileContent = file.readText()

            val gson = Gson()
            val type = object : TypeToken<Map<String, TokenAttributes>>() {}.type
            val tokensMap: Map<String, TokenAttributes> = gson.fromJson(fileContent, type)
            tokenMap.clear()
            tokenMap.putAll(tokensMap)
            return true
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    suspend fun createTokenizerModel() {
        if (tokenizerModel != 0L) {
            Log.d(LOG_TAG, "There is already a tokenizer active, please destroy the previous session to create a new tokenizer model.")
            return
        }
        // TODO: Track metrics?
        tokenizerModel = createTokenizerSession("${tokenizerDir}/${tokenizerFileName}")
    }

    fun destroySession() {
        if (tokenizerModel == 0L) {
            Log.d(LOG_TAG, "No tokenizer is active.")
            return
        }
        releaseTokenizerSession(tokenizerModel)
        tokenizerModel = 0L
    }

    fun tokenize(sequence: String,
                 prependBos : Boolean = false,
                 prependCls: Boolean = false,
                 appendSep : Boolean = false,
                 dropZero : Boolean = false) : IntArray {

        // Fail closed on an unopened session. `tokenizeString` dereferences this handle in native
        // code, so a 0 here is a SIGSEGV that takes down the whole instrumentation run and names
        // nothing — the constructor reads the configs but deliberately does NOT open the native
        // session, so calling tokenize() before createTokenizerModel() is an easy and silent mistake.
        // (It cost a device run on 2026-08-14.)
        check(tokenizerModel != 0L) {
            "tokenizer session is not open: call createTokenizerModel() before tokenize(). " +
                "The constructor only loads the JSON configs."
        }

        var tokens = tokenizeString(tokenizerModel, sequence)

        // Drop trailing zeros if requested
        if (dropZero) {
            tokens = dropTrailingZeros(tokens)
        }

        // Prepend CLS token if requested and available
        if (prependCls && this.clsToken != null) {
            tokens = prependClsToken(tokens)
        }

        // Append SEP token if requested and available
        if (appendSep && this.sepToken != null) {
            tokens = appendSepToken(tokens)
        }

        if (prependBos)
            return prependBosToken(tokens)
        return tokens
    }

    fun batchTokenize(
        sequences: List<String>,
        maxSequenceLength: Int? = null,
        padTokenId: Int = 0, // Usually 0 for padding, but should match your tokenizer's pad token
        paddingStrategy: PaddingStrategy = PaddingStrategy.RIGHT
    ) : Array<IntArray> {

        // First, tokenize all sequences
        val tokenizedSequences = sequences.map { sequence ->
            tokenize(sequence)
        }

        // Determine target length
        val targetLength = if (maxSequenceLength != null) {
            val maxActualLength = tokenizedSequences.maxOfOrNull { it.size } ?: 0
            minOf(maxSequenceLength, maxActualLength)
        } else {
            tokenizedSequences.maxOfOrNull { it.size } ?: 0
        }

        // Pad or truncate sequences to target length
        return tokenizedSequences.map { tokens ->
            when {
                tokens.size == targetLength -> tokens
                tokens.size > targetLength -> {
                    // Truncate from the right (keep beginning)
                    tokens.sliceArray(0 until targetLength)
                }
                else -> {
                    // Pad to target length
                    when (paddingStrategy) {
                        PaddingStrategy.RIGHT -> {
                            tokens + IntArray(targetLength - tokens.size) { padTokenId }
                        }
                        PaddingStrategy.LEFT -> {
                            IntArray(targetLength - tokens.size) { padTokenId } + tokens
                        }
                    }
                }
            }
        }.toTypedArray()
    }

    // Alternative version with attention masks
    fun batchTokenizeWithAttentionMask(
        sequences: List<String>,
        maxSequenceLength: Int? = null,
        padTokenId: Int = 0,
        paddingStrategy: PaddingStrategy = PaddingStrategy.RIGHT
    ) : Pair<Array<IntArray>, Array<IntArray>> {

        val tokenizedSequences = sequences.map { sequence ->
            tokenize(sequence)
        }

        val targetLength = if (maxSequenceLength != null) {
            val maxActualLength = tokenizedSequences.maxOfOrNull { it.size } ?: 0
            minOf(maxSequenceLength, maxActualLength)
        } else {
            tokenizedSequences.maxOfOrNull { it.size } ?: 0
        }

        val paddedTokens = mutableListOf<IntArray>()
        val attentionMasks = mutableListOf<IntArray>()

        tokenizedSequences.forEach { tokens ->
            when {
                tokens.size == targetLength -> {
                    paddedTokens.add(tokens)
                    attentionMasks.add(IntArray(targetLength) { 1 })
                }
                tokens.size > targetLength -> {
                    val truncated = tokens.sliceArray(0 until targetLength)
                    paddedTokens.add(truncated)
                    attentionMasks.add(IntArray(targetLength) { 1 })
                }
                else -> {
                    val paddingLength = targetLength - tokens.size
                    when (paddingStrategy) {
                        PaddingStrategy.RIGHT -> {
                            val padded = tokens + IntArray(paddingLength) { padTokenId }
                            val mask = IntArray(tokens.size) { 1 } + IntArray(paddingLength) { 0 }
                            paddedTokens.add(padded)
                            attentionMasks.add(mask)
                        }
                        PaddingStrategy.LEFT -> {
                            val padded = IntArray(paddingLength) { padTokenId } + tokens
                            val mask = IntArray(paddingLength) { 0 } + IntArray(tokens.size) { 1 }
                            paddedTokens.add(padded)
                            attentionMasks.add(mask)
                        }
                    }
                }
            }
        }

        return paddedTokens.toTypedArray() to attentionMasks.toTypedArray()
    }

    enum class PaddingStrategy {
        LEFT,   // Pad at the beginning
        RIGHT   // Pad at the end (default)
    }

    fun prependBosToken(originalArray: IntArray): IntArray {
        if (originalArray.isNotEmpty() && originalArray[0] == this.bosToken) {
            return originalArray
        }

        if (this.bosToken == null) {
            return originalArray
        }

        val newArray = IntArray(originalArray.size + 1)
        newArray[0] = this.bosToken!!
        System.arraycopy(originalArray, 0, newArray, 1, originalArray.size)

        return newArray
    }

    private fun prependClsToken(originalArray: IntArray): IntArray {
        if (originalArray.isNotEmpty() && originalArray[0] == this.clsToken) {
            return originalArray
        }

        val newArray = IntArray(originalArray.size + 1)
        newArray[0] = this.clsToken!!
        System.arraycopy(originalArray, 0, newArray, 1, originalArray.size)

        return newArray
    }

    private fun appendSepToken(originalArray: IntArray): IntArray {
        if (originalArray.isNotEmpty() && originalArray[originalArray.size - 1] == this.sepToken) {
            return originalArray
        }

        val newArray = IntArray(originalArray.size + 1)
        System.arraycopy(originalArray, 0, newArray, 0, originalArray.size)
        newArray[originalArray.size] = this.sepToken!!

        return newArray
    }

    private fun dropTrailingZeros(originalArray: IntArray): IntArray {
        if (originalArray.isEmpty()) {
            return originalArray
        }

        // Find the last non-zero element
        var lastNonZeroIndex = originalArray.size - 1
        while (lastNonZeroIndex >= 0 && originalArray[lastNonZeroIndex] == 0) {
            lastNonZeroIndex--
        }

        // If all elements are zero, return empty array
        if (lastNonZeroIndex < 0) {
            return IntArray(0)
        }

        // If no trailing zeros, return original array
        if (lastNonZeroIndex == originalArray.size - 1) {
            return originalArray
        }

        // Create new array without trailing zeros
        val newArray = IntArray(lastNonZeroIndex + 1)
        System.arraycopy(originalArray, 0, newArray, 0, lastNonZeroIndex + 1)

        return newArray
    }

    fun decode(sequence: IntArray) : String {
        val stringSequence = decodeString(tokenizerModel, sequence)
        return stringSequence
    }

    fun decodeToken(tokenId : Int) : String {

        // Check if it's a special token, skip
        if (addedTokensMap[tokenId] != null) {
            return ""
        }

        var tokenString = decodeToken(tokenizerModel, tokenId)

        Log.d(LOG_TAG, "Token: $tokenString")

        // Handle subword markers
        // TODO: Somehow determine in advance which are they
        tokenString = tokenString.replace("▁", " ")
        tokenString = tokenString.replace("Ġ", " ")
        tokenString = tokenString.replace("Ċ", "\n")

        // Handle newline tokens
        tokenString = tokenString.replace("<0x0A>", "\n")

        return tokenString
    }

    external fun decodeToken(tokenizerModel: Long, tokenId: Int) : String

    external fun decodeString(tokenizerModel: Long, sequence: IntArray) : String

    external fun tokenizeString(tokenizerModel : Long, sequence : String) : IntArray

    external fun createTokenizerSession(tokenizerFilePath : String) : Long

    external fun releaseTokenizerSession(tokenizerModel: Long)

    /**
 * Chat-template resolution, kept static and side-effect-free so it can be tested on the JVM.
     *
 * The instance side of this class cannot be constructed off-device — `init` runs
 * `System.loadLibrary` — so anything living only there is unreachable by the host suite. That is
 * how the sibling-file bug survived: nothing host-side could observe [resolveChatTemplate]'s
 * result. Mirrors the shape of `packages.ToolCallSupport`.
 */
    companion object {
        private const val TAG = "ORTTokenizerNative"

        /** Where the exporter puts the template, as a sibling of `tokenizer_config.json`. */
        const val CHAT_TEMPLATE_FILE_NAME = "chat_template.jinja"

    /**
 * The package's Jinja chat template, from either place it may live.
     *
 * The sibling file is the normal case, and reading only the inline key was the bug:
 * `export/pipeline.py::_emit_chat_template` writes the template to
 * [CHAT_TEMPLATE_FILE_NAME] beside `tokenizer_config.json` and leaves **no** `chat_template`
 * key behind, and the installers flatten it into `tokenizer/`. So the key lookup found
 * nothing for every package the exporter has ever produced, [ORTConversationState] was never
 * constructed, and no plain-chat prompt was ever wrapped in the model's turn format.
     *
 * The inline key is still checked first: packages predating that change carry it, and one
 * shipping both is stating a deliberate override.
     *
 * Note this does NOT share `ToolCallSupport.readChatTemplate`. That function's fallback
 * returns the whole of `tokenizer_config.json` when the substring matches — correct for
 * sniffing a dialect out of it, useless as a template, and megabytes wide on a large vocab.
 */
        @JvmStatic
        fun resolveChatTemplate(tokenizerConfig: JsonObject?, tokenizerDir: File?): String? {
            // Guard the type instead of calling asString blind: chat_template is sometimes a LIST of
            // named templates ({name, template}) rather than a string, and asString throws on that.
            val inline = tokenizerConfig?.get("chat_template")
            ?.takeIf { it.isJsonPrimitive }
            ?.asString
            ?.takeIf { it.isNotBlank() }
            if (inline != null) return inline

            val sibling = tokenizerDir?.let { File(it, CHAT_TEMPLATE_FILE_NAME) } ?: return null
            if (!sibling.isFile) return null
            return runCatching { sibling.readText(Charsets.UTF_8) }
.onFailure { Log.w(TAG, "Could not read $sibling", it) }
.getOrNull()
            ?.takeIf { it.isNotBlank() }
}

    /**
 * [candidate] if Pebble can evaluate it against a probe turn, otherwise null.
     *
 * A template that throws is strictly worse than no template: without this check the failure
 * lands mid-generation, once per turn, rather than once at load. Pebble is not Jinja —
 * FunctionGemma's template alone uses `namespace`, `dictsort` and macros it does not
 * implement — so catching that here and falling back to an unwrapped prompt is the designed
 * outcome, not a regression.
     *
 * @param specialTokens must already be populated; templates routinely reference
 * `bos_token`/`eos_token`, and probing against an empty map would reject working templates.
 */
        @JvmStatic
        fun validatedChatTemplate(candidate: String?, specialTokens: Map<String, String>): String? {
            if (candidate == null) return null
            val rendered = runCatching {
                val context = mutableMapOf<String, Any>(
                    "messages" to listOf(
                        mapOf("role" to "user", "content" to "ping"),
                        mapOf("role" to "assistant", "content" to "pong"),
),
                    "add_generation_prompt" to true,
        )
                context.putAll(specialTokens)
                ORTChatTemplateHandler(candidate).buildInput(context)
            }.getOrElse { failure ->
                Log.w(TAG, "Chat template failed its probe render; continuing unwrapped.", failure)
                return null
    }

            // A template evaluating to nothing is a silent prompt-eater — generation would be handed
            // an empty string every turn. Treat it exactly like one that threw.
            if (rendered.isBlank()) {
                Log.w(TAG, "Chat template rendered empty on probe; continuing unwrapped.")
                return null
    }

            Log.i(TAG, "Chat template active (${candidate.length} chars); prompts are turn-wrapped.")
            return candidate
    }
    }
}
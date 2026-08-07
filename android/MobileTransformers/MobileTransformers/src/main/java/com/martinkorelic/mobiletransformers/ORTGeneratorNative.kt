package com.martinkorelic.mobiletransformers

import android.util.Log
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.internal.runtime.HandoffPrecondition
import com.martinkorelic.mobiletransformers.repository.GenerationCallback
import com.martinkorelic.mobiletransformers.runtime.EngineCapabilities
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.ModelRuntime
import java.io.File

class ORTGeneratorNative(val cacheDir : String, private var tokenizer: ORTTokenizerNative, var _generationConfig : ORTGenerationConfig) : ModelRuntime {

    private var LOG_TAG = "ORTGeneratorNative"

    private var inferenceModel : Long = 0

    // #11: the guaranteed engine floor. maxContextLength reflects the configured generation length.
    // #23: supportsLoadMergedWeights now reflects the hardened handoff precondition for THIS model —
    // it is a non-throwing presence query (schema + file existence, no hashing); the throwing,
    // checksum-verifying gate lives in createInferenceModel below.
    override val capabilities: EngineCapabilities
        get() = EngineCapabilities(
            engine = InferenceEngine.NATIVE,
            supportsStreaming = true,
            supportsLoadMergedWeights = HandoffPrecondition.mergedWeightsPresent(inferenceDir()),
            maxContextLength = _generationConfig.maxSequenceLength,
        )

    private fun inferenceDir(): File = File("${cacheDir}/${_generationConfig.repoName}/inference")

    /** #11 [ModelRuntime.load]: open the Native session over `<cacheDir>/<repoName>/inference`. */
    override suspend fun load(cacheDir: String, config: ORTGenerationConfig) {
        generationConfig = config
        // #23: a freshly loaded session starts a fresh conversation — clear any KV/attention/history
        // state so a new session never inherits the previous conversation's prepend state.
        resetConversation()
        createInferenceModel()
    }

    /** #11 [ModelRuntime.release]. */
    override fun release() = destroySession()

    private var modelLoadTimeMs : Long = 0L

    var pastAttentionMaskLength : Int = 0

    var generationConfig: ORTGenerationConfig
        get() = _generationConfig
        set(value) {
            if (_generationConfig != value) {
                _generationConfig = value
                updateSamplingOptions(_generationConfig.sampling)
            }
        }

    // Conversation state if we are using multi-turn conversation
    val conversationState: ORTConversationState? = tokenizer.chatTemplate?.let {
        ORTConversationState(
            ORTChatTemplateHandler(it),
            tokenizer.getSpecialTokensWithContent(),
            generationConfig.systemPrompt
        )
    }

    suspend fun createInferenceModel() {
        var start : Long = 0L
        if (generationConfig.trackMetrics) {
            start = System.currentTimeMillis()
        }

        if (generationConfig.loadMergedWeights) {
            // #23: map-driven, fail-closed load precondition (replaces the retired inference/merged probe).
            // #9 writes merged tensors as flat per-tensor <name>.bin in inference/, keyed by
            // weight_handoff_map.json (+ sibling .sha256). loadMergedWeightsReady throws (naming the
            // offending tensor) if the map is present but any .bin is missing or its checksum fails —
            // no silent downgrade. An ABSENT map means nothing was merged: fall back to base weights.
            if (!HandoffPrecondition.loadMergedWeightsReady(inferenceDir())) {
                Log.w(LOG_TAG, "No weight_handoff_map.json in ${inferenceDir().absolutePath}; loading base weights")
                generationConfig.loadMergedWeights = false
            }
        }

        // Check if ONNX name has .onnx extension, add it if missing
        if (!generationConfig.onnxName.endsWith(".onnx", ignoreCase = true)) {
            generationConfig.onnxName += ".onnx"
        }

        // Create the inference session
        inferenceModel = createInferenceSession(
            "${cacheDir}/${generationConfig.repoName}/inference",
            generationConfig.onnxName,
            cacheDir,
            generationConfig.loadMergedWeights,
            generationConfig.deviceOptions.coreConfigId,
            generationConfig.deviceOptions.memoryConfigId,
            generationConfig.deviceOptions.executionProvider,
            generationConfig.deviceOptions.enableProfiling
        )

        // #23 fail-closed: a 0 handle means native session construction failed. When merged weights
        // were requested that is specifically "the merged tensors could not be applied" — dtype/shape
        // and byte-size validation are C++-side, so the Kotlin precondition above cannot catch them.
        // Never continue with a session built from the frozen base weights: it would generate
        // confidently from an untrained model.
        if (inferenceModel == 0L) {
            throw MissingArtifactException(
                if (generationConfig.loadMergedWeights) {
                    "failed to create the native inference session with merged weights from " +
                        "${inferenceDir().absolutePath} (see logcat for the offending tensor); " +
                        "refusing to fall back to base weights"
                } else {
                    "failed to create the native inference session for " +
                        "${inferenceDir().absolutePath}/${generationConfig.onnxName}"
                },
            )
        }

        // Update with sampling configurations
        updateSamplingOptions(generationConfig.sampling)

        if (generationConfig.trackMetrics) {
            modelLoadTimeMs = System.currentTimeMillis() - start
        }
    }

    /**
     * Generation loop with KV caching enabled in native inference model.
     * This generation loop generates token by token.
     * Generates multi-turn conversation if chat template is enabled in tokenizer configuration.
     */
    override fun generate(promptText: String,
                         generationArgs : ORTGenerationConfig,
                         callback: GenerationCallback?) : String {

        val decodedText = StringBuilder()

        // Apply new system prompt if exists
        generationArgs.systemPrompt?.let {
            conversationState?.setSystemPrompt(it)
        }

        var inputText = promptText

        try {

            conversationState?.let {
                // NOTE: Sometimes one token from the previous assistant message keeps prepending and sometimes not
                // TODO: Will need fix
                inputText = it.addUserMessage(inputText)
            }

            // Generate input tokens, but do not add if we already have past attention mask
            var inputTokens = tokenizer.tokenize(inputText, prependBos = pastAttentionMaskLength == 0)

            if (inputTokens == null) {
                Log.d(LOG_TAG, "Failed to generate tokens for the prompt.")
                return ""
            }

            var (inputIds, attentionMask, positionIds) = createModelInputs(inputTokens)
            val generatedIds = inputIds

            var decoded = 0

            var currentGenerationTime : Long = 0L
            var cumulativeGenerationTime : Long = 0L
            var prefillStartMs : Long = 0L
            var prefillTimeMs : Long = 0L
            var avgTokensPerS : Double = 0.0
            var isEosToken : Boolean = false

            Log.d(LOG_TAG, "Starting generation...")

            callback?.onStartGeneration(
                InferenceProgress(
                    token = "",
                    tokenId = -1,
                    totalDecodedTokens = decoded,
                    prefillTimeMs = prefillTimeMs,
                    timeToLoadModelMs = modelLoadTimeMs,
                    generationTimeMs = cumulativeGenerationTime,
                    avgTokensPerSecond =  avgTokensPerS,
                    isCompleted = this.tokenizer.isEosToken(inputIds.last().toInt())
                )
            )

            prefillStartMs = System.currentTimeMillis()

            // #24: maxSequenceLength carries the public `maxNewTokens` (new tokens to emit), so the
            // bound is EXCLUSIVE — `maxNewTokens = N` must emit exactly N tokens. This was `<=`, one
            // more token than GenAI produced for the same config; the two engines are now identical.
            while (decoded < generationArgs.maxSequenceLength) {

                // Trim maximum tokens from beginning of the sequence if they go over the limit
                if (inputIds.size > tokenizer.maximumTokenLength) {
                    val trimmedInputs = tokenizer.trimModelInputs(inputIds, attentionMask, positionIds)
                    inputIds = trimmedInputs.first
                    attentionMask = trimmedInputs.second
                    positionIds = trimmedInputs.third
                }

                Log.d(LOG_TAG, inputIds.toString())

                if (generationArgs.trackMetrics && decoded == 0) {
                    prefillTimeMs = prefillStartMs - System.currentTimeMillis()
                } else if (generationArgs.trackMetrics) {
                    currentGenerationTime = System.currentTimeMillis()
                }

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

                if (generationArgs.trackMetrics && decoded != 0) {
                    cumulativeGenerationTime += (System.currentTimeMillis() - currentGenerationTime)
                }

                if (generationArgs.trackMetrics && decoded == 0) {
                    prefillTimeMs = System.currentTimeMillis() - prefillStartMs
                } else if (generationArgs.trackMetrics) {
                    avgTokensPerS = decoded.toDouble() / (cumulativeGenerationTime / 1000.0)
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

                var decodedToken = tokenizer.decodeToken(nextTokenId)

                // Append the new token to the decoded text
                isEosToken = this.tokenizer.isEosToken(inputIds.last().toInt())

                // Let's not append eosToken
                if (!isEosToken)
                    decodedText.append(decodedToken)
                // Emit token if needed
                callback?.onPartialResult(
                    InferenceProgress(
                        token = decodedToken,
                        tokenId = inputIds.last().toInt(),
                        totalDecodedTokens = decoded,
                        prefillTimeMs = prefillTimeMs,
                        timeToLoadModelMs = modelLoadTimeMs,
                        generationTimeMs = cumulativeGenerationTime,
                        avgTokensPerSecond =  avgTokensPerS,
                        isCompleted = isEosToken
                    )
                )
                decoded++

                Log.d(LOG_TAG, decodedText.toString())

                // Break if end of sequence
                // Also break if assistant has completed the sequence if multi-turn is enabled
                if (isEosToken)
                    break
            }

            // If using multi-turn conversation, we need to add assistant message back
            conversationState?.let {
                it.addAssistantMessage(decodedText.toString())
                pastAttentionMaskLength = attentionMask.size - 2
            }

            callback?.onCompletion(
                InferenceProgress(
                    token = "",
                    tokenId = -1,
                    totalDecodedTokens = decoded,
                    prefillTimeMs = prefillTimeMs,
                    timeToLoadModelMs = modelLoadTimeMs,
                    generationTimeMs = cumulativeGenerationTime,
                    avgTokensPerSecond =  avgTokensPerS,
                    isCompleted = true
            ))
        } catch (e: Throwable) {
            Log.e(LOG_TAG, e.toString())
            callback?.onError(e)
        }

        return decodedText.toString()
    }

    fun resetConversation() {
        conversationState?.resetForNewConversation()
        pastAttentionMaskLength = 0
    }

    fun destroySession() {
        releaseInferenceSession(inferenceModel)
        resetConversation()
        inferenceModel = 0
        modelLoadTimeMs = 0L
    }

    fun createModelInputs(inputIds: IntArray): Triple<MutableList<Long>, MutableList<Long>, MutableList<Long>> {
        val inputIdsList = inputIds.map { it.toLong() }.toMutableList()
        val attentionMaskList = MutableList(pastAttentionMaskLength + inputIds.size) { 1L }
        val positionIdsList = MutableList(inputIds.size) { it.toLong() }
        return Triple(inputIdsList, attentionMaskList, positionIdsList)
    }

    fun updateSamplingOptions(args : SamplingOptions) {
        // #24: single source for the native ordinal via SamplingMethod.nativeOrdinal (replaces the old
        // methodMap). fromWire fails closed on an unknown method rather than silently defaulting to greedy.
        // (topK is always passed explicitly here, so the C++ struct's top_k=50 default is never observed.)
        val methodInt = SamplingMethod.fromWire(args.method).nativeOrdinal
        setSamplingConfig(inferenceModel, methodInt, args.temperature, args.topK, args.topP, args.seed)
    }

    external fun performInferenceStep(session: Long, input_ids: LongArray, attention_mask: LongArray, position_ids : LongArray, batchSize: Int, sequenceLength: Int, pastSequenceLength : Int, vocabSize : Int) : Int

    external fun createInferenceSession(inferenceModelPath : String, inferenceModelName : String, cacheDirPath : String, loadMergedWeights : Boolean, coreConfigId : String, memoryConfigId : String, executionProvider : String, enableProfiling : Boolean) : Long

    external fun releaseInferenceSession(session: Long)

    external fun setSamplingConfig(session: Long, samplingMethod : Int, temperature : Float, topK : Int, topP : Float, seed : Int)

}
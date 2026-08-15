package com.martinkorelic.mobiletransformers

import android.util.Log
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.internal.runtime.GenerationInputs
import com.martinkorelic.mobiletransformers.internal.runtime.HandoffPrecondition
import com.martinkorelic.mobiletransformers.repository.GenerationCallback
import com.martinkorelic.mobiletransformers.runtime.EngineCapabilities
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.ModelRuntime
import java.io.File
import com.martinkorelic.mobiletransformers.packages.PackagePaths

class ORTGeneratorNative(val cacheDir : String, private var tokenizer: ORTTokenizerNative, var _generationConfig : ORTGenerationConfig) : ModelRuntime {

    init {
        NativeLibrary.ensureLoaded()
    }


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

    private fun inferenceDir(): File = PackagePaths.forCache(cacheDir, _generationConfig.repoName).inference

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
            PackagePaths.forCache(cacheDir, generationConfig.repoName).inference.absolutePath,
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

            // Captured before the loop mutates inputIds down to a single token per step.
            val promptTokens = inputIds.size
            val contextLimit = tokenizer.maximumTokenLength

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
                    isCompleted = this.tokenizer.isEosToken(inputIds.last().toInt()),
                    promptTokenCount = promptTokens,
                    contextLimit = contextLimit,
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
                        isCompleted = isEosToken,
                        promptTokenCount = promptTokens,
                        contextLimit = contextLimit,
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
                // The native KV cache holds every token that has been *run through* the model. The loop
                // appends a mask slot for each newly sampled token, and the last sampled token has not
                // been fed forward, so the cache length is exactly `attentionMask.size - 1`.
                //
                // This was `- 2`, so the next turn built a mask one entry short of `past + new` and ORT
                // aborted the process inside the first attention Add:
                //   "Attempting to broadcast an axis by a dimension other than 1. 51 by 52".
                // Only reachable on a *second* generate in one session, which no host test can reach.
                pastAttentionMaskLength = attentionMask.size - 1
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
                    isCompleted = true,
                    promptTokenCount = promptTokens,
                    contextLimit = contextLimit,
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
        // Reset the NATIVE cache too. Clearing only the Kotlin counter left the session still holding
        // the previous conversation's keys and values, so the two halves disagreed about how many
        // tokens were cached — the same disagreement that surfaces as a short attention mask.
        if (inferenceModel != 0L) {
            nativeResetKvCache(inferenceModel)
        }
    }

    /**
     * The KV-cache length according to the SESSION, which is the only authority on it.
     *
     * `pastAttentionMaskLength` is kept as a mirror for logging and for the `prependBos` decision, but
     * the mask is built from this. Two independent counts of the same thing is what allowed a mask of
     * `past + new - 1` to be sent, and on a transformers >= 4.57 graph that fails inside ORT at
     * `/model/Gather_5` with a message naming neither the mask nor the cache.
     */
    private fun cachedTokenCount(): Int =
        if (inferenceModel != 0L) nativePastSequenceLength(inferenceModel) else 0

    /**
     * Numbers off one prefill pass over [tokens], for conformance assertions.
     *
     * The device mirror of the host's `train_inference_parity` gate: same causal shift, so
     * [InferenceMetrics.crossEntropyNats] is directly comparable to the number the exporter checks.
     * Nothing else on device could see logits at all — `performInferenceStep` samples internally and
     * returns a token id — which is why post-merge numerical correctness went unasserted.
     *
     * Runs as a single pass against an EMPTY cache (it resets first, and the native side resets after),
     * so repeated calls are independent and the conversation is not advanced.
     */
    fun inferenceMetrics(tokens: IntArray, vocabSize: Int): InferenceMetrics {
        check(inferenceModel != 0L) { "inference session is not open" }
        require(tokens.size >= 2) {
            "need at least 2 tokens to score one (prediction, target) pair, got ${tokens.size}"
        }
        nativeResetKvCache(inferenceModel)
        val plan = GenerationInputs.plan(tokens, pastLength = 0)
        val raw = nativeInferenceMetrics(
            inferenceModel,
            plan.inputIds.toLongArray(),
            plan.attentionMask.toLongArray(),
            plan.positionIds.toLongArray(),
            1,
            plan.attentionMask.size,
            tokens.size,
            vocabSize,
        ) ?: error("native inference metrics returned no result")
        pastAttentionMaskLength = 0
        return InferenceMetrics(
            argmax = raw[0].toInt(),
            maxLogit = raw[1],
            sum = raw[2],
            sumOfSquares = raw[3],
            crossEntropyNats = raw[4],
        )
    }

    /** @see inferenceMetrics */
    data class InferenceMetrics(
        val argmax: Int,
        val maxLogit: Double,
        val sum: Double,
        val sumOfSquares: Double,
        val crossEntropyNats: Double,
    ) {
        /**
         * True when this reduction differs from [other] beyond float noise.
         *
         * Four statistics rather than one: a constant shift leaves `argmax` alone, a redistribution
         * leaves `sum` alone. Used to assert a merge actually changed the computation.
         */
        fun differsFrom(other: InferenceMetrics, tolerance: Double = 1e-6): Boolean =
            argmax != other.argmax ||
                kotlin.math.abs(maxLogit - other.maxLogit) > tolerance ||
                kotlin.math.abs(sum - other.sum) > tolerance ||
                kotlin.math.abs(sumOfSquares - other.sumOfSquares) > tolerance
    }

    fun destroySession() {
        releaseInferenceSession(inferenceModel)
        resetConversation()
        inferenceModel = 0
        modelLoadTimeMs = 0L
    }

    /**
     * The step inputs for [inputIds], continuing from whatever is already in the KV cache.
     *
     * The planning itself lives in [GenerationInputs] so it is host-testable — the position-ids/mask
     * disagreement this used to carry was only reachable on a second turn, i.e. only on a phone. See
     * that object for the invariant and the defect it now pins.
     *
     * Within a turn the decode loop continues the positions itself (`positionIds.last() + 1`), and
     * `trimModelInputs` drops from the FRONT, so a trimmed sequence stays contiguous.
     */
    fun createModelInputs(inputIds: IntArray): Triple<MutableList<Long>, MutableList<Long>, MutableList<Long>> {
        // Ask the session, do not trust the counter — see [cachedTokenCount].
        val cached = cachedTokenCount()
        if (cached != pastAttentionMaskLength) {
            Log.w(
                LOG_TAG,
                "KV cache length $cached disagrees with the tracked $pastAttentionMaskLength; " +
                    "using the session's value.",
            )
            pastAttentionMaskLength = cached
        }
        val plan = GenerationInputs.plan(inputIds, cached)
        return Triple(plan.inputIds, plan.attentionMask, plan.positionIds)
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

    /** Tokens currently in the session's KV cache — the single authority on the cache length. */
    external fun nativePastSequenceLength(session: Long) : Int

    /**
     * One forward pass reduced to `[argmax, maxLogit, sum, sumOfSquares, causalCrossEntropyNats]`.
     * A probe: it resets the KV cache afterwards and advances nothing. See [inferenceMetrics].
     */
    external fun nativeInferenceMetrics(
        session: Long,
        input_ids: LongArray,
        attention_mask: LongArray,
        position_ids: LongArray,
        batchSize: Int,
        sequenceLength: Int,
        newTokenCount: Int,
        vocabSize: Int,
    ) : DoubleArray?

    /** Drops the KV cache back to zero-length past for a new conversation. */
    external fun nativeResetKvCache(session: Long)

}
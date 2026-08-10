package com.martinkorelic.mobiletransformers

import android.util.Log
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.internal.runtime.HandoffPrecondition
import com.martinkorelic.mobiletransformers.repository.GenerationCallback
import com.martinkorelic.mobiletransformers.runtime.EngineCapabilities
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.ModelRuntime
import java.io.File
import com.martinkorelic.mobiletransformers.packages.PackagePaths

/**
 * GenAI [ModelRuntime] engine (#11) — the ONNX Runtime GenAI implementation over the SAME `inference/`
 * package the Native engine reads. Backed by `cpp/genai_runtime.cpp` (stable C API; runs on the
 * genai-paired stock ORT `libort_gen.so`). The generation loop lives here in Kotlin (one JNI step per
 * token) so it drives the **identical** [GenerationCallback]/[InferenceProgress] sequence as
 * [ORTGeneratorNative] — the facade/UI never branch on engine.
 *
 * Supersedes the deleted `ORTGenAINative` (which threw `NotImplementedError`) and `onnx-genai.cpp`.
 */
class ORTGeneratorGenAI(
    val cacheDir: String,
    private val tokenizer: ORTTokenizerNative,
    private var _generationConfig: ORTGenerationConfig,
) : ModelRuntime {

    private val LOG_TAG = "ORTGeneratorGenAI"
    private var handle: Long = 0
    private var modelLoadTimeMs: Long = 0L

    /**
     * Same chat-template rendering the Native engine applies (#24 parity).
     *
     * Without this the two engines were fed *different token sequences* for one `generate(prompt)`
     * call: Native rendered the prompt through the model's chat template while GenAI passed the raw
     * string to `OgaGenerator`. On the same package the first greedy token then differed ("Hello" vs
     * ","), which reads as a weights/graph divergence but is purely prompt construction — and it is
     * exactly what Gate 0.1 #1 asserts.
     */
    private val conversationState: ORTConversationState? = tokenizer.chatTemplate?.let {
        ORTConversationState(
            ORTChatTemplateHandler(it),
            tokenizer.getSpecialTokensWithContent(),
            _generationConfig.systemPrompt,
        )
    }

    override val capabilities: EngineCapabilities =
        EngineCapabilities(
            engine = InferenceEngine.GENAI,
            supportsStreaming = true,
            // Reads the SAME external-initializer folder as Native (Gate 0.1), so it must run the
            // same fail-closed gate — hardcoding `true` would have loaded base weights after a merge
            // with no warning, the exact silent downgrade #23 forbids on the Native side.
            supportsLoadMergedWeights =
                HandoffPrecondition.mergedWeightsPresent(PackagePaths.forCache(cacheDir, _generationConfig.repoName).inference),
            maxContextLength = _generationConfig.maxSequenceLength,
        )

    override suspend fun load(cacheDir: String, config: ORTGenerationConfig) {
        _generationConfig = config
        val start = System.currentTimeMillis()
        val dir = PackagePaths.forCache(cacheDir, config.repoName).inference.absolutePath
        // #23 parity: same map-driven precondition the Native engine runs. An absent map means
        // nothing was merged (fall back to base); a present-but-broken one throws.
        if (config.loadMergedWeights && !HandoffPrecondition.loadMergedWeightsReady(File(dir))) {
            Log.w(LOG_TAG, "No weight_handoff_map.json in $dir; loading base weights")
            config.loadMergedWeights = false
        }
        handle = nativeCreate(dir)
        if (handle == 0L) {
            throw IllegalStateException("GenAI OgaCreateModel failed for $dir (see logcat GenAIRuntime)")
        }
        val s = config.sampling
        // #24: the ONE source for the native sampling ordinal, identical to ORTGeneratorNative.
        nativeSetSampling(
            handle, SamplingMethod.fromWire(s.method).nativeOrdinal, s.temperature, s.topK, s.topP, s.seed,
        )
        modelLoadTimeMs = System.currentTimeMillis() - start
    }

    override fun generate(
        promptText: String,
        generationArgs: ORTGenerationConfig,
        callback: GenerationCallback?,
    ): String {
        val decodedText = StringBuilder()
        try {
            check(handle != 0L) { "GenAI session not loaded" }
            generationArgs.systemPrompt?.let { conversationState?.setSystemPrompt(it) }
            val renderedPrompt = conversationState?.addUserMessage(promptText) ?: promptText
            if (!nativeStart(handle, renderedPrompt, generationArgs.maxSequenceLength)) {
                throw IllegalStateException("GenAI failed to start generation")
            }

            var decoded = 0
            val genStart = System.currentTimeMillis()
            var prefillTimeMs = 0L

            callback?.onStartGeneration(
                InferenceProgress(
                    token = "", tokenId = -1, totalDecodedTokens = 0,
                    prefillTimeMs = 0L, timeToLoadModelMs = modelLoadTimeMs,
                    generationTimeMs = 0L, avgTokensPerSecond = 0.0, isCompleted = false,
                ),
            )

            while (decoded < generationArgs.maxSequenceLength && !nativeIsDone(handle)) {
                val piece = nativeStep(handle)
                if (decoded == 0) prefillTimeMs = System.currentTimeMillis() - genStart
                val tokenId = nativeLastToken(handle)
                if (tokenId < 0) break

                val genMs = System.currentTimeMillis() - genStart
                val avgTps = if (genMs > 0) decoded.toDouble() / (genMs / 1000.0) else 0.0
                val isEos = tokenizer.isEosToken(tokenId)
                if (!isEos) decodedText.append(piece)

                callback?.onPartialResult(
                    InferenceProgress(
                        token = piece, tokenId = tokenId, totalDecodedTokens = decoded,
                        prefillTimeMs = prefillTimeMs, timeToLoadModelMs = modelLoadTimeMs,
                        generationTimeMs = genMs, avgTokensPerSecond = avgTps, isCompleted = isEos,
                    ),
                )
                decoded++
                if (isEos) break
            }

            // Multi-turn parity with Native: the reply has to go back into the transcript, or a second
            // generate() would re-render the conversation without it.
            conversationState?.addAssistantMessage(decodedText.toString())

            callback?.onCompletion(
                InferenceProgress(
                    token = "", tokenId = -1, totalDecodedTokens = decoded,
                    prefillTimeMs = prefillTimeMs, timeToLoadModelMs = modelLoadTimeMs,
                    generationTimeMs = System.currentTimeMillis() - genStart,
                    // #24: was hardcoded 0.0, so GenAI always reported zero throughput through
                    // GenerationResult.avgTokensPerSecond. Recompute over the full run.
                    avgTokensPerSecond = finalAvgTps(decoded, genStart), isCompleted = true,
                ),
            )
        } catch (e: Throwable) {
            Log.e(LOG_TAG, e.toString())
            callback?.onError(e)
        }
        return decodedText.toString()
    }

    override fun release() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0
        }
        modelLoadTimeMs = 0L
    }

    private fun finalAvgTps(decoded: Int, genStart: Long): Double {
        val elapsedMs = System.currentTimeMillis() - genStart
        return if (elapsedMs > 0) decoded.toDouble() / (elapsedMs / 1000.0) else 0.0
    }

    private external fun nativeCreate(dir: String): Long
    private external fun nativeSetSampling(h: Long, method: Int, temperature: Float, topK: Int, topP: Float, seed: Int)
    private external fun nativeStart(h: Long, prompt: String, maxNewTokens: Int): Boolean
    private external fun nativeIsDone(h: Long): Boolean
    private external fun nativeStep(h: Long): String
    private external fun nativeLastToken(h: Long): Int
    private external fun nativeRelease(h: Long)

    companion object {
        init {
            NativeLibrary.ensureLoaded()
        }
    }
}

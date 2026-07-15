package com.martinkorelic.mobiletransformers

import android.util.Log
import com.martinkorelic.mobiletransformers.repository.GenerationCallback
import com.martinkorelic.mobiletransformers.runtime.EngineCapabilities
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.ModelRuntime

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

    override val capabilities: EngineCapabilities =
        EngineCapabilities(
            engine = InferenceEngine.GENAI,
            supportsStreaming = true,
            supportsLoadMergedWeights = true, // reads the same external-initializer folder (Gate 0.1)
            maxContextLength = _generationConfig.maxSequenceLength,
        )

    override suspend fun load(cacheDir: String, config: ORTGenerationConfig) {
        _generationConfig = config
        val start = System.currentTimeMillis()
        val dir = "$cacheDir/${config.repoName}/inference"
        handle = nativeCreate(dir)
        if (handle == 0L) {
            throw IllegalStateException("GenAI OgaCreateModel failed for $dir (see logcat GenAIRuntime)")
        }
        val s = config.sampling
        nativeSetSampling(handle, samplingMethodInt(s.method), s.temperature, s.topK, s.topP, s.seed)
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
            if (!nativeStart(handle, promptText, generationArgs.maxSequenceLength)) {
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

            callback?.onCompletion(
                InferenceProgress(
                    token = "", tokenId = -1, totalDecodedTokens = decoded,
                    prefillTimeMs = prefillTimeMs, timeToLoadModelMs = modelLoadTimeMs,
                    generationTimeMs = System.currentTimeMillis() - genStart,
                    avgTokensPerSecond = 0.0, isCompleted = true,
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

    private fun samplingMethodInt(method: String): Int =
        when (method) {
            "top_k" -> 1
            "top_p" -> 2
            else -> 0 // greedy
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
            System.loadLibrary("mobiletransformers")
        }
    }
}

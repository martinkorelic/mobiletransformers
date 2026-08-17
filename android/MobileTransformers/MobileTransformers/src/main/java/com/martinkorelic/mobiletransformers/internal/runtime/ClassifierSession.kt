package com.martinkorelic.mobiletransformers.internal.runtime

import android.content.Context
import com.martinkorelic.mobiletransformers.MissingArtifactException
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.ORTRetriever
import com.martinkorelic.mobiletransformers.ORTTokenizerNative
import com.martinkorelic.mobiletransformers.config.DeviceConfig
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import com.martinkorelic.mobiletransformers.packages.PackageTask
import com.martinkorelic.mobiletransformers.runtime.ClassificationResult
import com.martinkorelic.mobiletransformers.runtime.LabelScore
import java.io.File
import kotlin.math.exp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Runs a sequence-classification graph and turns its logits into named labels.
 *
 * ### Why this exists
 *
 * Encoder fine-tuning (#33) works end to end — export, training artifacts, a train step, a metric —
 * and there was **no way to run the resulting model**. `MobileTransformerModel` offered
 * generate / retrieve / ingest / train and nothing else, all of which assume a decoder. So a BERT
 * classifier could be trained on device and then never asked a question: the one thing a user would
 * want to do with it after training was the one thing the API could not do.
 *
 * ### Why it borrows the embedding session
 *
 * The native embedding entry points are exactly what a classification forward pass needs and nothing
 * more: `createEmbeddingSession` opens an encoder graph, and `performEmbeddingStep` feeds
 * `input_ids`/`attention_mask`/`token_type_ids` and returns the **raw first output tensor** as floats
 * (`inference::generateEmbedding` does no pooling of its own). For a classification head that tensor
 * is `logits[batch, num_labels]`, so the same call yields logits when pointed at the inference stage
 * with `embeddingDim = numLabels`. No new JNI, no C++ change.
 *
 * What it deliberately does **not** reuse is `ORTRetriever.createEmbeddingModel`, which resolves the
 * *embedding* stage, opens the embedder's own tokenizer and creates an ObjectBox vector store — three
 * things a classifier has no use for, one of which (`DimensionRegistry.requireSupported`) would reject
 * a two-label head outright.
 */
internal class ClassifierSession(
    private val context: Context,
    private val cacheDir: String,
    private val sanitizedRepoId: String,
    private val task: PackageTask,
) {

    private val paths get() = PackagePaths.forCache(cacheDir, sanitizedRepoId)

    /** Holds the JNI entry points; never used as a retriever. */
    private val native = ORTRetriever(cacheDir, context, ORTRagConfig(repoName = sanitizedRepoId))

    private var tokenizer: ORTTokenizerNative? = null
    private var session: Long = 0L

    suspend fun classify(text: String, device: DeviceConfig, topK: Int): ClassificationResult =
        withContext(Dispatchers.IO) {
            val labels = task.id2label
            if (labels.isEmpty()) {
                // Running the graph would still work and every prediction would come back as an
                // index, which is a number in a costume rather than an answer.
                throw MissingArtifactException(
                    "package '$sanitizedRepoId' declares no id2label, so a predicted class has no " +
                        "name. Re-export it with a newer exporter, which copies id2label into " +
                        "inference/${PackageTask.FILENAME}.",
                )
            }

            ensureOpen(device)
            val tok = tokenizer ?: throw MissingArtifactException("tokenizer did not open")

            // CLS/SEP the way an encoder expects them — the same framing `ORTRetriever.ingestData`
            // uses for the embedder, because it is the same family of graph.
            val tokens = tok.tokenize(text, prependCls = true, appendSep = true, dropZero = true)
            val maxLen = tok.maximumTokenLength.takeIf { it > 0 } ?: 512
            val length = minOf(tokens.size, maxLen)
            val inputIds = LongArray(length) { tokens[it].toLong() }
            val attentionMask = LongArray(length) { 1L }
            val tokenTypeIds = LongArray(length) { 0L }

            val logits = native.performEmbeddingStep(
                session = session,
                inputIds = inputIds,
                attentionMask = attentionMask,
                tokenTypeIds = tokenTypeIds,
                batchSize = 1,
                sequenceLength = length,
                // The head's width, not an embedding width. This is the whole trick.
                embeddingDim = labels.size,
            ) ?: throw MissingArtifactException("the classification graph returned no output")

            ClassificationResult(scores = softmaxToLabels(logits, labels), topK = topK)
        }

    private suspend fun ensureOpen(device: DeviceConfig) {
        val inferenceDir = paths.inference
        if (!File(inferenceDir, "model.onnx").isFile) {
            throw MissingArtifactException(
                "no inference/model.onnx in '$sanitizedRepoId' — nothing to classify with",
            )
        }
        if (tokenizer == null) {
            // The package's shared tokenizer, not `embedding/tokenizer`: a classifier's inputs are
            // tokenized by the model's own tokenizer, and an embedding stage need not even exist.
            tokenizer = ORTTokenizerNative(paths.tokenizer.absolutePath).also { it.createTokenizerModel() }
        }
        if (session == 0L) {
            session = native.createEmbeddingSession(
                inferenceDir.absolutePath,
                "model.onnx",
                cacheDir,
                device.memoryConfigId.wire,
                device.coreConfigId.wire,
                device.executionProvider.wire,
                device.enableProfiling,
            )
        }
    }

    fun close() {
        if (session != 0L) {
            runCatching { native.releaseEmbeddingSession(session) }
            session = 0L
        }
        tokenizer = null
    }

    companion object {
        /**
         * Softmax over the head's logits, paired with the package's own label names.
         *
         * Max-subtracted, which is not a nicety: a classification head's logits routinely exceed 80,
         * and `exp(80f)` overflows a Float to infinity, so the naive form returns NaN for exactly the
         * confident predictions it matters most to report.
         */
        fun softmaxToLabels(logits: FloatArray, id2label: Map<Int, String>): List<LabelScore> {
            val n = minOf(logits.size, id2label.size)
            if (n == 0) return emptyList()
            val head = logits.take(n)
            val max = head.max()
            val exps = head.map { exp((it - max).toDouble()) }
            val sum = exps.sum().takeIf { it > 0.0 } ?: 1.0
            return head.indices
                .map { i ->
                    LabelScore(
                        label = id2label[i] ?: "LABEL_$i",
                        score = exps[i] / sum,
                        index = i,
                    )
                }
                .sortedByDescending { it.score }
        }
    }
}

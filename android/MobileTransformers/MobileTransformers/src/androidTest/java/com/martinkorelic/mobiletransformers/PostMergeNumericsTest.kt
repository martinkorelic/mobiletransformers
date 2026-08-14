package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Post-merge numerical correctness — the conformance assertion the project did not have.
 *
 * The export pipeline gates every package on `artifacts/train_inference_parity.py`: the same tokens
 * through the train and inference graphs, one cross-entropy each, one bounded delta. **Nothing checked
 * the numbers after an ON-DEVICE merge.** `TrainMergeGenerateTest` hashes the trainable `.bin` files
 * and is explicit that this is not a numerical test; `TrainConvergenceTest` reads the training loss,
 * which never touches the merged inference graph at all.
 *
 * So the seam between "the merge wrote bytes" and "the merged graph computes the right thing" was
 * unverified — the recurring failure shape in this project, and the reason a frozen MARS adapter and a
 * no-op merge both survived every gate.
 *
 * This test closes it by measuring the merged inference graph directly, via
 * [ORTGeneratorNative.inferenceMetrics] (the only path by which logits reach Kotlin — the normal step
 * samples internally and returns a token id).
 *
 * **Both assertions are relative**, per the standing rule that an absolute threshold encodes one model
 * and silently measures the wrong thing when the fixture changes.
 *
 * ## Package-mutation hazard — read before running this class
 *
 * These tests train, so each stashes and restores `train/checkpoint` + `training_state.json` and
 * deletes its own dataset fixture, exactly as `ScheduledTrainingDeviceTest` does.
 *
 * **What is NOT restored is `inference/*.bin`**: a merge rewrites the per-tensor weights in place, and
 * stashing ~60 tensors on device per test would cost more than re-pushing. So **every test here needs a
 * freshly pushed package**, and running the class end to end leaves later tests reading whatever the
 * earlier ones merged. That is not hypothetical — it produced a confusing failure on 2026-08-14, where
 * `aLargeAdapterDeltaSurvivedTheMerge`'s pristine control tripped on a graph the *previous* test had
 * already merged.
 *
 * The controls fail loudly rather than silently mis-measuring, which is the intended behaviour. Run
 * one test at a time against a fresh package:
 *
 * ```
 * adb shell "rm -rf <dest> && mkdir -p <dest>" && adb push build/device_cache/. <dest>
 * ./gradlew :MobileTransformers:connectedDebugAndroidTest \
 *   -Pandroid.testInstrumentationRunnerArguments.class=…PostMergeNumericsTest#<oneTest>
 * ```
 */
@RunWith(AndroidJUnit4::class)
class PostMergeNumericsTest {

    private companion object {
        /**
         * Ceiling on the post-merge cross-entropy, as a fraction of the pre-merge value.
         *
         * Relative, and generous in the direction that matters: a few LoRA steps may move the loss
         * either way by a little, but a merge that corrupted the weights lands at or above the
         * uniform-prediction floor `ln(vocab_size)` — for this tokenizer ~10.8 against a healthy ~3-4,
         * i.e. multiples away rather than percents. This bound excludes that regime without pretending
         * to know which direction a one-step fine-tune moves a held-out probe.
         */
        const val MAX_LOSS_RATIO = 2.0

        /** Fixed probe tokens. Small ids, inside every supported vocabulary, deliberately constant. */
        val PROBE = intArrayOf(1, 338, 263, 1243, 310, 278, 1904, 29889)

        const val TRAIN_FILE = "mt_post_merge_cola"

        const val LOG_TAG = "PostMergeNumericsTest"

        /** Big enough that a wrong `scale * B@A` cannot hide behind a near-zero delta. */
        const val STEPS_FOR_LARGE_DELTA = 100
    }

    @Test
    fun mergedWeightsChangeTheComputationAndStayNumericallySane(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val trainDir = File(root, "$repoId/train")
        val checkpoint = File(trainDir, "checkpoint")
        val trainingState = File(trainDir, "training_state.json")
        val stash = File(root, "$repoId/.post_merge_stash").apply { mkdirs() }
        stashInto(stash, checkpoint, trainingState)

        File(trainDir, "$TRAIN_FILE.jsonl").writeText(
            (1..8).joinToString("\n") { i ->
                """{"sentence": "The cat sat on the mat number $i.", "label": ${i % 2}}"""
            } + "\n",
        )

        try {
            val before = probeMergedGraph(root, repoId)

            val model = MobileTransformers.fromPretrained(
                context = ctx,
                repoId = repoId,
                cacheDir = root.absolutePath,
                features = setOf(ModelFeature.Inference, ModelFeature.Training),
            )
            try {
                model.train(
                    DatasetConfig(
                        trainFile = TRAIN_FILE,
                        task = "cola",
                        maxSequenceLength = 64,
                        maxDatasetLength = 8,
                        datasetBatchSize = 4,
                    ),
                    TrainConfig(maxSteps = 3, batchSize = 2, mergeAtEnd = true),
                )
                model.merge()
            } finally {
                model.close()
            }

            val after = probeMergedGraph(root, repoId)

            android.util.Log.i(
                "PostMergeNumericsTest",
                "before=$before after=$after",
            )

            // 1. The merge must change the COMPUTATION, not merely the bytes on disk. A merge that
            //    rewrote identical values, or wrote to the wrong initializers, leaves this identical
            //    while `TrainMergeGenerateTest`'s byte hashes still change.
            assertTrue(
                "the merged inference graph computes exactly the same thing as before the merge " +
                    "($before vs $after). Either the merge wrote values the graph does not read, or it " +
                    "re-merged an already-merged package — re-push a pristine one (make device-package).",
                after.differsFrom(before),
            )

            // 2. It must still be a language model. This is the on-device mirror of the host's
            //    train/inference parity gate, using the same causal shift so the numbers are comparable.
            assertTrue(
                "post-merge cross-entropy ${after.crossEntropyNats} is not finite — the merged graph " +
                    "is producing NaN/Inf logits",
                after.crossEntropyNats.isFinite(),
            )
            assertTrue(
                "post-merge cross-entropy ${after.crossEntropyNats} nats exploded relative to the " +
                    "pre-merge ${before.crossEntropyNats} nats (ratio > $MAX_LOSS_RATIO). A merge that " +
                    "lost or corrupted weights lands at or above ln(vocab_size); this is that regime.",
                after.crossEntropyNats <= before.crossEntropyNats * MAX_LOSS_RATIO,
            )
        } finally {
            File(trainDir, "$TRAIN_FILE.jsonl").delete()
            restoreFrom(stash, checkpoint, trainingState)
            stash.deleteRecursively()
        }
    }

    /**
     * **A training run that takes ZERO optimizer steps must leave the model unchanged.**
     *
     * Isolates the training *save* path from the merge. With `gradientAccumulationSteps = 4` (the
     * default) the optimizer only steps on `globalStep % 4 == 0`, so a 3-step run applies **no update
     * at all** — LoRA's `B` stays exactly at its zero initialization, which the merge instrumentation
     * confirms (`adapter_B l2=0.000000`). Nothing about the model should differ afterwards.
     *
     * `mergeAtEnd = false` and no `merge()` call, so the merge is entirely out of the picture: the only
     * thing exercised is training's write-back of trainable parameters into `inference/`.
     *
     * This exists because the investigation on 2026-08-14 spent two rounds blaming the merge for
     * damage that a 3-step run — which trains nothing — reproduced on its own.
     */
    @Test
    fun aTrainingRunThatAppliesNoUpdateLeavesTheModelUnchanged(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val trainDir = File(root, "$repoId/train")
        val trainFile = "mt_zero_update"
        val stash = File(root, "$repoId/.zero_update_stash").apply { mkdirs() }
        stashInto(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))

        File(trainDir, "$trainFile.jsonl").writeText(
            (1..8).joinToString("\n") { i ->
                """{"sentence": "The cat sat on the mat number $i.", "label": ${i % 2}}"""
            } + "\n",
        )

        try {
            val tokenizer = ORTTokenizerNative(PackagePaths.forCache(root, repoId).tokenizer.absolutePath)
            tokenizer.createTokenizerModel()
            val text: IntArray
            try {
                text = tokenizer.tokenize(
                    "The history of the printing press begins in the fifteenth century.",
                    prependBos = true,
                )
            } finally {
                tokenizer.destroySession()
            }

            val before = probeMergedGraph(root, repoId, text).crossEntropyNats

            val model = MobileTransformers.fromPretrained(
                context = ctx,
                repoId = repoId,
                cacheDir = root.absolutePath,
                features = setOf(ModelFeature.Inference, ModelFeature.Training),
            )
            try {
                model.train(
                    DatasetConfig(
                        trainFile = trainFile,
                        task = "cola",
                        maxSequenceLength = 64,
                        maxDatasetLength = 8,
                        datasetBatchSize = 4,
                    ),
                    // 3 steps at the DEFAULT gradientAccumulationSteps = 4 -> zero optimizer steps.
                    TrainConfig(maxSteps = 3, batchSize = 2, mergeAtEnd = false),
                )
            } finally {
                model.close()
            }

            val after = probeMergedGraph(root, repoId, text).crossEntropyNats
            android.util.Log.i(LOG_TAG, "zero-update training: before=$before after=$after")

            val drift = kotlin.math.abs(after - before) / before
            assertTrue(
                "a training run that applied NO optimizer step changed the model: $before -> $after " +
                    "nats (${"%.1f".format(drift * 100)}% drift).\n" +
                    "With gradientAccumulationSteps = 4 and maxSteps = 3 the optimizer never fires and " +
                    "LoRA's B stays at its zero initialization, so there is no update to write. The " +
                    "merge is not involved (mergeAtEnd = false, no merge() call). This is training's " +
                    "write-back of trainable parameters into inference/ corrupting the graph.",
                after.isFinite() && drift < 0.01,
            )
        } finally {
            File(trainDir, "$trainFile.jsonl").delete()
            restoreFrom(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))
            stash.deleteRecursively()
        }
    }

    /**
     * **Merging an UNTRAINED adapter must be the identity.**
     *
     * The sharpest possible test of the merge path, and the one that was missing. LoRA initializes
     * `B = 0`, so before any training `B @ A` is exactly zero and `base + scale * (B @ A) == base` for
     * *every* scale. A merge that changes the model here cannot be blamed on the adapter, the training
     * run, the learning rate, the scale, or the dataset — the only thing left is how the weights are
     * read and written.
     *
     * That makes this a **decisive** discriminator, and it costs no training at all. It exists because
     * two rounds of investigation on 2026-08-14 chased the delta (step count, then the LoRA scale) when
     * the corruption turned out to be independent of the delta's magnitude: a 3-step merge damaged the
     * model exactly as much as a 100-step one.
     *
     * Self-calibrating: it compares the graph against itself, so it encodes no model's numbers.
     */
    @Test
    fun mergingAnUntrainedAdapterLeavesTheModelUnchanged(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val trainDir = File(root, "$repoId/train")
        val stash = File(root, "$repoId/.zero_merge_stash").apply { mkdirs() }
        stashInto(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))

        try {
            val tokenizer = ORTTokenizerNative(PackagePaths.forCache(root, repoId).tokenizer.absolutePath)
            tokenizer.createTokenizerModel()
            val text: IntArray
            try {
                text = tokenizer.tokenize(
                    "The history of the printing press begins in the fifteenth century.",
                    prependBos = true,
                )
            } finally {
                tokenizer.destroySession()
            }

            val before = probeMergedGraph(root, repoId, text).crossEntropyNats

            val model = MobileTransformers.fromPretrained(
                context = ctx,
                repoId = repoId,
                cacheDir = root.absolutePath,
                features = setOf(ModelFeature.Inference, ModelFeature.Training),
            )
            try {
                model.merge() // NO training: the adapter is still at its initialization, B = 0.
            } finally {
                model.close()
            }

            val after = probeMergedGraph(root, repoId, text).crossEntropyNats
            android.util.Log.i(LOG_TAG, "zero-adapter merge: before=$before after=$after")

            val drift = kotlin.math.abs(after - before) / before
            assertTrue(
                "merging an UNTRAINED adapter changed the model: $before -> $after nats " +
                    "(${"%.1f".format(drift * 100)}% drift).\n" +
                    "B is zero at initialization, so the delta is exactly zero and this merge must be " +
                    "the identity at any scale. A change here is not about the adapter, the scale or " +
                    "the training run — it is the weight read/write path itself (layout, dtype, or " +
                    "which initializer is written). Every merged model this SDK has ever produced is " +
                    "affected.",
                after.isFinite() && drift < 0.01,
            )
        } finally {
            restoreFrom(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))
            stash.deleteRecursively()
        }
    }

    /**
     * **Does the merge survive a delta large enough to matter?**
     *
     * Every other merge assertion in this repo trains a near-zero adapter: `TrainMergeGenerateTest`
     * takes **1** step, and the test above takes **3**. A merge that is systematically wrong by a
     * factor or an orientation contributes `scale * B@A`, so at three steps it perturbs the graph by
     * almost nothing and every one of those gates passes. The first run to train hard — 108 steps at
     * 5e-4, in `ToolCallDeviceTest` — produced a model that reached 0.006 training loss and then
     * emitted one repeated token forever. This test exists to decide whether those two facts are the
     * same fact.
     *
     * ### Why it compares two cross-entropies instead of using a threshold
     *
     * Measuring the merged graph on *general* text after a heavy fine-tune cannot answer the question:
     * catastrophic forgetting legitimately raises that number, so a high value proves nothing. Instead
     * this measures the merged graph on **the exact text the model just memorised** and on unrelated
     * text, and compares the two:
     *
     * * merge correct → the model memorised the corpus, so CE(memorised) is far *below* CE(unrelated);
     * * merge wrong → the adapter's contribution is garbage, so the merged graph has learned nothing
     *   and the two are indistinguishable (or both are junk).
     *
     * Relative and self-calibrating, so it neither encodes one model's numbers nor cares which way
     * forgetting moved the absolute values — the failure mode an absolute threshold would have.
     */
    @Test
    fun aLargeAdapterDeltaSurvivesTheMergeIntoTheInferenceGraph(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val trainDir = File(root, "$repoId/train")
        val trainFile = "mt_large_delta"
        val stash = File(root, "$repoId/.large_delta_stash").apply { mkdirs() }
        stashInto(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))

        // One prompt/completion pair, repeated. Memorisation is the point: it is what makes
        // CE(memorised) a meaningful number rather than a measure of general fluency.
        val prompt = "wake me at 07:30"
        val completion = """{"actionName": "set_alarm", "parameters": {"time": "07:30"}}"""
        File(trainDir, "$trainFile.jsonl").writeText(
            (1..STEPS_FOR_LARGE_DELTA * 2).joinToString("\n") {
                org.json.JSONObject().put("prompt", prompt).put("completion", completion).toString()
            } + "\n",
        )

        try {
            val tokenizer = ORTTokenizerNative(PackagePaths.forCache(root, repoId).tokenizer.absolutePath)
            // The constructor reads the configs (so `vocabSize` is set) but does NOT open the native
            // tokenizer session. Calling `tokenize` first passes a 0 handle straight into
            // `tokenizeString`, which dereferences it — a SIGSEGV rather than an error. Noted as a
            // separate fail-open defect in the JNI layer; here we simply open the session first.
            tokenizer.createTokenizerModel()
            val memorised: IntArray
            val unrelated: IntArray
            try {
                memorised = tokenizer.tokenize(prompt + completion, prependBos = true)
                unrelated = tokenizer.tokenize(
                    "The history of the printing press begins in the fifteenth century.",
                    prependBos = true,
                )
            } finally {
                tokenizer.destroySession()
            }
            assertTrue("tokenizer reported no vocabulary size", tokenizer.vocabSize > 0)

            // CONTROL, before anything is trained: is the pristine inference graph a language model at
            // all? Nothing in the repo asserted this. `mergedWeightsChangeTheComputationAndStayNumericallySane`
            // measures a `before` value but only ever compares `after` to it, so a graph that was junk
            // from the start would satisfy every existing bound. If this fails, the merge is innocent
            // and the defect is in the inference graph or the prefill inputs (attention mask, position
            // ids) — a completely different investigation.
            val uniformFloor = kotlin.math.ln(tokenizer.vocabSize.toDouble())
            val baseline = probeMergedGraph(root, repoId, unrelated).crossEntropyNats
            android.util.Log.i(
                LOG_TAG,
                "pristine-graph CE on unrelated text = $baseline nats (uniform floor $uniformFloor)",
            )
            assertTrue(
                "the PRISTINE inference graph scores unrelated English at $baseline nats against a " +
                    "uniform-prediction floor of $uniformFloor — it is not behaving as a language " +
                    "model before any adapter is involved. The merge is not the defect; look at the " +
                    "inference graph and the prefill inputs (attention mask / position ids).",
                baseline.isFinite() && baseline < uniformFloor * 0.75,
            )

            val losses = mutableListOf<Float>()
            val model = MobileTransformers.fromPretrained(
                context = ctx,
                repoId = repoId,
                cacheDir = root.absolutePath,
                features = setOf(ModelFeature.Inference, ModelFeature.Training),
            )
            try {
                model.train(
                    DatasetConfig(
                        trainFile = trainFile,
                        task = "mobile_actions",
                        maxSequenceLength = 160,
                        maxDatasetLength = STEPS_FOR_LARGE_DELTA * 2,
                        datasetBatchSize = 4,
                    ),
                    TrainConfig(
                        maxSteps = STEPS_FOR_LARGE_DELTA,
                        batchSize = 2,
                        learningRate = 5e-4f,
                        gradientAccumulationSteps = 1,
                        mergeAtEnd = true,
                    ),
                    object : TrainCallback {
                        override fun onStepEnd(progress: TrainProgress) { losses.add(progress.stepLoss) }
                    },
                )
            } finally {
                model.close()
            }

            val finalLoss = losses.takeLast(5).average()
            android.util.Log.i(LOG_TAG, "largeDelta steps=${losses.size} finalTrainLoss=$finalLoss")

            // If training itself did not converge, this test cannot say anything about the merge —
            // fail for that reason explicitly rather than blaming the merge for a training problem.
            assertTrue(
                "training did not converge (final loss $finalLoss over ${losses.size} steps), so this " +
                    "run cannot distinguish a bad merge from a bad fine-tune",
                losses.size >= 20 && finalLoss < 0.5,
            )

            val onMemorised = probeMergedGraph(root, repoId, memorised).crossEntropyNats
            val onUnrelated = probeMergedGraph(root, repoId, unrelated).crossEntropyNats
            android.util.Log.i(
                LOG_TAG,
                "merged-graph CE: memorised=$onMemorised unrelated=$onUnrelated " +
                    "trainingLoss=$finalLoss",
            )

            assertTrue("merged-graph cross-entropy is not finite", onMemorised.isFinite())

            // THE assertion. The training graph says this text costs ~$finalLoss nats. If the merged
            // inference graph does not also find it far cheaper than unrelated text, then what the
            // adapter learned did not survive the merge.
            assertTrue(
                "the merged inference graph has NOT learned the text the adapter memorised.\n" +
                    "  training loss on it: $finalLoss nats\n" +
                    "  merged-graph CE on the SAME text: $onMemorised nats\n" +
                    "  merged-graph CE on unrelated text: $onUnrelated nats\n" +
                    "A correctly merged adapter makes memorised text far cheaper than unrelated text. " +
                    "These being comparable means the delta written into the inference graph is not " +
                    "the delta that was trained — i.e. the merge is numerically wrong in a way that " +
                    "1- and 3-step merges are too small to reveal.",
                onMemorised < onUnrelated * 0.5,
            )
        } finally {
            File(trainDir, "$trainFile.jsonl").delete()
            restoreFrom(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))
            stash.deleteRecursively()
        }
    }

    /**
     * Opens the Native engine over the merged `inference/` directory and measures one prefill pass over
     * [tokens].
     *
     * Built directly rather than through the facade because the facade exposes text, not logits, and a
     * text-level comparison cannot distinguish "the merge is numerically wrong" from "greedy decoding
     * happened to pick the same eight tokens" — the exact ambiguity `TrainMergeGenerateTest` documents.
     */
    private suspend fun probeMergedGraph(
        root: File,
        repoId: String,
        tokens: IntArray = PROBE,
    ): ORTGeneratorNative.InferenceMetrics {
        val paths = PackagePaths.forCache(root, repoId)
        val tokenizer = ORTTokenizerNative(paths.tokenizer.absolutePath)
        // The graph filename must be resolved, not assumed: leaving `onnxName` empty makes
        // `createInferenceModel` open `inference/.onnx`, which does not exist. The repository resolves
        // it the same way (`LLMRepository.resolveInferenceGraphName`) — `model.onnx` when present, and
        // otherwise the single `.onnx` in the directory.
        val graphName = File(paths.inference, "model.onnx").let { canonical ->
            if (canonical.isFile) canonical.name
            else paths.inference.listFiles { f: File -> f.isFile && f.name.endsWith(".onnx") }
                .orEmpty().singleOrNull()?.name ?: "model.onnx"
        }
        val config = ORTGenerationConfig(
            repoName = repoId,
            onnxName = graphName,
            loadMergedWeights = true,
        )
        val generator = ORTGeneratorNative(root.absolutePath, tokenizer, config)
        return try {
            generator.load(root.absolutePath, config)
            assertTrue("tokenizer reported no vocabulary size", tokenizer.vocabSize > 0)
            generator.inferenceMetrics(tokens, tokenizer.vocabSize)
        } finally {
            generator.release()
        }
    }

    private fun stashInto(stash: File, vararg files: File) {
        for (f in files) {
            if (f.exists()) f.copyRecursively(File(stash, f.name), overwrite = true)
        }
    }

    private fun restoreFrom(stash: File, vararg files: File) {
        for (f in files) {
            val saved = File(stash, f.name)
            if (saved.exists()) {
                if (f.exists()) f.deleteRecursively()
                saved.copyRecursively(f, overwrite = true)
            }
        }
    }
}

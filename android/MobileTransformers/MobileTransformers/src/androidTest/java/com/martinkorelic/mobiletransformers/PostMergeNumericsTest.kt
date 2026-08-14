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
import org.junit.Rule
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
 * ## Package mutation — handled by [PristinePackageRule], no longer the caller's problem
 *
 * Every test here trains and merges, which rewrites `train/checkpoint`, `training_state.json` and the
 * per-tensor weight blobs under `inference/` (spelled without a leading slash-star on purpose: Kotlin
 * block comments NEST, so `/`+`*` inside KDoc opens a nested comment and eats the closing delimiter).
 *
 * **This KDoc previously said those weight blobs are NOT restored and that every test here needs a
 * freshly pushed package.** That was true when written and is now false: [PristinePackageRule]
 * captures and restores them around each test, including when the test fails. The per-test
 * `stashInto`/`restoreFrom` calls below are the older, narrower mechanism and are kept as an inner
 * belt — they are redundant with the rule, not in conflict with it.
 *
 * The cost of the old position was measured on 2026-08-14: a full-suite run reported **3 failures**
 * that were all this contamination, including `TrainConvergenceTest` seeing `NaN` losses because this
 * class had memorised one sentence to a training loss of 0.0028 before it ran. All three pass against
 * a pristine package. Restoring the fixture is cheaper than a suite whose results cannot be read.
 */
@RunWith(AndroidJUnit4::class)
class PostMergeNumericsTest {

    /**
     * Restore the package after every test in this class. It trains and/or merges, which rewrites the
     * checkpoint and the `inference/` weight blobs in place — see [PristinePackageRule] for the three
     * suite failures this prevents.
     */
    @get:Rule
    val pristinePackage = PristinePackageRule()

    private companion object {
        /**
         * Ceiling on the post-merge cross-entropy, as a fraction of the pre-merge value.
         *
         * Relative, and generous in the direction that matters: a few LoRA steps may move the loss
         * either way by a little, but a merge that corrupted the weights lands at or above the
         * uniform-prediction floor `ln(vocab_size)` — ~10.8 here against a healthy ~4.65 on
         * [PROBE_TEXT], i.e. multiples away rather than percents.
         *
         * This bound only means something because the probe is REAL text. Against the arbitrary token
         * ids it used to score, a healthy model already sat at ~10.19 nats — at chance — so "within 2x"
         * held no matter what the merge did.
         */
        const val MAX_LOSS_RATIO = 2.0

        /**
         * Text the probes score. **Real English, tokenized by the package's own tokenizer** — not a
         * hand-written array of token ids.
         *
         * This used to be `intArrayOf(1, 338, 263, 1243, 310, 278, 1904, 29889)`: ids chosen to be
         * "inside every supported vocabulary", which for this tokenizer spell nothing. A healthy model
         * scores them at ~10.19 nats against a 10.80 uniform floor, i.e. at chance — so the ratio bound
         * below was comparing two near-uniform numbers and could not fail. That is one of the four
         * reasons the transposed-merge defect survived for months (2026-08-14).
         */
        const val PROBE_TEXT = "The history of the printing press begins in the fifteenth century."

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
            val probeTokens = tokenizeWithPackage(root, repoId, PROBE_TEXT)
            val before = probeMergedGraph(root, repoId, probeTokens)

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

            val after = probeMergedGraph(root, repoId, probeTokens)

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
            val text = tokenizeWithPackage(root, repoId, PROBE_TEXT)

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
     * That makes this a **decisive** discriminator. It exists because two rounds of investigation on
     * 2026-08-14 chased the delta (step count, then the LoRA scale) when the corruption turned out to
     * be independent of the delta's magnitude: a 3-step merge damaged the model exactly as much as a
     * 100-step one.
     *
     * **The merge must be FORCED to run, and this test must prove that it did.** As first written it
     * called `merge()` on a freshly loaded model and asserted nothing changed — which passed, but
     * vacuously: `merge()` with no prior training in the same session never executes a merge at all,
     * so it was comparing the package with itself. It was reported as a passing control before the
     * logs were checked, and that wrong control cost a round of investigation.
     *
     * The fix is a real training run whose optimizer never fires (`maxSteps = 3` at the default
     * `gradientAccumulationSteps = 4`), with `mergeAtEnd = true`. The merge then genuinely runs while
     * `B` is still at its zero initialization, so the delta is exactly zero — and the `.bin`
     * modification times are checked to confirm the write path actually executed. **Without that
     * check this test silently reverts to measuring nothing.**
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
        val trainFile = "mt_zero_merge"
        val stash = File(root, "$repoId/.zero_merge_stash").apply { mkdirs() }
        stashInto(stash, File(trainDir, "checkpoint"), File(trainDir, "training_state.json"))

        File(trainDir, "$trainFile.jsonl").writeText(
            (1..8).joinToString("\n") { i ->
                """{"sentence": "The cat sat on the mat number $i.", "label": ${i % 2}}"""
            } + "\n",
        )

        try {
            val text = tokenizeWithPackage(root, repoId, PROBE_TEXT)

            val before = probeMergedGraph(root, repoId, text).crossEntropyNats

            // Modification times of the tensors the merge writes. A merge that runs rewrites these
            // atomically (temp + rename), so mtime moves even though a zero delta leaves the BYTES
            // identical -- which is why bytes cannot be used to prove the merge happened here.
            val weights = File(root, "$repoId/inference")
                .listFiles { f -> f.name.endsWith(".MatMul.weight.bin") }
                .orEmpty()
            assertTrue("package has no per-tensor trainable .bin files", weights.isNotEmpty())
            val mtimesBefore = weights.associate { it.name to it.lastModified() }

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
                    // 3 steps at the DEFAULT gradientAccumulationSteps = 4 -> the optimizer never
                    // fires, so B stays at zero and the delta this merge applies is exactly zero.
                    TrainConfig(maxSteps = 3, batchSize = 2, mergeAtEnd = true),
                )
                model.merge()
            } finally {
                model.close()
            }

            val rewritten = weights.count { it.lastModified() != mtimesBefore[it.name] }
            assertTrue(
                "no merge actually ran: all ${weights.size} trainable .bin files have their original " +
                    "modification time, so this test compared the package with itself and proved " +
                    "NOTHING. That is exactly how the original version of this test passed while the " +
                    "merge was corrupting every weight. Check logcat for 'Starting weight merging " +
                    "process' before trusting any result from this class.",
                rewritten > 0,
            )

            val after = probeMergedGraph(root, repoId, text).crossEntropyNats
            android.util.Log.i(
                LOG_TAG,
                "zero-adapter merge: before=$before after=$after ($rewritten/${weights.size} rewritten)",
            )

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
            File(trainDir, "$trainFile.jsonl").delete()
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
            val memorised = tokenizeWithPackage(root, repoId, prompt + completion)
            val unrelated = tokenizeWithPackage(root, repoId, PROBE_TEXT)

            // CONTROL, before anything is trained: is the pristine inference graph a language model at
            // all? Nothing in the repo asserted this. `mergedWeightsChangeTheComputationAndStayNumericallySane`
            // measures a `before` value but only ever compares `after` to it, so a graph that was junk
            // from the start would satisfy every existing bound. If this fails, the merge is innocent
            // and the defect is in the inference graph or the prefill inputs (attention mask, position
            // ids) — a completely different investigation.
            val uniformFloor = kotlin.math.ln(vocabSizeOf(root, repoId).toDouble())
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
    /** The package's declared vocabulary size, for the `ln(vocab)` uniform-prediction floor. */
    private fun vocabSizeOf(root: File, repoId: String): Int {
        val tokenizer = ORTTokenizerNative(PackagePaths.forCache(root, repoId).tokenizer.absolutePath)
        assertTrue("tokenizer reported no vocabulary size", tokenizer.vocabSize > 0)
        return tokenizer.vocabSize
    }

    /** Tokenize [text] with the package's tokenizer, opening and closing the native session. */
    private suspend fun tokenizeWithPackage(root: File, repoId: String, text: String): IntArray {
        val tokenizer = ORTTokenizerNative(PackagePaths.forCache(root, repoId).tokenizer.absolutePath)
        tokenizer.createTokenizerModel()
        return try {
            tokenizer.tokenize(text, prependBos = true)
        } finally {
            tokenizer.destroySession()
        }
    }

    private suspend fun probeMergedGraph(
        root: File,
        repoId: String,
        tokens: IntArray,
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

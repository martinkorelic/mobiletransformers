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
 * ## Package-mutation hazard
 *
 * This test trains, so it stashes and restores `train/checkpoint` + `training_state.json` and deletes
 * its own dataset fixture, exactly as `ScheduledTrainingDeviceTest` does. Without that it would turn
 * the two training suites red by leaving a merged, already-trained package behind.
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
    }

    @Test
    fun mergedWeightsChangeTheComputationAndStayNumericallySane(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
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
     * Opens the Native engine over the merged `inference/` directory and measures one prefill pass.
     *
     * Built directly rather than through the facade because the facade exposes text, not logits, and a
     * text-level comparison cannot distinguish "the merge is numerically wrong" from "greedy decoding
     * happened to pick the same eight tokens" — the exact ambiguity `TrainMergeGenerateTest` documents.
     */
    private suspend fun probeMergedGraph(root: File, repoId: String): ORTGeneratorNative.InferenceMetrics {
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
            generator.inferenceMetrics(PROBE, tokenizer.vocabSize)
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

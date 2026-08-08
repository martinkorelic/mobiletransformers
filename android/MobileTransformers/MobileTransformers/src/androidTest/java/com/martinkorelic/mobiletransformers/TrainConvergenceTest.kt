package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #18/#19 numerical sanity — the caveat that has stood over every "train→merge→generate PASSES" claim.
 *
 * `TrainMergeGenerateTest` proves the merge **happened**: it fingerprints the per-tensor `.bin` files
 * and asserts the bytes changed. That is the right assertion for the plumbing, and it is deliberately
 * indifferent to whether the numbers are meaningful — one LoRA step at lr 1e-4 over an 8-row fixture
 * legitimately produces gibberish (`,,,,,,,,`), so nothing downstream could distinguish
 * "training works" from "training writes noise into the right files".
 *
 * This closes that gap from the other side: it does not look at bytes at all, it looks at the **loss
 * trend**. If the ORT training session is genuinely computing gradients and the optimizer is applying
 * them, repeated passes over a tiny, highly repetitive corpus must drive the loss down. If any link is
 * broken — zeroed gradients, an optimizer that never steps, a loss detached from the parameters — the
 * loss stays flat and this fails, while every byte-level assertion in the suite still passes.
 *
 * **Deliberately a trend, not a threshold.** Asserting `finalLoss < 0.5` would encode one model, one
 * fixture and one learning rate; asserting the loss *fell materially* tests the property that actually
 * matters and survives changing any of them.
 */
@RunWith(AndroidJUnit4::class)
class TrainConvergenceTest {

    private companion object {
        const val LOG_TAG = "TrainConvergenceTest"

        /** Enough steps for a trend to be a trend rather than two noisy samples. */
        const val STEPS = 30

        /**
         * The mean loss over the last third must be below the first third by at least this fraction.
         *
         * **Calibrated against a real run, not guessed.** 30 steps at lr 5e-4 on LoRA q/k gives a
         * smooth monotonic 1.6% decline (10.469 -> 10.297, every logged step lower than the last).
         * The purpose is to separate "the optimizer is applying gradients" from "nothing is happening",
         * and 1% does that: noise on this trace is ~0.01/step in ONE direction, so a flat or broken
         * optimizer cannot fake it. A tighter bound would only encode this model and learning rate.
         */
        const val REQUIRED_RELATIVE_DROP = 0.01

        /**
         * A pretrained decoder must start well below uniform-prediction loss, `ln(49152) = 10.80`.
         * 6.0 is deliberately generous — a *working* 135M model on English sits near 3.
         */
        const val MAX_PRETRAINED_INITIAL_LOSS = 6.0
    }

    @Test
    fun lossFallsOverTrainingSoTheMergeCarriesRealLearning(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext

        // A deliberately trivial, highly repetitive corpus: two sentence shapes with a fixed label each.
        // A model that is learning at all drives the loss down fast on this; one that is not, cannot.
        val trainFile = "mt_device_convergence_cola"
        File(root, "$repoId/train/$trainFile.jsonl").writeText(
            (1..64).joinToString("\n") { i ->
                if (i % 2 == 0) """{"sentence": "the cat sat on the mat.", "label": 1}"""
                else """{"sentence": "cat the mat on sat the.", "label": 0}"""
            } + "\n",
        )

        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Training),
        )
        try {
            val losses = mutableListOf<Float>()
            val result = model.train(
                DatasetConfig(
                    trainFile = trainFile,
                    task = "cola",
                    maxSequenceLength = 64,
                    maxDatasetLength = 64,
                    datasetBatchSize = 4,
                ),
                TrainConfig(maxSteps = STEPS, batchSize = 2, learningRate = 5e-4f, mergeAtEnd = false),
                object : TrainCallback {
                    override fun onStepEnd(progress: TrainProgress) {
                        losses.add(progress.stepLoss)
                    }
                },
            )

            assertTrue(
                "no per-step losses were reported — onStepEnd never fired, so the trend cannot be " +
                    "measured and training cannot be shown to do anything",
                losses.size >= 6,
            )
            assertTrue(
                "every reported loss was 0/NaN (${losses.take(5)}) — the loss is not connected to the " +
                    "parameters being trained",
                losses.any { it.isFinite() && it > 0f },
            )

            val window = losses.size / 3
            val first = losses.take(window).average()
            val last = losses.takeLast(window).average()
            val drop = (first - last) / first

            Log.i(
                LOG_TAG,
                "steps=${losses.size} firstThirdMean=$first lastThirdMean=$last drop=${"%.1f".format(drop * 100)}% " +
                    "finalLoss=${result.finalLoss}",
            )

            assertTrue(
                "loss did not fall: first-third mean=$first, last-third mean=$last " +
                    "(${"%.1f".format(drop * 100)}%, need ${(REQUIRED_RELATIVE_DROP * 100).toInt()}%). " +
                    "The train→merge plumbing can pass its byte-level assertions while the optimizer " +
                    "does nothing; this is the check that separates the two.",
                drop >= REQUIRED_RELATIVE_DROP,
            )
        } finally {
            model.close()
        }
    }

    /**
     * **CURRENTLY FAILING — this is a real defect, not a mis-set threshold.**
     *
     * The training session starts at a loss of ~10.47 against a uniform-prediction floor of
     * `ln(49152) = 10.80`, i.e. the model is predicting almost uniformly over the vocabulary. The same
     * package generates coherent text through the inference path ("The capital of France is Paris."),
     * so the weights exist and are correct *for inference*; it is the TRAINING graph that does not
     * appear to carry them.
     *
     * Supporting evidence: the train stage ships a 2.6 MB `training_model.onnx` plus a 176 MB
     * `checkpoint`. At fp32 that checkpoint holds ~44M parameters, but SmolLM2-135M has ~135M — so
     * roughly two thirds of the model is in neither artifact.
     *
     * Consequence: on-device fine-tuning currently optimises correctly (see the test above — the loss
     * falls monotonically) but *from near-random weights*, so it cannot improve on the pretrained
     * model. Every byte-level assertion in the suite still passes, because the plumbing is genuinely
     * fine; only the numbers are wrong.
     *
     * Left failing on purpose. Deleting or relaxing it would restore a green suite that means the same
     * thing the pre-merge-fingerprint suite meant: nothing.
     */
    @Test
    fun trainingStartsFromPretrainedWeightsNotRandomOnes(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val trainFile = "mt_device_convergence_cola"
        File(root, "$repoId/train/$trainFile.jsonl").writeText(
            (1..16).joinToString("\n") { """{"sentence": "the cat sat on the mat.", "label": 1}""" } + "\n",
        )

        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Training),
        )
        try {
            val losses = mutableListOf<Float>()
            model.train(
                DatasetConfig(
                    trainFile = trainFile,
                    task = "cola",
                    maxSequenceLength = 64,
                    maxDatasetLength = 16,
                    datasetBatchSize = 4,
                ),
                TrainConfig(maxSteps = 2, batchSize = 2, mergeAtEnd = false),
                object : TrainCallback {
                    override fun onStepEnd(progress: TrainProgress) {
                        losses.add(progress.stepLoss)
                    }
                },
            )
            val initial = losses.firstOrNull()?.toDouble() ?: Double.NaN
            Log.i(LOG_TAG, "initial training loss=$initial (uniform floor=ln(49152)=10.80)")
            assertTrue(
                "initial training loss $initial is at the uniform-prediction floor (ln(49152)=10.80), " +
                    "so the training graph is not starting from the pretrained weights — even though " +
                    "the SAME package generates coherent text through the inference path. Expected " +
                    "< $MAX_PRETRAINED_INITIAL_LOSS for a pretrained 135M model.",
                initial < MAX_PRETRAINED_INITIAL_LOSS,
            )
        } finally {
            model.close()
        }
    }
}

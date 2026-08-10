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
         * How much better a pretrained model must score coherent English than token soup, as a
         * fraction of its loss on the soup.
         *
         * **Self-calibrating by construction.** Both corpora go through the same preprocessor, the
         * same tokenizer and the same graph; the ONLY difference is whether the supervised target is
         * natural language. A model holding pretrained weights separates these by several nats; a
         * randomly-initialised one cannot separate them at all, because neither is more probable than
         * the other under an untrained distribution. 15% is far below the gap a working model shows
         * and far above anything noise produces, and it encodes no model, tokenizer or fixture.
         */
        const val REQUIRED_COHERENCE_MARGIN = 0.15
    }

    @Test
    fun lossFallsOverTrainingSoTheMergeCarriesRealLearning(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
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
     * The training graph starts from the **pretrained** weights, not random ones.
     *
     * ### Why this is a comparison and not a threshold
     *
     * This test previously asserted `initialLoss < 6.0`, reasoning that a pretrained 135M model on
     * English sits near 3 and the uniform-prediction floor is `ln(49152) = 10.80`. It failed at ~14.25
     * and was recorded as a v1 blocker: "two thirds of the model is in neither artifact", from reading
     * the 176 MB checkpoint as `176MB / 4 bytes ≈ 44M` fp32 parameters.
     *
     * **Both halves of that were wrong.** ~90% of the checkpoint tensors are uint8, not fp32; the
     * training graph carries all 135,436,911 parameters (verified against the shipped artifact, and
     * now gated on the host by `artifacts/parameter_budget.py`). And the threshold measured something
     * other than what it claimed: `ORTDataCurator` masks every prompt token to `-100`, and under
     * `CoLAPreprocessor` the answer `"acceptable"` is a **single token** following a trailing-space
     * prompt — so "initial loss" was the cross-entropy of one improbable token, not a sequence LM
     * loss. ~14 is the correct value for that, and it reproduces from the fp32 *inference* graph.
     *
     * An absolute bound cannot distinguish "not pretrained" from "this particular answer string is
     * unlikely". So this asserts the property that actually implies pretrained weights and needs no
     * model-specific constant: **coherent English must cost materially less than token soup.** Only a
     * model carrying real weights can tell them apart; an untrained one scores both alike, because
     * under a near-uniform distribution neither is more probable.
     *
     * The two corpora are matched deliberately — same preprocessor, same field, same approximate token
     * length — so the only variable is coherence.
     */
    @Test
    fun trainingStartsFromPretrainedWeightsNotRandomOnes(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext

        // The SUPERVISED text, not the prompt: `ORTDataCurator` masks prompt tokens to -100, so only
        // this side reaches the loss. `mini_recommendation` supervises its `recommendation` field
        // verbatim, which is why this test uses it rather than `cola` (whose target is one of two
        // fixed words chosen by `label`, and so cannot be varied at all).
        val coherent = "open the calendar app and create a meeting for tomorrow morning at nine"
        val soup = "gzt qwx vbn plok zrf mjud xqi wbe trng kvo phz lmq brx qynt"

        val coherentLoss = initialLossFor(ctx, root, repoId, "mt_device_coherent", coherent)
        val soupLoss = initialLossFor(ctx, root, repoId, "mt_device_soup", soup)
        val margin = (soupLoss - coherentLoss) / soupLoss

        Log.i(
            LOG_TAG,
            "initial loss: coherent=$coherentLoss soup=$soupLoss " +
                "margin=${"%.1f".format(margin * 100)}%",
        )

        assertTrue(
            "both initial losses must be finite and positive (coherent=$coherentLoss soup=$soupLoss) " +
                "— otherwise the loss is not connected to the parameters and neither number means " +
                "anything",
            coherentLoss.isFinite() && soupLoss.isFinite() && coherentLoss > 0.0 && soupLoss > 0.0,
        )
        assertTrue(
            "the training graph scores coherent English (loss=$coherentLoss) no better than random " +
                "token soup (loss=$soupLoss) — margin ${"%.1f".format(margin * 100)}%, need " +
                "${(REQUIRED_COHERENCE_MARGIN * 100).toInt()}%. A model holding its pretrained " +
                "weights separates these by several nats; one starting from random weights cannot " +
                "separate them at all. Check the training-stage export: the host-side parameter " +
                "budget and train/inference parity gates in artifacts/ should have caught this first.",
            margin >= REQUIRED_COHERENCE_MARGIN,
        )
    }

    /**
     * Initial (step-0) training loss over a corpus of one repeated sentence.
     *
     * A fresh model per corpus: `mergeAtEnd = false` leaves the on-disk checkpoint untouched, so each
     * call genuinely starts from the packaged weights rather than from the previous call's updates.
     */
    private suspend fun initialLossFor(
        ctx: android.content.Context,
        root: File,
        repoId: String,
        trainFile: String,
        supervisedText: String,
    ): Double {
        // Identical prompt in both corpora so the only variable is the supervised continuation.
        File(root, "$repoId/train/$trainFile.jsonl").writeText(
            (1..16).joinToString("\n") {
                """{"prompt": "what should I do next?", "recommendation": "$supervisedText"}"""
            } + "\n",
        )

        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Training),
        )
        return try {
            val losses = mutableListOf<Float>()
            model.train(
                DatasetConfig(
                    trainFile = trainFile,
                    task = "mini_recommendation",
                    maxSequenceLength = 64,
                    maxDatasetLength = 16,
                    datasetBatchSize = 4,
                ),
                TrainConfig(maxSteps = 1, batchSize = 2, mergeAtEnd = false),
                object : TrainCallback {
                    override fun onStepEnd(progress: TrainProgress) {
                        losses.add(progress.stepLoss)
                    }
                },
            )
            losses.firstOrNull()?.toDouble() ?: Double.NaN
        } finally {
            model.close()
        }
    }
}

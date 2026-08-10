package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.federated.NativeCheckpointTensorStore
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import com.martinkorelic.mobiletransformers.repository.TrainingCallback
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #33 device leg: a real training step on an **encoder** package, on hardware.
 *
 * Provision with:
 * ```
 * make device-package MODEL=sentence-transformers/all-MiniLM-L6-v2 TASK=text-classification TRAIN=1 RAG=0
 * ```
 *
 * ## What makes this different from the decoder train suites
 *
 * The objective, and therefore the data shape. A classification head supervises **one label per
 * sequence** (`labels[batch]`), not one per token, so this drives the `TaskPreprocessor.classLabel` →
 * `perSequenceLabel` → unpadded-collation path that no decoder suite touches. There is no `generate()`
 * here at all: an encoder has no token loop, and asserting on generated text would be asserting the
 * wrong contract (the decoder suites now skip on an encoder package for the same reason —
 * `DeviceModel.requireDecoder`).
 *
 * ## What is asserted
 *
 * That the step **moved the adapter**, read back from the ORT checkpoint by name — not merely that
 * training returned without throwing. A run that loads the graph, computes a loss and applies nothing
 * looks identical from the outside, and that is not hypothetical: the #36 round-trip found exactly
 * that shape, where `gradAccumSteps` defaulting to 4 made a short run accumulate gradients it never
 * applied.
 *
 * Hermetic: trains with `saveModelAtEnd = false` / `mergeWeightsAtEnd = false`, so neither the
 * checkpoint nor the inference weights on disk are touched.
 */
@RunWith(AndroidJUnit4::class)
class EncoderTrainStepDeviceTest {

    @Test
    fun oneTrainStepMovesTheEncodersAdapter(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        val task = DeviceModel.selectedTask(root, repoId)
        assumeTrue(
            "installed package '$repoId' was exported for task '$task'; this suite needs an encoder " +
                "package (make device-package TASK=text-classification TRAIN=1)",
            task == "text-classification",
        )
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val paths = PackagePaths.forCache(root.absolutePath, repoId)

        // A separable two-class set, the same fixture shape the host gate uses. `cola_cls` is the
        // preprocessor that keeps the label a CLASS INDEX instead of stringifying it for a decoder.
        val trainFile = "mt_encoder_cls"
        File(paths.train, "$trainFile.jsonl").writeText(
            listOf(
                """{"sentence": "this film was wonderful", "label": 1}""",
                """{"sentence": "an absolute delight to watch", "label": 1}""",
                """{"sentence": "brilliant and moving", "label": 1}""",
                """{"sentence": "a masterpiece of storytelling", "label": 1}""",
                """{"sentence": "this film was terrible", "label": 0}""",
                """{"sentence": "a complete waste of time", "label": 0}""",
                """{"sentence": "boring and painfully dull", "label": 0}""",
                """{"sentence": "an awful, incoherent mess", "label": 0}""",
            ).joinToString("\n") + "\n",
        )

        val tokenizer = ORTTokenizerNative(paths.tokenizer.absolutePath)
        tokenizer.createTokenizerModel()
        val trainer = ORTTrainerNative(
            ctx,
            root.absolutePath,
            tokenizer,
            ORTTrainingConfig(
                repoName = repoId,
                taskName = "cola_cls",
                batchSize = 2,
                maxSteps = 2,
                // See the #36 round-trip: `optimizerStep` fires on `globalStep % gradAccumSteps == 0`
                // and never at step 0, so the shipping default of 4 would apply nothing in 2 steps.
                gradAccumSteps = 1,
                // Hermetic: leave the package exactly as pushed.
                mergeWeightsAtEnd = false,
                saveModelAtEnd = false,
                loadFromState = false,
                keepSessionAtEnd = true,
                datasetOptions = DatasetOptions(
                    trainFile = trainFile,
                    datasetBatchSize = 2,
                    maxDatasetLength = 8,
                    maxSequenceLength = 64,
                ),
            ),
        )

        try {
            assertTrue(
                "no native training session was created for the encoder package",
                trainer.trainingSessionHandle() != 0L,
            )

            // The trainable factors, by their checkpoint names, straight from the package's own
            // handoff map — nothing here re-derives a layer identity.
            val handoff = WeightHandoffMap.load(paths.weightHandoff)
            val specs = handoff.adapterTensorSpecs()
            assertTrue("the encoder package declares no adapter factors", specs.isNotEmpty())

            val store = NativeCheckpointTensorStore(trainer)
            val before = specs.associate { it.name to (store.read(it.name) ?: ByteArray(0)) }
            assertTrue(
                "no adapter factor could be read from the encoder checkpoint: " +
                    "${specs.take(2).map { it.name }}",
                before.values.all { it.isNotEmpty() },
            )

            val losses = mutableListOf<Float>()
            trainer.startTraining(object : TrainingCallback {
                override fun onStepEnd(trainingProgress: TrainingProgress) {
                    losses += trainingProgress.stepLoss
                }
            })

            assertTrue("no training step ran", losses.isNotEmpty())
            assertTrue(
                "training produced a non-finite loss: $losses",
                losses.all { it.isFinite() },
            )

            val moved = specs.count { spec ->
                val after = store.read(spec.name)
                after != null && !after.contentEquals(before.getValue(spec.name))
            }
            assertTrue(
                "the train step moved none of the ${specs.size} adapter factors — the graph loaded " +
                    "and a loss was computed, but no update was applied",
                moved > 0,
            )
            Log.i(
                TAG,
                "encoder train step: ${losses.size} step(s), losses=$losses, " +
                    "$moved/${specs.size} adapter factors moved",
            )
        } finally {
            trainer.destroySession(false)
            File(paths.train, "$trainFile.jsonl").delete()
        }
    }

    private companion object {
        const val TAG = "EncoderTrainStepDeviceTest"
    }
}

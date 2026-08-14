package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.agent.ActionSpec
import com.martinkorelic.mobiletransformers.agent.FunctionCallValidator
import com.martinkorelic.mobiletransformers.agent.ToolCallResult
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.SamplingConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * **#37 CHECKPOINT — the differentiation gate, end to end on hardware.**
 *
 * per-user action set → on-device fine-tune → validated tool call → dry-run intent, in one run, with
 * nothing leaving the device.
 *
 * ### What this asserts, and why it is the accepted path
 *
 * A demo that "passes" by exercising the **refusal** path would show nothing: the validator rejecting
 * garbage is already covered by 15 JVM tests, and an untrained model refusing is the trivial outcome.
 * The claim worth making is the hard one — that a model fine-tuned **here, on this user's own action
 * vocabulary** emits a call its own app accepts, which then becomes a real Android intent it was never
 * allowed to name.
 *
 * ### Why the corpus is deliberately tiny and repetitive
 *
 * The assertion is convergence-dependent, and this runs on a 135M base with a LoRA adapter in the
 * minutes available to an instrumented test. So the target is **memorisation, not generalisation**:
 * three actions, one prompt shape each, repeated. That is enough to demonstrate the loop, and claiming
 * more from it would overstate what a run this size can show. Generalisation is what the imported
 * `google/mobile-actions` corpus is for (`mobiletransformers agent-dataset`), which needs a longer run
 * than an instrumented test should hold.
 *
 * ### The two settings that decide whether this can work at all
 *
 * * **`gradientAccumulationSteps = 1`.** `optimizerStep` fires on `globalStep % gradAccumSteps == 0`
 *   and the default is 4, so a bounded run can complete, report success on every callback, and apply
 *   **no update whatsoever**. That defect cost a cycle to find in #36; it is pinned here on purpose.
 * * **Greedy decoding.** Sampling would make the assertion flaky for reasons unrelated to whether the
 *   model learned anything — a temperature that occasionally breaks JSON is not a finding.
 *
 * A failure here is a real result to record, not something to loosen the assertion for.
 */
@RunWith(AndroidJUnit4::class)
class ToolCallDeviceTest {

    private companion object {
        const val LOG_TAG = "ToolCallDeviceTest"

        /** Long enough to memorise three phrasings; short enough for an instrumented run. */
        const val STEPS = 120
    }

    /**
     * The app's declaration — the ONLY source of intent strings, and the same object the dataset is
     * generated from. `mobiletransformers agent-dataset` writes this as `action_schema.json`; the test
     * holds it inline so the corpus and the boundary provably come from one value.
     */
    private val allowlist = listOf(
        ActionSpec(
            actionName = "set_alarm",
            parameters = mapOf("time" to "string"),
            allowedIntent = "android.intent.action.SET_ALARM",
            validationRules = mapOf("time" to "HH:mm"),
            privacyClass = "harmless-demo",
        ),
        ActionSpec(
            actionName = "set_timer",
            parameters = mapOf("seconds" to "string"),
            allowedIntent = "android.intent.action.SET_TIMER",
            validationRules = mapOf("seconds" to "/[0-9]{1,4}/"),
            privacyClass = "harmless-demo",
        ),
        ActionSpec(
            actionName = "open_wifi_settings",
            parameters = emptyMap(),
            allowedIntent = "android.settings.WIFI_SETTINGS",
            privacyClass = "harmless-demo",
        ),
    )

    /** `{"prompt","completion"}` rows — the shape `MobileActionsPreprocessor` (task `mobile_actions`) reads. */
    private fun corpus(): String {
        val rows = mutableListOf<Pair<String, String>>()
        for (time in listOf("06:15", "07:30", "08:00", "22:05")) {
            rows += "wake me at $time" to
                """{"actionName": "set_alarm", "parameters": {"time": "$time"}}"""
        }
        for (seconds in listOf("30", "60", "300", "900")) {
            rows += "timer for $seconds seconds" to
                """{"actionName": "set_timer", "parameters": {"seconds": "$seconds"}}"""
        }
        rows += "open wifi settings" to """{"actionName": "open_wifi_settings", "parameters": {}}"""

        // Repeated so a bounded number of steps sees each pair many times.
        return (1..12).flatMap { rows }.joinToString("\n") { (prompt, completion) ->
            org.json.JSONObject()
                .put("prompt", prompt)
                .put("completion", completion)
                .toString()
        } + "\n"
    }

    @Test
    fun aLocallyFineTunedModelEmitsACallTheAppAcceptsAndBindsToAnIntent(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val trainFile = "mt_device_mobile_actions"
        File(root, "$repoId/train/$trainFile.jsonl").writeText(corpus())

        val validator = FunctionCallValidator(allowlist)
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
                    task = "mobile_actions",
                    maxSequenceLength = 96,
                    maxDatasetLength = 128,
                    datasetBatchSize = 4,
                ),
                TrainConfig(
                    maxSteps = STEPS,
                    batchSize = 2,
                    learningRate = 5e-4f,
                    // See the class docstring: 4 (the default) means a bounded run trains nothing.
                    gradientAccumulationSteps = 1,
                    mergeAtEnd = true,
                ),
                object : TrainCallback {
                    override fun onStepEnd(progress: TrainProgress) {
                        losses.add(progress.stepLoss)
                    }
                },
            )

            assertTrue(
                "no per-step losses were reported — training cannot be shown to have done anything",
                losses.size >= 6,
            )
            val window = losses.size / 3
            val drop = (losses.take(window).average() - losses.takeLast(window).average()) /
                losses.take(window).average()
            Log.i(LOG_TAG, "steps=${losses.size} lossDrop=${"%.1f".format(drop * 100)}%")

            // The instruction is one the corpus taught, because memorisation is the claim being made.
            val instruction = "wake me at 07:30"
            val result = model.generateToolCall(
                instruction = instruction,
                validator = validator,
                config = GenerationConfig(
                    maxNewTokens = 48,
                    // Greedy: a flaky sample is not a finding about whether the model learned.
                    sampling = SamplingConfig(method = SamplingMethod.GREEDY),
                    loadMerged = true,
                ),
            )
            Log.i(LOG_TAG, "instruction='$instruction' raw='${result.raw}' -> ${result::class.simpleName}")

            assertTrue(
                "the model did not emit a call this app accepts after $STEPS steps on its own action " +
                    "set.\n  raw output: '${result.raw}'\n  reason: " +
                    (result as? ToolCallResult.Rejected)?.reason +
                    "\n  loss drop over training: ${"%.1f".format(drop * 100)}%\n" +
                    "This is the #37 differentiation gate. A refusal here is a real result — record it " +
                    "rather than weakening the assertion; asserting on the refusal path would prove " +
                    "nothing, since an untrained model also refuses.",
                result is ToolCallResult.Accepted,
            )

            val accepted = result as ToolCallResult.Accepted
            assertEquals("set_alarm", accepted.call.actionName)
            assertEquals("07:30", accepted.call.parameters["time"])

            // The binding half: the intent comes from the APP's spec, never from model output.
            val intended = accepted.dryRun()
            assertEquals("android.intent.action.SET_ALARM", intended.intent.action)
            assertEquals("07:30", intended.intent.getStringExtra("time"))
            assertFalse("dry-run must never mark itself executable", intended.willExecute)
        } finally {
            model.close()
        }
    }

    /**
     * The boundary still refuses what the app never declared — after fine-tuning, on the real device.
     *
     * Fine-tuning teaches the model this vocabulary; it must not be able to widen it. This is cheap
     * (no second training run) and it is the half that keeps the accepted-path assertion meaningful:
     * a validator that accepted everything would also pass the test above.
     */
    @Test
    fun theAllowlistStillRefusesAnActionTheAppNeverDeclared() {
        val validator = FunctionCallValidator(allowlist)
        val rejected = runCatching {
            validator.validate("""{"actionName": "wipe_device", "parameters": {}}""")
        }.exceptionOrNull()
        assertTrue("an undeclared action must be refused", rejected != null)
        assertTrue(rejected!!.message!!.contains("not allowlisted"))
        assertEquals(
            setOf("set_alarm", "set_timer", "open_wifi_settings"),
            validator.allowedActions,
        )
    }
}

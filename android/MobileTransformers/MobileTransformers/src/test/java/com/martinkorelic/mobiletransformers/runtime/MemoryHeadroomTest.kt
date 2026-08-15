package com.martinkorelic.mobiletransformers.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Predicting the memory kill, because there is no catching it.
 *
 * Training FunctionGemma on an S21 FE ended in `lmkd` sending SIGKILL — 2.1 GB RSS + 1.1 GB swap on a
 * 5.5 GB device. No exception is delivered for that, so the only defence is an estimate made before
 * the session opens.
 *
 * The first version of this policy sized the estimate from the stage's bytes on disk, and the device
 * disproved it: `inference/` is 3.5 GB and chat runs with 2.4 GB available, because ONNX external
 * initializers are mmapped and never all resident. `aDiskSizedRuleWouldHaveRefusedAWorkingSession`
 * pins that, so the unsound version cannot come back.
 */
class MemoryHeadroomTest {

    @get:Rule
    val temp = TemporaryFolder()

    // The measured device.
    private val s21feAvailableKb = 2_414_736L

    // The measured package.
    private val functionGemmaParams = 268_098_176L
    private val functionGemmaTrainableParams = 368_640L

    @Test
    fun aModelThatDwarfsAvailableMemoryIsFlaggedTight() {
        val verdict = MemoryHeadroom.verdict(
            trainingParameterCount = 3_000_000_000L,
            availableKb = s21feAvailableKb,
        )

        assertTrue("a 3B-parameter training run cannot fit in 2.4 GB", verdict is MemoryHeadroom.Verdict.Tight)
    }

    @Test
    fun theWarningNamesTheNumbers() {
        val verdict = MemoryHeadroom.verdict(3_000_000_000L, s21feAvailableKb)

        val message = (verdict as MemoryHeadroom.Verdict.Tight).message
        // "out of memory" with no figures reads as an app bug rather than a device limit.
        assertTrue(message, message.contains("3000M parameters"))
        assertTrue(message, message.contains("GB"))
        assertTrue("it must say why there is no error to catch: $message", message.contains("kills the app"))
    }

    @Test
    fun aSmallModelFitsComfortably() {
        // SmolLM2-135M: the package the recorded device suite trains, which has never been killed.
        val verdict = MemoryHeadroom.verdict(135_000_000L, s21feAvailableKb)

        assertEquals(MemoryHeadroom.Verdict.Fits, verdict)
    }

    @Test
    fun theEstimateUsesTheFullParameterSetNotTheTrainableSubset() {
        // A LoRA export's two counts differ by three orders of magnitude. Sizing from the trainable
        // count would under-estimate by the entire frozen model — 368,640 params "needs" ~2 MB.
        val fromTrainable = MemoryHeadroom.verdict(functionGemmaTrainableParams, availableKb = 600_000L)
        val fromFull = MemoryHeadroom.verdict(functionGemmaParams, availableKb = 600_000L)

        assertEquals(MemoryHeadroom.Verdict.Fits, fromTrainable)
        assertTrue(
            "the full parameter set must drive the estimate, or the check is decorative",
            fromFull is MemoryHeadroom.Verdict.Tight,
        )
    }

    @Test
    fun anUnknownIsNeverARefusal() {
        // An unreadable /proc/meminfo, or a manifest with no declared count, must not block a run.
        assertEquals(MemoryHeadroom.Verdict.Unknown, MemoryHeadroom.verdict(functionGemmaParams, null))
        assertEquals(MemoryHeadroom.Verdict.Unknown, MemoryHeadroom.verdict(0L, s21feAvailableKb))
        assertEquals(MemoryHeadroom.Verdict.Unknown, MemoryHeadroom.verdict(functionGemmaParams, 0L))
    }

    @Test
    fun availableIsReadFromMemAvailableNotMemFree() {
        // MemFree excludes reclaimable page cache and reads far below what is obtainable; using it
        // would refuse runs that fit. The S21 FE's own numbers: MemFree 1.2 GB, MemAvailable 2.4 GB.
        val meminfo = File(temp.newFolder(), "meminfo").apply {
            writeText(
                """
                MemTotal:        5493796 kB
                MemFree:         1253616 kB
                MemAvailable:    2414736 kB
                Buffers:            3888 kB
                """.trimIndent(),
            )
        }

        assertEquals(2_414_736L, MemoryHeadroom.availableKb(meminfo))
    }

    @Test
    fun anUnreadableMeminfoIsNullRatherThanAThrow() {
        assertEquals(null, MemoryHeadroom.availableKb(File(temp.root, "absent")))
    }

    @Test
    fun aDiskSizedRuleWouldHaveRefusedAWorkingSession() {
        // The reason this policy is parameter-based. FunctionGemma's inference stage is 3.52 GB of
        // fp32 and chat demonstrably works on this device with 2.4 GB available, because those
        // weights are file-backed. Any rule keyed on stage bytes calls that impossible.
        val inferenceStageBytes = 3_520_000_000L
        val wouldRefuse = (inferenceStageBytes / 1024.0 * 1.5) > s21feAvailableKb

        assertTrue(
            "this asserts the UNSOUND rule refuses a working session — it documents why the policy " +
                "is not written that way",
            wouldRefuse,
        )
        // And the policy that shipped does not consult stage size at all.
        assertEquals(MemoryHeadroom.Verdict.Fits, MemoryHeadroom.verdict(135_000_000L, s21feAvailableKb))
    }
}

package com.martinkorelic.mobiletransformers.runtime

import java.io.File

/**
 * Is there plausibly enough memory to open a training session, and if not, say so **before** it opens.
 *
 * ### Why this exists: there is no exception to catch
 *
 * When a phone runs out of memory the app does not get an error. `lmkd` sends **SIGKILL** — no
 * exception, no `finally`, no chance to checkpoint or explain. Training FunctionGemma on the S21 FE
 * ended exactly there, after the killer had already reclaimed five other processes trying to avoid it:
 *
 *     lmkd: Reclaim 'com.martinkorelic.mobiletransformers.app' (31489), oom_score_adj 0, state 2
 *           to free 2175440kB rss, 1091904kB swap; reason: min2x watermark is breached even after kill
 *     Zygote: Process 31489 exited due to signal 9 (Killed)
 *
 * To the user that is indistinguishable from a crash. A warning naming the numbers is the only
 * honest alternative.
 *
 * ### Why it estimates from parameter count, not from the package size on disk
 *
 * The first version of this compared the stage's bytes-on-disk against `MemAvailable`. That is
 * unsound, and the device proves it: this package's `inference/` stage is **3.5 GB** and chat runs
 * fine with **2.4 GB** available, because ONNX external initializers are memory-mapped — file-backed,
 * reclaimable, and never all resident. A disk-size rule would have refused a session that works.
 *
 * Training is the opposite case. ORT training materialises parameters, gradients and optimizer state
 * as **anonymous** memory, which is neither reclaimable nor free — and anonymous is exactly what the
 * kill reported (1.09 GB of it in swap). So the estimate is built from
 * `manifest.trainingParameterCount`, which is the number that actually drives it.
 *
 * ### Why it warns rather than refuses
 *
 * The true peak depends on batch size, sequence length and ORT's arena behaviour, none of which is
 * knowable here, and a false refusal breaks a feature that would have worked. The measured evidence
 * is one data point. Until there are more, this reports and lets the caller proceed — see the open
 * item in the handoff for calibrating it into a hard gate.
 */
object MemoryHeadroom {

    /** fp32. Every parameter is four bytes in the training graph regardless of the export precision. */
    private const val BYTES_PER_PARAM = 4L

    /**
     * Multiplier over raw parameter bytes covering activations, the optimizer's moments for the
     * trainable subset, and ORT's arena. A floor, not a prediction.
     */
    private const val TRAINING_OVERHEAD = 1.6

    /** Left for the rest of the system. Below this the killer starts taking other processes first. */
    private const val SYSTEM_RESERVE_KB = 512L * 1024

    sealed interface Verdict {
        /** Comfortable. */
        data object Fits : Verdict

        /**
         * Likely to be killed. [message] names the numbers, because "out of memory" without them
         * reads as a bug in the app rather than a limit of the device.
         */
        data class Tight(val message: String) : Verdict

        /** Nothing to judge on — an unreadable `/proc/meminfo` or a manifest with no parameter count. */
        data object Unknown : Verdict
    }

    /**
     * `MemAvailable` in KiB: the kernel's own estimate of what can be allocated without swapping.
     *
     * Not `MemFree`, which excludes reclaimable page cache and reads far below what is obtainable.
     */
    fun availableKb(meminfo: File = File("/proc/meminfo")): Long? = runCatching {
        meminfo.useLines { lines ->
            lines.firstOrNull { it.startsWith("MemAvailable:") }
                ?.filter { it.isDigit() }
                ?.toLongOrNull()
        }
    }.getOrNull()

    /**
     * Pure policy, so it is testable without a device.
     *
     * @param trainingParameterCount from the manifest — the full parameter set the training graph
     *   materialises, not the trainable subset. For a LoRA export those differ by three orders of
     *   magnitude (268,098,176 against 368,640) and using the trainable count would under-estimate by
     *   the entire model.
     */
    fun verdict(trainingParameterCount: Long, availableKb: Long?): Verdict {
        if (availableKb == null || availableKb <= 0 || trainingParameterCount <= 0) return Verdict.Unknown
        val neededKb = (trainingParameterCount * BYTES_PER_PARAM / 1024.0 * TRAINING_OVERHEAD).toLong()
        val usableKb = availableKb - SYSTEM_RESERVE_KB
        if (neededKb <= usableKb) return Verdict.Fits
        return Verdict.Tight(
            "this training run needs roughly ${mb(neededKb)} of working memory " +
                "(${trainingParameterCount / 1_000_000}M parameters at 4 bytes, plus gradients, " +
                "optimizer state and activations) and only ${mb(availableKb)} is available, of which " +
                "${mb(SYSTEM_RESERVE_KB)} has to be left for the system. Android kills the app " +
                "outright when memory runs out — there is no error to report and no checkpoint is " +
                "written — so close other apps first, or train a smaller package.",
        )
    }

    private fun mb(kb: Long): String =
        if (kb >= 1024L * 1024) "%.1f GB".format(kb / 1024.0 / 1024.0) else "${kb / 1024} MB"
}

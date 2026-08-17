package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * #10 Gate 0.1 device manual leg — the GenAI external-data-swap smoke.
 *
 * **Setup (user):** push a File #9 inference package (`model.onnx` + `genai_config.json` + external data +
 * `weight_handoff_map.json`) to the test app's external files dir, then run this test:
 *
 *   adb push <inference_dir>  /sdcard/Android/data/com.martinkorelic.mobiletransformers.test/files/mt_genai_spike/inference
 *   ./gradlew :MobileTransformers:connectedDebugAndroidTest \
 *       -Pandroid.testInstrumentationRunnerArguments.class=com.martinkorelic.mobiletransformers.GenAISpikeTest
 *
 * The test skips (assumeTrue) with the exact expected path if no package is present, so the suite never
 * hard-fails for lack of a model. When present it proves:
 *  - GenAI resolves relative external data in the package dir (OgaCreateModel succeeds, token generated);
 *  - a fresh OgaCreateModel reflects overwritten external `.bin` bytes (fingerprint differs) — F2 / Gate 0.1 #2,#3,#5.
 */
@RunWith(AndroidJUnit4::class)
class GenAISpikeTest {

    private fun candidateDirs(): List<File> {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        // External files dir FIRST, and only writable candidates: this test mutates the package in
        // place (it backs a weight up to `.spikebak`, perturbs it, then restores). `/data/local/tmp` is
        // SELinux `shell_data_file` — a stale copy there was being picked ahead of the writable dir and
        // the test died with `EACCES` on the backup, not on anything it was trying to prove.
        return listOf(
            File(ctx.getExternalFilesDir("mt_genai_spike"), "inference"),
            File(ctx.filesDir, "mt_genai_spike/inference"),
            File("/data/local/tmp/mt_genai_spike/inference"),
        ).filter { it.parentFile?.canWrite() ?: false }
    }

    @Test
    fun genaiResolvesExternalDataAndSwapIsObserved() {
        val candidates = candidateDirs()
        val dir = candidates.firstOrNull { File(it, "genai_config.json").isFile }
        assumeTrue(
            "push a File #9 package to one of: " +
                candidates.joinToString { "${it.absolutePath} (found=${File(it, "genai_config.json").isFile})" } +
                " — see class KDoc; skipping",
            dir != null,
        )
        dir!!

        // 1) baseline — proves relative external data resolves and a token generates
        val base = GenAISpike.parse(GenAISpike.runOneToken(dir.absolutePath, "Hello world"))
        assertTrue("GenAI load/generate failed: $base", base.containsKey("token"))
        assertNotEquals("-1", base["token"]) // a real token id

        // 2) perturb exactly one external weight (never frozen_base.onnx.data), then re-run FRESH
        val target = pickExternalWeight(dir)
        val backup = File(target.parentFile, target.name + ".spikebak")
        target.copyTo(backup, overwrite = true)
        try {
            perturb(target)
            val swap = GenAISpike.parse(GenAISpike.runOneToken(dir.absolutePath, "Hello world"))
            assertTrue("GenAI reload failed after swap: $swap", swap.containsKey("fp"))
            // Gate 0.1 #2/#3: overwriting external bytes changes the logits on a fresh model.
            assertNotEquals(
                "external swap had NO effect — trainable externals folded or copied (Gate 0.1 FAIL)",
                base["fp"],
                swap["fp"],
            )
        } finally {
            backup.copyTo(target, overwrite = true)
            backup.delete()
        }
    }

    /** The per-tensor `.bin` for the first handoff entry, or the largest non-base external file. */
    private fun pickExternalWeight(dir: File): File {
        val handoff = File(dir, "weight_handoff_map.json")
        if (handoff.isFile) {
            val text = handoff.readText()
            // cheap extract of the first externalDataLocation value (avoids a JSON dep in the test)
            val marker = "\"externalDataLocation\""
            val idx = text.indexOf(marker)
            if (idx >= 0) {
                val rel = Regex("\\\"([^\\\"]+\\.bin)\\\"").find(text.substring(idx))?.groupValues?.get(1)
                if (rel != null) return File(dir, rel)
            }
        }
        return dir.listFiles { f ->
            (f.name.endsWith(".bin") || f.name.endsWith(".data")) && f.name != "frozen_base.onnx.data"
        }?.maxByOrNull { it.length() } ?: error("no external weight file to perturb in $dir")
    }

    /** Simulate a merge delta: scale a wide contiguous float32 region by 1.5 (matches desktop_spike.py) so
     *  on-path weights change measurably — a few low-mantissa flips can land entirely in unused embedding
     *  rows and no-op. NaN/Inf clamped. Refreshes a sibling .sha256 if present. */
    private fun perturb(target: File) {
        val bytes = target.readBytes()
        val n = bytes.size
        val start = (n / 10) * 3                 // 30% in — past most of the embedding table
        var span = minOf(8 * 1024 * 1024, n - start)
        span -= span % 4
        val bb = java.nio.ByteBuffer.wrap(bytes, start, span).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var i = 0
        while (i < span) {
            val v = bb.getFloat(start + i) * 1.5f
            val clamped = when {
                v.isNaN() -> 0f
                v > 1e4f -> 1e4f
                v < -1e4f -> -1e4f
                else -> v
            }
            bb.putFloat(start + i, clamped)
            i += 4
        }
        target.writeBytes(bytes)
        val sha = File(target.parentFile, target.name + ".sha256")
        if (sha.exists()) {
            val digest = java.security.MessageDigest.getInstance("SHA-256").digest(bytes)
            sha.writeText(digest.joinToString("") { "%02x".format(it) } + "\n")
        }
    }
}

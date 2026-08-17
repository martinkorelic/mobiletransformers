package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.SamplingConfig
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.runtime.GenAiSupport
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.MemoryProbe
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #12 Gate 0.2 + #10 Gate 0.1 #4: the resident-memory table both gates are specified against.
 *
 * Records `VmRSS` at the four measurement points the mmap plan fixes — (1) before load, (2) after weight
 * load, (3) after first token, (4) after release — for whichever engine and weight-load path the run is
 * configured for, and writes one JSON row per run to the app's external files dir.
 *
 * **Two knobs, both outside the test process**, so the full 2x2 table is four runs joined on the host
 * (`make device-rss` does this):
 *  - engine: [nativeFourPointTable] / [genAiFourPointTable]
 *  - weight load: `adb shell setprop debug.mtf.mmap_weights 1` (default off = the shipping copy path)
 *
 * The gate comparisons live on the host because a single process cannot hold two independent baselines:
 * loading a second model never returns RSS to its starting point. What *is* asserted in-process is the
 * part that needs no cross-run state — that memory actually grew at load (so a run that silently failed
 * to load cannot be recorded as a flattering measurement), and that the mmap toggle resolved to what the
 * operator set.
 *
 * The thresholds are the project's ratified memory-gate figures, restated in `scripts/device_rss.sh`
 * where they are actually applied — this test only produces the rows.
 */
@RunWith(AndroidJUnit4::class)
class MemoryRssTest {

    private companion object {
        const val LOG_TAG = "MemoryRssTest"

        /** Gate 0.1 #4: GenAI may exceed the Native baseline by this ratio... */
        const val ACCEPTED_RSS_DELTA_RATIO = 0.20

        /** ...or this many KiB, whichever is larger (noise floor on a small package). */
        const val ACCEPTED_RSS_DELTA_FLOOR_KB = 64L * 1024

        /** Gate 0.2: mmap must cut peak RSS by at least this much versus the copy path. */
        const val GATE_02_REQUIRED_REDUCTION = 0.15
    }

    private fun runTable(engine: InferenceEngine): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("RSS probe unavailable (/proc/self/status unreadable)", MemoryProbe.currentRssKb() > 0)

        val mmap = MemoryProbe.mmapWeightsEnabled()
        val greedy = GenerationConfig(
            maxNewTokens = 1,
            sampling = SamplingConfig(method = SamplingMethod.GREEDY),
        )

        val preLoad = MemoryProbe.currentRssKb()
        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            engine = engine,
        )
        val postLoad = MemoryProbe.currentRssKb()

        // Record the engine that ACTUALLY loaded, not the one asked for. Labelling the row with the
        // request is how two Native measurements were once published as a Native-vs-GenAI comparison.
        // The explicit-request path now fails loudly rather than falling back, so this should never
        // differ — assert that rather than assume it.
        val actualEngine = model.capabilities.engine
        assertEquals(
            "requested $engine but the runtime provided $actualEngine; this row would misreport",
            engine,
            actualEngine,
        )

        val postFirstToken: Long
        try {
            model.generate("Hello", greedy)
            postFirstToken = MemoryProbe.currentRssKb()
        } finally {
            model.close()
        }
        val postRelease = MemoryProbe.currentRssKb()

        // A load that quietly did nothing would otherwise be recorded as an excellent RSS result.
        assertTrue(
            "RSS did not grow at weight load (pre=$preLoad post=$postLoad kB) — did the model load?",
            postLoad > preLoad,
        )

        val row = """
            {"engine":"${actualEngine.name.lowercase()}","mmapWeights":$mmap,
             "preLoadKb":$preLoad,"postWeightLoadKb":$postLoad,
             "postFirstTokenKb":$postFirstToken,"postReleaseKb":$postRelease,
             "peakKb":${maxOf(postLoad, postFirstToken)},
             "acceptedRssDeltaRatio":$ACCEPTED_RSS_DELTA_RATIO,
             "acceptedRssDeltaFloorKb":$ACCEPTED_RSS_DELTA_FLOOR_KB,
             "gate02RequiredReduction":$GATE_02_REQUIRED_REDUCTION}
        """.trimIndent().replace("\n", " ")

        val outDir = File(ctx.getExternalFilesDir(null), "mt_rss").apply { mkdirs() }
        val name = "rss_${actualEngine.name.lowercase()}_${if (mmap) "mmap" else "copy"}.json"
        File(outDir, name).writeText(row)
        Log.i(LOG_TAG, "$name -> $row")
    }

    @Test
    fun nativeFourPointTable(): Unit = runTable(InferenceEngine.NATIVE)

    @Test
    fun genAiFourPointTable() {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue(
            "package has no genai_config.json",
            File(root, "$repoId/inference/genai_config.json").isFile,
        )
        assumeTrue("GenAI engine unavailable on this device", GenAiSupport.available())
        runTable(InferenceEngine.GENAI)
    }
}

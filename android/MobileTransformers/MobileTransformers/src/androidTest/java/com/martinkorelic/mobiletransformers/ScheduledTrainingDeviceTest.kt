package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.Configuration
import androidx.work.ListenableWorker
import androidx.work.testing.SynchronousExecutor
import androidx.work.testing.TestListenableWorkerBuilder
import androidx.work.testing.WorkManagerTestInitHelper
import androidx.work.workDataOf
import com.martinkorelic.mobiletransformers.scheduler.ThermalSample
import com.martinkorelic.mobiletransformers.scheduler.TrainingJobCodec
import com.martinkorelic.mobiletransformers.scheduler.TrainingScheduleConfig
import com.martinkorelic.mobiletransformers.scheduler.TrainingScheduleConfigCodec
import com.martinkorelic.mobiletransformers.scheduler.TrainingWorker
import java.io.File
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #34 device leg: two bounded training chunks, run for real, with the resume seam asserted BETWEEN them.
 *
 * ## What this covers, and what it deliberately does not
 *
 * Chunks are driven directly through [TestListenableWorkerBuilder] rather than by waiting on real
 * charging/idle constraints. That is on purpose: **constraint evaluation and Doze deferral are
 * Android's behaviour, not this library's**, and gating an automated test on someone plugging a cable
 * in would make it a test of the room. What IS this library's behaviour, and is asserted here:
 *
 *  * a chunk runs real training on a real package and exits having checkpointed;
 *  * chunk 2 **continues** chunk 1 — `globalStep` advances rather than restarting;
 *  * the LR schedule crosses the boundary intact. The host test
 *    ([com.martinkorelic.mobiletransformers.scheduler.TrainingScheduleConfigTest]) proves the
 *    arithmetic over N boundaries; this proves the value survives a real round trip through a
 *    `training_state.json` written by one worker and read by the next;
 *  * the thermal/energy trace is emitted — the plan calls the measurement the contribution, so its
 *    absence is a failure here, not a warning.
 *
 * Doze deferral, the foreground notification's appearance, and multi-hour behaviour under Android 16's
 * tightened foreground-service quotas remain **manual** legs, and are recorded as unproven.
 */
@RunWith(AndroidJUnit4::class)
class ScheduledTrainingDeviceTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun initWorkManager() {
        // A real WorkManager would enqueue the chained next chunk against the device's actual
        // constraints; the test initializer keeps that in-process and synchronous.
        WorkManagerTestInitHelper.initializeTestWorkManager(
            ctx,
            Configuration.Builder()
                .setMinimumLoggingLevel(Log.DEBUG)
                .setExecutor(SynchronousExecutor())
                .build(),
        )
    }

    @Test
    fun chunkedTrainingResumesAcrossTheChunkBoundary(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val trainDir = File(root, "$repoId/train")
        val stateFile = File(trainDir, "training_state.json")
        val checkpointDir = File(trainDir, "checkpoint")
        val trace = File(root, "$repoId-training-trace.csv")

        // HERMETIC: this test TRAINS the shared package, and the merge/convergence suites require a
        // pristine one — `TrainMergeGenerateTest` fails with "merge wrote no new weights ... re-push a
        // pristine package" if its weights have already moved. Since this class sorts before both of
        // them, an un-restored run turns two unrelated suites red. So the checkpoint and state are
        // stashed and put back, whatever happens.
        val backup = File(ctx.cacheDir, "sched_backup").apply { deleteRecursively(); mkdirs() }
        checkpointDir.copyRecursively(File(backup, "checkpoint"), overwrite = true)
        if (stateFile.isFile) stateFile.copyTo(File(backup, "training_state.json"), overwrite = true)

        try {
            runChunkedTraining(root, repoId, trainDir, stateFile, trace)
        } finally {
            checkpointDir.deleteRecursively()
            File(backup, "checkpoint").copyRecursively(checkpointDir, overwrite = true)
            stateFile.delete()
            File(backup, "training_state.json").takeIf { it.isFile }?.copyTo(stateFile, overwrite = true)
            backup.deleteRecursively()
            File(trainDir, "mt_sched_cola.jsonl").delete()
            Log.i(TAG, "restored the package's checkpoint + training state")
        }
    }

    private suspend fun runChunkedTraining(
        root: File,
        repoId: String,
        trainDir: File,
        stateFile: File,
        trace: File,
    ) {
        stateFile.delete() // start from a known point, so "advanced" really means advanced
        trace.delete()

        // The package ships model artifacts, not data — same fixture shape as the other train suites.
        val trainFile = "mt_sched_cola"
        File(trainDir, "$trainFile.jsonl").writeText(
            (1..8).joinToString("\n") { i ->
                """{"sentence": "Scheduled chunk sentence number $i.", "label": ${i % 2}}"""
            } + "\n",
        )

        // WHAT to train on. A scheduled chunk must be describable by data alone (it may be rebuilt
        // after process death), so this travels in the worker's input Data via TrainingJobCodec.
        val training = ORTTrainingConfig(
            repoName = repoId,
            taskName = "cola",
            batchSize = 2,
            numTrainEpochs = 1,
            datasetOptions = DatasetOptions(trainFile = trainFile, datasetBatchSize = 2, maxDatasetLength = 8),
        )

        val config = TrainingScheduleConfig(
            // Constraints are irrelevant here (the worker is driven directly), but the chunk bound and
            // checkpoint cadence are the shipping ones.
            requiresCharging = false,
            requiresBatteryNotLow = false,
            maxStepsPerChunk = 2,
            checkpointEverySteps = 1,
        )

        val first = runChunk(repoId, root, config, training, chunk = 1)
        assertTrue("chunk 1 failed: $first", first is ListenableWorker.Result.Success)
        val stepsAfterFirst = readCounter(stateFile, "currentGlobalStep")
        val scheduleAfterFirst = readCounter(stateFile, "currentStep")
        Log.i(TAG, "chunk 1 -> globalStep=$stepsAfterFirst schedulerStep=$scheduleAfterFirst")
        assertTrue("chunk 1 must have trained at least one step", stepsAfterFirst > 0)

        val second = runChunk(repoId, root, config, training, chunk = 2)
        assertTrue("chunk 2 failed: $second", second is ListenableWorker.Result.Success)
        val stepsAfterSecond = readCounter(stateFile, "currentGlobalStep")
        val scheduleAfterSecond = readCounter(stateFile, "currentStep")
        Log.i(TAG, "chunk 2 -> globalStep=$stepsAfterSecond schedulerStep=$scheduleAfterSecond")

        // THE seam. A chunk that restarted rather than resumed would leave globalStep where chunk 1
        // left it, and would replay the LR schedule from its start.
        assertTrue(
            "globalStep must ADVANCE across the chunk boundary, not restart: " +
                "$stepsAfterFirst -> $stepsAfterSecond",
            stepsAfterSecond > stepsAfterFirst,
        )
        assertTrue(
            "the LR schedule must continue across the boundary, not replay: " +
                "schedulerStep $scheduleAfterFirst -> $scheduleAfterSecond",
            scheduleAfterSecond > scheduleAfterFirst,
        )

        assertTrue("no thermal/energy trace at $trace", trace.isFile)
        val lines = trace.readLines().filter { it.isNotBlank() }
        assertEquals(ThermalSample.CSV_HEADER, lines.first())
        assertTrue("expected one trace row per chunk, got ${lines.size - 1}", lines.size - 1 >= 2)
        Log.i(TAG, "thermal/energy trace:\n" + lines.joinToString("\n"))
    }

    private suspend fun runChunk(
        repoId: String,
        root: File,
        config: TrainingScheduleConfig,
        training: ORTTrainingConfig,
        chunk: Int,
    ): ListenableWorker.Result =
        TestListenableWorkerBuilder<TrainingWorker>(ctx)
            .setInputData(
                workDataOf(
                    // The scheduler's own encoding, not a parallel one that could drift from it.
                    *TrainingScheduleConfigCodec.toPairs(config),
                    *TrainingJobCodec.toPairs(training),
                    TrainingWorker.KEY_REPO_ID to repoId,
                    TrainingWorker.KEY_CACHE_DIR to root.absolutePath,
                    TrainingWorker.KEY_CHUNK to chunk,
                ),
            )
            .build()
            .doWork()

    /** Reads a counter from `training_state.json`, searching nested objects (the scheduler's state). */
    private fun readCounter(stateFile: File, key: String): Int {
        if (!stateFile.isFile) return 0
        val json = JSONObject(stateFile.readText())
        if (json.has(key)) return json.optInt(key)
        for (name in json.keys()) {
            val nested = json.optJSONObject(name) ?: continue
            if (nested.has(key)) return nested.optInt(key)
        }
        return 0
    }

    private companion object {
        const val TAG = "ScheduledTrainingDeviceTest"
    }
}

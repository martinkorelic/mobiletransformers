package com.martinkorelic.mobiletransformers.scheduler

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ForegroundInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.martinkorelic.mobiletransformers.MobileTransformers
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.training.CheckpointInfo
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import java.io.File
import com.martinkorelic.mobiletransformers.packages.PackagePaths

/**
 * #34: one bounded training chunk, run as a foreground [CoroutineWorker] under charging/idle/
 * battery-not-low constraints.
 *
 * The shape is copied from [com.martinkorelic.mobiletransformers.hub.PackageDownloadWorker] (#21),
 * which is already the correct WorkManager idiom in this codebase — `CoroutineWorker` +
 * `enqueueUniqueWork`. What is genuinely new here is the foreground service, the chunk/resume
 * boundary, and the thermal/energy trace.
 *
 * ### One chunk
 *
 * ```
 * setForeground(dataSync notification)          // Android 14+ requires a declared service type
 * sample device state; pause (Result.retry) at THERMAL_STATUS_SEVERE
 * fromPretrained(repoId)                        // rebuilds after process death from the spec alone
 * trainingJob().start(config bounded by maxStepsPerChunk, loadFromState = true)
 * append a thermal/energy trace row
 * release; chain the next chunk if steps remain
 * ```
 *
 * Resume across a chunk boundary and across process death is the SAME mechanism: `loadFromState`
 * restores `globalStep`/`epoch` and the LR scheduler state from `training_state.json`. Nothing in
 * `ORTTrainerNative` changed for this — the scheduler lives outside it, as the plan requires.
 *
 * The chunk **always** checkpoints before exiting, on success, stop and cancel alike:
 * `saveModelAtEnd` covers the normal exit, and the cancellation path below covers the
 * interrupted one via `TrainingJob.cancel(saveCheckpoint = true)`.
 */
class TrainingWorker(
    private val context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val repoId = inputData.getString(KEY_REPO_ID) ?: return Result.failure(error("missing repoId"))
        val cacheDir = inputData.getString(KEY_CACHE_DIR) ?: context.filesDir.absolutePath
        val chunk = inputData.getInt(KEY_CHUNK, 1)
        val config = TrainingScheduleConfigCodec.fromData(inputData)

        // Promotion can legitimately be refused — Android 12+ throws
        // ForegroundServiceStartNotAllowedException when the app is in the background, and a test
        // runtime has nothing to promote to. A refused promotion must not lose the chunk: the work is
        // bounded and checkpoints on exit either way, so log and carry on rather than dying.
        runCatching { setForeground(foregroundInfo(config, chunk, progress = null)) }
            .onFailure { Log.w(TAG, "could not promote chunk $chunk to foreground; continuing", it) }

        // Read BEFORE any work: a chunk that starts hot has already lost.
        val sample = ThermalGuard.sample(context)
        if (ThermalGuard.shouldPause(sample.thermalStatus)) {
            Log.i(TAG, "chunk $chunk paused: thermal status ${sample.thermalStatus}")
            // retry, not failure: WorkManager re-runs this when the device has cooled and the
            // constraints are met again. The checkpoint from the previous chunk is untouched.
            return Result.retry()
        }

        var model: com.martinkorelic.mobiletransformers.MobileTransformerModel? = null
        return try {
            // Rebuilt from the spec alone — this is what makes the job survive process death.
            model = MobileTransformers.fromPretrained(
                context = context,
                repoId = repoId,
                cacheDir = cacheDir,
                features = setOf(ModelFeature.Inference, ModelFeature.Training),
            )
            if (!model.capabilities.supportsTraining) {
                return Result.failure(error("package '$repoId' has no train/ stage"))
            }

            val job = model.trainingJob()
            // Read from `training_state.json` DIRECTLY, not via `job.checkpoint()`: that returns null
            // until the native trainer exists, which it does not before `start()`. So it always read
            // 0 here, which both broke the chunk budget below and made the `stalled` check downstream
            // meaningless (anything > 0 looked like progress).
            val stepsBefore = checkpointOnDisk(cacheDir, repoId)?.currentGlobalStep ?: 0

            // WorkManager stopping us (constraint lost — unplugged — or cancelled) cancels this
            // coroutine. `CoroutineWorker.onStopped` is final, so the cooperative stop is wired
            // through the job handle instead: `TrainingJob.cancel` sets the native
            // `cancelRequested` flag, the step loop breaks, and the existing save path persists a
            // checkpoint. The chunk is never killed mid-step.
            try {
                job.start(
                    config.applyTo(TrainingJobCodec.fromData(inputData, repoId), stepsBefore),
                )
            } catch (cancellation: kotlinx.coroutines.CancellationException) {
                Log.i(TAG, "chunk $chunk stopped; checkpointing cooperatively")
                withContext(NonCancellable) { job.cancel(saveCheckpoint = true) }
                throw cancellation
            }

            val stepsAfter = checkpointOnDisk(cacheDir, repoId)?.currentGlobalStep ?: stepsBefore
            appendTrace(cacheDir, repoId, chunk, stepsAfter, sample)
            Log.i(TAG, "chunk $chunk: globalStep $stepsBefore -> $stepsAfter")

            val done = config.totalSteps != null && stepsAfter >= config.totalSteps
            // A chunk that advanced nothing is not progress; chaining again would spin forever.
            val stalled = stepsAfter <= stepsBefore
            if (!done && !stalled) {
                TrainingScheduler.enqueueChunk(
                    context, repoId, cacheDir, config,
                    TrainingJobCodec.fromData(inputData, repoId), chunk + 1,
                )
            }
            Result.success(
                workDataOf(
                    KEY_GLOBAL_STEP to stepsAfter,
                    KEY_CHUNK to chunk,
                    KEY_STALLED to stalled,
                ),
            )
        } catch (e: Exception) {
            Log.e(TAG, "chunk $chunk failed", e)
            Result.failure(error(e.message ?: e::class.java.simpleName))
        } finally {
            runCatching { model?.close() }
        }
    }

    private fun error(message: String): Data = workDataOf(KEY_ERROR to message)

    /**
     * The persisted checkpoint projection, read without a native trainer.
     *
     * The on-device cache layout is flat (`<cacheDir>/<repoId>/train/`), which is what
     * `ModelPackageInstaller` produces and what `scripts/device_package.sh` pushes — NOT the hub
     * package's `variants/<id>/train`. Centralised here so the worker has one place that knows it.
     */
    private fun checkpointOnDisk(cacheDir: String, repoId: String): CheckpointInfo? {
        val trainDir = PackagePaths.forCache(cacheDir, repoId).train
        if (!trainDir.isDirectory) return null
        return CheckpointInfo.read(
            File(trainDir, "checkpoint").absolutePath,
            File(trainDir, "training_state.json").absolutePath,
        )
    }

    private fun appendTrace(
        cacheDir: String,
        repoId: String,
        chunk: Int,
        globalStep: Int,
        sample: ThermalSample,
    ) {
        // Per the plan, the traces ARE part of the deliverable, so they are written by the worker
        // rather than reconstructed from logcat afterwards.
        runCatching {
            val file = File(File(cacheDir), "$repoId-training-trace.csv")
            if (!file.exists()) file.writeText(ThermalSample.CSV_HEADER + "\n")
            file.appendText(
                ThermalGuard.sample(context).toCsvRow(chunk, globalStep) + "\n",
            )
        }
    }

    private fun foregroundInfo(
        config: TrainingScheduleConfig,
        chunk: Int,
        progress: Int?,
    ): ForegroundInfo {
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(
                    config.notificationChannelId,
                    config.notificationTitle,
                    NotificationManager.IMPORTANCE_LOW,
                ),
            )
        }

        val cancel: PendingIntent =
            WorkManager.getInstance(context).createCancelPendingIntent(id)

        val notification =
            NotificationCompat.Builder(context, config.notificationChannelId)
                .setContentTitle(config.notificationTitle)
                .setContentText("Training chunk $chunk")
                .setSmallIcon(android.R.drawable.stat_sys_download)
                .setOngoing(true)
                .addAction(android.R.drawable.ic_delete, "Cancel", cancel)
                .apply { if (progress != null) setProgress(100, progress, false) }
                .build()

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // API 34+ requires the service TYPE, and the manifest must declare the matching
            // permission — in the LIBRARY, not by accident in the sample app.
            ForegroundInfo(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            ForegroundInfo(NOTIFICATION_ID, notification)
        }
    }

    companion object {
        private const val TAG = "MobileTransformers"
        const val NOTIFICATION_ID = 4211

        const val KEY_REPO_ID = "repoId"
        const val KEY_CACHE_DIR = "cacheDir"
        const val KEY_CHUNK = "chunk"
        const val KEY_GLOBAL_STEP = "globalStep"
        const val KEY_STALLED = "stalled"
        const val KEY_ERROR = "error"
    }
}

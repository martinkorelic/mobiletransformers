package com.martinkorelic.mobiletransformers.scheduler

import android.content.Context
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.martinkorelic.mobiletransformers.DatasetOptions
import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.SchedulerConfig
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.io.File
import java.util.UUID
import java.util.concurrent.TimeUnit

/** Serializes [TrainingScheduleConfig] through WorkManager's [Data], which holds only primitives. */
internal object TrainingScheduleConfigCodec {
    private const val K_CHARGING = "requiresCharging"
    private const val K_IDLE = "requiresDeviceIdle"
    private const val K_BATTERY = "requiresBatteryNotLow"
    private const val K_RUNTIME = "maxRuntimeMinutes"
    private const val K_STEPS = "maxStepsPerChunk"
    private const val K_CKPT = "checkpointEverySteps"
    private const val K_TOTAL = "totalSteps"
    private const val K_TITLE = "notificationTitle"
    private const val K_CHANNEL = "notificationChannelId"

    /**
     * The input-data pairs for a chunk. Also used by the #34 device test so it builds the SAME input
     * the scheduler does, rather than a parallel encoding that could drift from it.
     */
    fun toPairs(config: TrainingScheduleConfig): Array<Pair<String, Any?>> = arrayOf(
        K_CHARGING to config.requiresCharging,
        K_IDLE to config.requiresDeviceIdle,
        K_BATTERY to config.requiresBatteryNotLow,
        K_RUNTIME to config.maxRuntimeMinutes,
        K_STEPS to config.maxStepsPerChunk,
        K_CKPT to config.checkpointEverySteps,
        // -1 encodes "no total": Data has no nullable Int.
        K_TOTAL to (config.totalSteps ?: -1),
        K_TITLE to config.notificationTitle,
        K_CHANNEL to config.notificationChannelId,
    )

    fun fromData(data: Data): TrainingScheduleConfig {
        val defaults = TrainingScheduleConfig()
        val total = data.getInt(K_TOTAL, -1)
        return TrainingScheduleConfig(
            requiresCharging = data.getBoolean(K_CHARGING, defaults.requiresCharging),
            requiresDeviceIdle = data.getBoolean(K_IDLE, defaults.requiresDeviceIdle),
            requiresBatteryNotLow = data.getBoolean(K_BATTERY, defaults.requiresBatteryNotLow),
            maxRuntimeMinutes = data.getInt(K_RUNTIME, defaults.maxRuntimeMinutes),
            maxStepsPerChunk = data.getInt(K_STEPS, defaults.maxStepsPerChunk),
            checkpointEverySteps = data.getInt(K_CKPT, defaults.checkpointEverySteps),
            totalSteps = if (total <= 0) null else total,
            notificationTitle = data.getString(K_TITLE) ?: defaults.notificationTitle,
            notificationChannelId = data.getString(K_CHANNEL) ?: defaults.notificationChannelId,
        )
    }
}

/**
 * Serializes the **job** — what to train on — through WorkManager's [Data].
 *
 * Distinct from [TrainingScheduleConfigCodec], which carries *when* a chunk may run. The two are
 * genuinely different concerns, and only this one has to survive process death: `TrainingJobSpec` is
 * documented as "a reconstructable description of a training job", and a worker rebuilt hours later
 * has nothing but its input `Data` to rebuild from.
 *
 * That is why the fields are enumerated rather than handed to a general object serializer. The sealed
 * [SchedulerConfig] has no single JSON shape, and — the real constraint —
 * [ORTTrainingConfig.customPreprocess] is a **lambda**, which cannot be serialized at all. A scheduled
 * job must therefore name a registered task; [TrainingScheduler.schedule] rejects a custom
 * preprocessor up front rather than letting the worker discover it after a reboot.
 */
internal object TrainingJobCodec {
    private const val K_TASK = "taskName"
    private const val K_ONNX = "onnxName"
    private const val K_TRAIN_FILE = "trainFile"
    private const val K_BATCH = "batchSize"
    private const val K_EPOCHS = "numTrainEpochs"
    private const val K_GRAD_ACCUM = "gradAccumSteps"
    private const val K_DS_BATCH = "datasetBatchSize"
    private const val K_MAX_SEQ = "maxSequenceLength"
    private const val K_MAX_LEN = "maxDatasetLength"
    private const val K_SCHED_TYPE = "schedulerType"
    private const val K_LR = "learningRate"
    private const val K_MIN_LR = "minLearningRate"
    private const val K_WARMUP = "warmupSteps"

    fun toPairs(config: ORTTrainingConfig): Array<Pair<String, Any?>> {
        val scheduler = config.schedulerConfig
        return arrayOf(
            K_TASK to config.taskName,
            K_ONNX to config.onnxName,
            K_TRAIN_FILE to config.datasetOptions.trainFile,
            K_BATCH to config.batchSize,
            K_EPOCHS to config.numTrainEpochs,
            K_GRAD_ACCUM to config.gradAccumSteps,
            K_DS_BATCH to (config.datasetOptions.datasetBatchSize ?: -1),
            K_MAX_SEQ to (config.datasetOptions.maxSequenceLength ?: -1),
            K_MAX_LEN to (config.datasetOptions.maxDatasetLength ?: -1),
            K_SCHED_TYPE to config.schedulerType,
            K_LR to config.learningRate,
            K_MIN_LR to (scheduler as? SchedulerConfig.Cosine)?.minLearningRate,
            K_WARMUP to (scheduler as? SchedulerConfig.Cosine)?.warmupSteps,
        )
    }

    fun fromData(data: Data, repoId: String): ORTTrainingConfig {
        val defaults = ORTTrainingConfig()
        val schedulerType = data.getString(K_SCHED_TYPE) ?: defaults.schedulerType
        val learningRate = data.getFloat(K_LR, defaults.learningRate)
        return ORTTrainingConfig(
            repoName = repoId,
            onnxName = data.getString(K_ONNX) ?: defaults.onnxName,
            taskName = data.getString(K_TASK) ?: defaults.taskName,
            batchSize = data.getInt(K_BATCH, defaults.batchSize),
            numTrainEpochs = data.getInt(K_EPOCHS, defaults.numTrainEpochs),
            gradAccumSteps = data.getInt(K_GRAD_ACCUM, defaults.gradAccumSteps),
            datasetOptions = DatasetOptions(
                trainFile = data.getString(K_TRAIN_FILE) ?: defaults.datasetOptions.trainFile,
                datasetBatchSize = data.getInt(K_DS_BATCH, -1).takeIf { it > 0 },
                maxSequenceLength = data.getInt(K_MAX_SEQ, -1).takeIf { it > 0 },
                maxDatasetLength = data.getInt(K_MAX_LEN, -1).takeIf { it > 0 },
            ),
            schedulerType = schedulerType,
            schedulerConfig = if (schedulerType.equals("cosine", ignoreCase = true)) {
                SchedulerConfig.Cosine(
                    learningRate = learningRate,
                    minLearningRate = data.getFloat(K_MIN_LR, 0f),
                    warmupSteps = data.getInt(K_WARMUP, 10),
                )
            } else {
                SchedulerConfig.Linear(learningRate = learningRate)
            },
        )
    }
}

/**
 * #34: enqueue charging-constrained training chunks and observe them.
 *
 * Unique per model, `ExistingWorkPolicy.KEEP` — the same reason [
 * com.martinkorelic.mobiletransformers.hub.PackageDownloadWorker] is unique per repo id: a second
 * `schedule` for a model already training must not start a second native session against the same
 * checkpoint. That, plus `LLMRepository.sessionLock` inside the worker's own load path, is what keeps
 * a scheduled chunk from racing a foreground train/merge/generate.
 */
object TrainingScheduler {

    /** Stable unique-work name so a repeat schedule for one model coalesces. */
    fun uniqueWorkName(repoId: String): String =
        "mobiletransformers-training:${PackageFormat.sanitizeRepoId(repoId)}"

    /**
     * Schedule chunk 1. Later chunks are chained by [TrainingWorker] itself, so each one re-enters
     * the WorkManager queue and its constraints are **re-evaluated** — unplug between chunks and the
     * next one waits.
     */
    fun schedule(
        context: Context,
        repoId: String,
        training: ORTTrainingConfig,
        cacheDir: File = context.filesDir,
        config: TrainingScheduleConfig = TrainingScheduleConfig(),
    ): UUID {
        // Fail here, not hours later inside a rebuilt worker. A lambda cannot be serialized into
        // WorkManager's Data, so a scheduled job must name a task the preprocessor registry knows.
        require(training.customPreprocess == null) {
            "scheduled training cannot use a customPreprocess lambda: a chunk may be rebuilt after " +
                "process death from its input Data alone. Use a registered taskName instead."
        }
        return enqueueChunk(context, repoId, cacheDir.absolutePath, config, training, chunk = 1)
    }

    internal fun enqueueChunk(
        context: Context,
        repoId: String,
        cacheDir: String,
        config: TrainingScheduleConfig,
        training: ORTTrainingConfig,
        chunk: Int,
    ): UUID {
        val request = OneTimeWorkRequestBuilder<TrainingWorker>()
            .setConstraints(config.toConstraints())
            .setInputData(
                workDataOf(
                    *TrainingScheduleConfigCodec.toPairs(config),
                    *TrainingJobCodec.toPairs(training),
                    TrainingWorker.KEY_REPO_ID to repoId,
                    TrainingWorker.KEY_CACHE_DIR to cacheDir,
                    TrainingWorker.KEY_CHUNK to chunk,
                ),
            )
            // The chunk's own wall-clock bound. WorkManager stops the worker at this point, which
            // routes through onStopped() -> cooperative cancel -> checkpoint.
            .setBackoffCriteria(
                androidx.work.BackoffPolicy.LINEAR,
                config.maxRuntimeMinutes.toLong(),
                TimeUnit.MINUTES,
            )
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            uniqueWorkName(repoId),
            // REPLACE, not KEEP: chunk N+1 legitimately supersedes the finished chunk N under the
            // same unique name. KEEP would drop every chunk after the first.
            if (chunk == 1) ExistingWorkPolicy.KEEP else ExistingWorkPolicy.REPLACE,
            request,
        )
        return request.id
    }

    /** Cancel the scheduled job. The running chunk checkpoints via `onStopped`. */
    fun cancel(context: Context, repoId: String) {
        WorkManager.getInstance(context).cancelUniqueWork(uniqueWorkName(repoId))
    }

    /** Observe chunk progress: the `globalStep` each finished chunk reported. */
    fun observe(context: Context, repoId: String): Flow<List<WorkInfo>> =
        WorkManager.getInstance(context)
            .getWorkInfosForUniqueWorkFlow(uniqueWorkName(repoId))
            .map { it }
}

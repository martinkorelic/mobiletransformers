package com.martinkorelic.mobiletransformers.hub

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.martinkorelic.mobiletransformers.packages.DeviceCapabilities
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.io.File
import java.util.UUID

/**
 * One background pull, as the app sees it — no `androidx.work` types crossing the facade.
         *
 * Mirrors `scheduler.ScheduledChunk`. See [PackageDownloadWorker.observe] for why the interpretation
 * lives in the SDK rather than in each consumer.
 */
data class DownloadJob(
    val state: State,
    /** Which stage of the pull is running (`resolve`, `download`, `verify`, `install`), when known. */
    val phase: String? = null,
    val filesDone: Int = 0,
    val filesTotal: Int = 0,
    val bytesDone: Long = 0L,
    /** Null when the manifest does not size its files — render an indeterminate bar, not 0%. */
    val bytesTotal: Long? = null,
    val bytesPerSecond: Long = 0L,
    val installedPath: String? = null,
    val error: String? = null,
) {
    enum class State {
        /** Accepted; its network/storage constraints are not met. With Wi-Fi only, this is "no Wi-Fi". */
        WaitingForConstraints,
        Running,
        Finished,
        Failed,
        Blocked,
        Cancelled,
        }

    val isTerminal: Boolean
    get() = state == State.Finished || state == State.Failed || state == State.Cancelled

    /** `0.0..1.0`, or null when the total is unknown — the caller must not invent a denominator. */
    val fraction: Double?
    get() = bytesTotal?.takeIf { it > 0 }?.let { (bytesDone.toDouble() / it).coerceIn(0.0, 1.0) }
}

/**
 * #21: background package download as a WorkManager [CoroutineWorker] — wraps [HubDownloader] and reports
 * per-file progress via [setProgress]. The scheduling/constraints (`unmetered`, `storage-not-low`) and the
 * actual on-device run are the manual device leg; the download/verify/install logic itself is the
 * MockWebServer-tested [PackageDownloader] + [HubDownloader].
 */
class PackageDownloadWorker(context: Context, params: WorkerParameters) :
    CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val repoId = inputData.getString(KEY_REPO_ID) ?: return Result.failure()
        val cacheDir = inputData.getString(KEY_CACHE_DIR) ?: return Result.failure()
        val revision = inputData.getString(KEY_REVISION) ?: "main"
        val variant = inputData.getString(KEY_VARIANT)
        val features = inputData.getStringArray(KEY_FEATURES)?.toSet() ?: setOf("inference")
        val genai = inputData.getBoolean(KEY_GENAI, false)
        val endpoint = inputData.getString(KEY_ENDPOINT) ?: HubResolver.DEFAULT_ENDPOINT
        val token = inputData.getString(KEY_TOKEN)

        return try {
            HubDownloader.downloadAndInstall(
                cacheDir = File(cacheDir),
                repoId = repoId,
                revision = revision,
                variant = variant,
                features = features,
                genai = genai,
                // Was omitted entirely, and the omission is silent: HubDownloader falls back to
                // `manifest.defaultVariant` when `abis` is empty, so this path took whatever the
                // publisher listed first regardless of what the phone can run — the exact behaviour
                // removed from `fromPretrained`, still present here because the worker had no
                // caller to expose it. Both paths now read the same DeviceCapabilities.
                abis = DeviceCapabilities.abis(),
                totalMemMb = DeviceCapabilities.totalMemoryMb(applicationContext),
                endpoint = endpoint,
                token = token,
                onProgress = { p ->
                    setProgressAsync(
                        workDataOf(
                            KEY_DONE to p.filesDone,
                            KEY_TOTAL to p.filesTotal,
                            KEY_PATH to p.path,
                            KEY_PHASE to p.phase.name,
                            KEY_BYTES_DONE to p.bytesDone,
                            KEY_BYTES_TOTAL to (p.bytesTotal ?: -1L),
                            KEY_BYTES_PER_SECOND to p.bytesPerSecond,
                        ),
                    )
                },
            )
            Result.success()
        } catch (e: Exception) {
            Result.failure(Data.Builder().putString(KEY_ERROR, e.message).build())
        }
    }

    companion object {
        /**
         * Schedule a background package download and return the enqueued request's id (#21).
         *
         * This was the missing half: the worker was fully written but nothing ever enqueued it, so
         * `androidx.work` was a declared dependency with no caller and background download was
         * unreachable from the SDK. Constraints match the plan — unmetered network + storage-not-low —
         * and the work is unique per repo id so a second request for the same model does not download
         * it twice.
         */
        fun enqueue(
            context: Context,
            repoId: String,
            cacheDir: File,
            revision: String = "main",
            variant: String? = null,
            features: Set<ModelFeature> = setOf(ModelFeature.Inference),
            genai: Boolean = false,
            endpoint: String = HubResolver.DEFAULT_ENDPOINT,
            token: String? = null,
            requireUnmetered: Boolean = true,
        ): UUID {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(
                    if (requireUnmetered) NetworkType.UNMETERED else NetworkType.CONNECTED,
                )
                .setRequiresStorageNotLow(true)
                .build()

            val request = OneTimeWorkRequestBuilder<PackageDownloadWorker>()
                .setConstraints(constraints)
                .setInputData(
                    workDataOf(
                        KEY_REPO_ID to repoId,
                        KEY_CACHE_DIR to cacheDir.absolutePath,
                        KEY_REVISION to revision,
                        KEY_VARIANT to variant,
                    // Converted here, once: a caller passing raw group strings is a caller that
                    // can disagree with `fromPretrained` about what "training" downloads.
                    KEY_FEATURES to DeviceCapabilities.downloadGroups(features).toTypedArray(),
                        KEY_GENAI to genai,
                        KEY_ENDPOINT to endpoint,
                        KEY_TOKEN to token,
                    ),
                )
                .build()

            WorkManager.getInstance(context).enqueueUniqueWork(
                uniqueWorkName(repoId),
                ExistingWorkPolicy.KEEP,
                request,
            )
            return request.id
        }

        /** Stable unique-work name so repeat requests for one repo coalesce. */
        fun uniqueWorkName(repoId: String): String = "mobiletransformers-download:$repoId"

        /** Stop the pull for [repoId]. The staging tree is cleaned by [HubDownloader]'s own finally. */
        fun cancel(context: Context, repoId: String) {
            WorkManager.getInstance(context).cancelUniqueWork(uniqueWorkName(repoId))
        }

        /**
 * The pull for [repoId], already interpreted — the same argument as
 * `TrainingScheduler.observeChunks`.
         *
 * Returning `List<WorkInfo>` would force every consumer to depend on `androidx.work` and to
 * decode this object's own progress keys, which is exactly the coupling that left the worker
 * with no caller: the app could not read a pull it had started without reaching past the
 * facade. The interpretation belongs here, once.
         *
 * `ENQUEUED` is the state whose name explains it worst and matters most — it means "accepted,
 * and its network/storage constraints are not met yet", which with the default
 * `requireUnmetered = true` means "waiting for Wi-Fi", an indefinite and entirely normal wait
 * that otherwise looks like a hang.
 */
        fun observe(context: Context, repoId: String): Flow<List<DownloadJob>> =
        WorkManager.getInstance(context)
.getWorkInfosForUniqueWorkFlow(uniqueWorkName(repoId))
.map { infos -> infos.map { it.toDownloadJob() } }

        private fun WorkInfo.toDownloadJob(): DownloadJob {
            val bytesTotal = progress.getLong(KEY_BYTES_TOTAL, -1L).takeIf { it >= 0 }
            return DownloadJob(
                state = when (state) {
                    WorkInfo.State.ENQUEUED -> DownloadJob.State.WaitingForConstraints
                    WorkInfo.State.RUNNING -> DownloadJob.State.Running
                    WorkInfo.State.SUCCEEDED -> DownloadJob.State.Finished
                    WorkInfo.State.FAILED -> DownloadJob.State.Failed
                    WorkInfo.State.BLOCKED -> DownloadJob.State.Blocked
                    WorkInfo.State.CANCELLED -> DownloadJob.State.Cancelled
                },
                phase = progress.getString(KEY_PHASE),
                filesDone = progress.getInt(KEY_DONE, 0),
                filesTotal = progress.getInt(KEY_TOTAL, 0),
                bytesDone = progress.getLong(KEY_BYTES_DONE, 0L),
                // -1 is the sentinel for "the manifest does not size its files", mirroring
                // DownloadProgress.bytesTotal == null. Reporting it as a real total would render a
                // progress bar running backwards from 100%.
                bytesTotal = bytesTotal,
                bytesPerSecond = progress.getLong(KEY_BYTES_PER_SECOND, 0L),
                // Only present once the work ends; on the failure path it is the reason.
                installedPath = outputData.getString(KEY_PATH),
                error = outputData.getString(KEY_ERROR),
                    )
        }

        const val KEY_REPO_ID = "repoId"
        const val KEY_CACHE_DIR = "cacheDir"
        const val KEY_REVISION = "revision"
        const val KEY_VARIANT = "variant"
        const val KEY_FEATURES = "features"
        const val KEY_GENAI = "genai"
        const val KEY_ENDPOINT = "endpoint"
        const val KEY_TOKEN = "token"
        const val KEY_DONE = "done"
        const val KEY_TOTAL = "total"
        const val KEY_PATH = "path"
        const val KEY_ERROR = "error"
        const val KEY_PHASE = "phase"
        const val KEY_BYTES_DONE = "bytesDone"
        /** `-1` when the manifest does not size its files, mirroring `DownloadProgress.bytesTotal == null`. */
        const val KEY_BYTES_TOTAL = "bytesTotal"
        const val KEY_BYTES_PER_SECOND = "bytesPerSecond"
    }
}

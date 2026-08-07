package com.martinkorelic.mobiletransformers.hub

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import java.io.File
import java.util.UUID

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
                endpoint = endpoint,
                token = token,
                onProgress = { done, total, path ->
                    setProgressAsync(workDataOf(KEY_DONE to done, KEY_TOTAL to total, KEY_PATH to path))
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
            features: Set<String> = setOf("inference"),
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
                        KEY_FEATURES to features.toTypedArray(),
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
    }
}

package com.martinkorelic.mobiletransformers.hub

/**
 * Progress of a Hub package pull (#21), as the public facade reports it.
 *
 * ### Why this exists
 *
 * `HubDownloader.downloadAndInstall` has always taken an `onProgress` callback, but
 * `MobileTransformers.fromPretrained` called it with the default no-op and dropped every update on
 * the floor. So the one thing a first-run user waits on — a multi-hundred-megabyte download — was
 * invisible to anyone using the public API, and a Models screen had no way to show it without
 * reaching around the facade into `hub.HubDownloader` directly. Found while building the showcase
 * app's Models/Hub screen; recorded against #17/#19.
 *
 * ### Why it reports BYTES, not just files
 *
 * The first version counted completed files, and `PackageDownloader` only called it after a whole
 * file finished. A real package's weights are one or two files of 1–4 GB, so the honest rendering of
 * that signal is "0 / 6 files" — unchanged, for ten minutes, with no bytes, no rate and no estimate.
 * There is no way to tell that from a stalled connection, which is exactly the report this fixes:
 * a pull that was working was indistinguishable from one that had hung.
 *
 * Byte totals are free: the manifest already declares `fileSizes` for every file in the plan, so the
 * denominator is known before the first GET rather than discovered at the end.
 *
 * The raw callback's `(Int, Int, String)` triple is wrapped in a named type deliberately: at a public
 * boundary `done`/`total` are trivially transposable, and a lambda signature does not say which is
 * which.
 */
data class DownloadProgress(
    /** Which stage of the pull this update describes. */
    val phase: Phase = Phase.Downloading,
    /** Files fully downloaded and verified. */
    val filesDone: Int = 0,
    /** Files in the resolved download plan. Zero until the plan is known. */
    val filesTotal: Int = 0,
    /** Repo-relative path of the file currently in flight, or the one that just completed. */
    val path: String = "",
    /** Bytes written so far across the whole plan, including bytes resumed from a `.partial`. */
    val bytesDone: Long = 0L,
    /** Bytes the plan declares in total, or `null` when the manifest does not size its files. */
    val bytesTotal: Long? = null,
    /** Recent throughput, exponentially smoothed. Zero until two samples exist. */
    val bytesPerSecond: Double = 0.0,
) {
    /** The stages a pull moves through, so a UI can say what it is waiting on rather than guessing. */
    enum class Phase {
        /** Fetching the manifest and choosing a variant. No large transfer has started. */
        Resolving,
        Downloading,
        /** Hashing a completed file against the manifest's sha256. */
        Verifying,
        /** Publishing the staged tree into the cache. */
        Installing,
    }

    /**
     * Completed fraction in `0.0..1.0`, or `null` when nothing is known yet.
     *
     * Prefers bytes over files: with a two-file plan whose second file is 99% of the package, the
     * file count jumps 0 → 50% → 100% and spends almost the whole download at 50%.
     */
    val fraction: Float?
        get() {
            val total = bytesTotal
            return when {
                total != null && total > 0 -> (bytesDone.toDouble() / total).coerceIn(0.0, 1.0).toFloat()
                filesTotal > 0 -> (filesDone.toFloat() / filesTotal).coerceIn(0f, 1f)
                else -> null
            }
        }

    /** Seconds remaining at the current rate, or `null` without both a total and a measured rate. */
    val etaSeconds: Long?
        get() {
            val total = bytesTotal ?: return null
            if (bytesPerSecond <= 0.0) return null
            val remaining = (total - bytesDone).coerceAtLeast(0L)
            return (remaining / bytesPerSecond).toLong()
        }
}

/**
 * Receives [DownloadProgress] updates during `MobileTransformers.fromPretrained`.
 *
 * A `fun interface` rather than a typealias so Java callers get a real SAM type, matching how the
 * rest of the facade's callbacks (`GenerateCallback`, `TrainCallback`) are shaped.
 */
fun interface DownloadProgressListener {
    fun onProgress(progress: DownloadProgress)
}

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
 * The raw callback's `(Int, Int, String)` triple is wrapped in a named type deliberately: at a public
 * boundary `done`/`total` are trivially transposable, and a lambda signature does not say which is
 * which.
 */
data class DownloadProgress(
    /** Files already installed. */
    val filesDone: Int,
    /** Files in the resolved download plan. Zero until the plan is known. */
    val filesTotal: Int,
    /** Repo-relative path of the file that just completed. */
    val path: String,
) {
    /** Completed fraction in `0.0..1.0`, or `null` while [filesTotal] is still unknown. */
    val fraction: Float?
        get() = if (filesTotal > 0) filesDone.toFloat() / filesTotal else null
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

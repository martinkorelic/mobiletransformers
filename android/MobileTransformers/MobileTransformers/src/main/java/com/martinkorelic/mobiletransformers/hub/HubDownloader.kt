package com.martinkorelic.mobiletransformers.hub

import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest
import com.martinkorelic.mobiletransformers.packages.ModelPackageInstaller
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.VariantSelector
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient

/**
 * #21: manifest-first Hub pull → verify → atomic install, mirroring the Python `hub/pull.py`
 * (`pull_package` + `install_package`). Downloads the manifest first, plans the file list
 * ([DownloadPlanner]), streams + sha256-verifies each file ([PackageDownloader]), then materializes via
 * the existing [ModelPackageInstaller] (atomic rename). Reuses the `packages/` verify/select/install half;
 * this is only the network front-end. `client` is injectable for tests.
 */
object HubDownloader {

    /**
     * How often byte progress reaches the caller.
     *
     * The write loop emits per 64 KB chunk — thousands of times a second on a fast link. Forwarding
     * all of it would drive a Compose recomposition per chunk for no gain, so updates are coalesced
     * to this interval. Small enough that the rate readout stays live, large enough to be free.
     */
    private const val PROGRESS_INTERVAL_MS = 250L

    suspend fun downloadAndInstall(
        cacheDir: File,
        repoId: String,
        revision: String = "main",
        variant: String? = null,
        features: Set<String> = setOf("inference"),
        genai: Boolean = false,
        // #21: device capabilities drive variant selection. Defaults keep this callable from a plain
        // JVM test; production callers pass Build.SUPPORTED_ABIS + ActivityManager memory.
        abis: List<String> = emptyList(),
        totalMemMb: Int? = null,
        quantization: String? = null,
        endpoint: String = HubResolver.DEFAULT_ENDPOINT,
        token: String? = null,
        client: OkHttpClient = PackageDownloader.defaultClient(),
        onProgress: (DownloadProgress) -> Unit = { },
    ): ModelPackageInstaller.Installed =
        withContext(Dispatchers.IO) {
            val sanitized = PackageFormat.sanitizeRepoId(repoId)
            val staging = File(cacheDir, ".download/$sanitized").apply {
                deleteRecursively()
                mkdirs()
            }
            val headers = HubResolver.authHeaders(token)
            val urlFor = { path: String -> HubResolver.fileUrl(endpoint, repoId, revision, path) }

            // Resolving is a real, visible wait on a slow link: the manifest GET plus variant
            // selection happens before a single weight byte moves, and reporting nothing here is what
            // made a pull look dead from the moment it started.
            onProgress(DownloadProgress(phase = DownloadProgress.Phase.Resolving))

            // Manifest first (no large GET precedes it) — no checksum yet (it names the others' checksums).
            PackageDownloader.download(
                client = client,
                files = listOf(PackageFormat.MANIFEST_FILENAME),
                urlFor = urlFor,
                headers = headers,
                expectedSha = emptyMap(),
                destRoot = staging,
            )
            val manifest = MobileTransformersManifest.load(File(staging, PackageFormat.MANIFEST_FILENAME))
            // #21: an explicit `variant` still wins, but otherwise select on DEVICE CAPABILITY rather
            // than blindly taking `manifest.defaultVariant` — that ignored abi, memory, feature and
            // engine constraints, so an incompatible variant downloaded happily and failed at load.
            // VariantSelector throws NoCompatibleVariantException when nothing matches.
            val variantId = variant ?: if (abis.isEmpty()) {
                manifest.defaultVariant
            } else {
                VariantSelector.select(
                    manifest = manifest,
                    abis = abis,
                    quantization = quantization,
                    totalMemMb = totalMemMb,
                    requestedFeatures = features.toList(),
                    requestedEngine = if (genai) "genai" else "native",
                ).id
            }

            val files = DownloadPlanner.planFiles(manifest, variantId, features, genai)

            // The denominator, known BEFORE the first GET: the manifest sizes every file it lists, so
            // there is no reason to discover the package's size by finishing the download. Null when
            // an older manifest omits `fileSizes` — the caller then falls back to counting files.
            val plannedBytes = files.mapNotNull { manifest.fileSizes[it] }
                .takeIf { it.size == files.size }
                ?.sum()

            val reporter = ProgressReporter(
                filesTotal = files.size,
                bytesTotal = plannedBytes,
                emit = onProgress,
            )

            PackageDownloader.download(
                client = client,
                files = files,
                urlFor = urlFor,
                headers = headers,
                expectedSha = manifest.sha256,
                destRoot = staging,
                onProgress = { done, _, path -> reporter.onFileDone(done, path) },
                onBytes = { path, delta, _ -> reporter.onBytes(path, delta) },
                onPhase = { path, phase -> reporter.onPhase(path, phase) },
            )

            reporter.emitNow(DownloadProgress.Phase.Installing)

            // The download staging tree is a FULL SECOND COPY of the package and must not outlive the
            // install. It used to be cleared only by the *next* pull of the same repo, so a 1.3 GB
            // package left 1.3 GB of `.download/` sitting in app storage indefinitely, and a user who
            // pulled two models paid for both. `finally`, not a trailing statement: a failed install
            // is exactly when the device is most likely to be out of space.
            try {
                ModelPackageInstaller.install(
                    stagedPackageDir = staging,
                    cacheDir = cacheDir,
                    repoId = repoId,
                    variantId = variantId,
                    consumeSource = true,
                    // Recorded so the cache can report which repo id and which groups produced this
                    // install; `sanitizeRepoId` is not invertible enough to reconstruct either.
                    features = if (genai) features + "genai" else features,
                )
            } finally {
                staging.deleteRecursively()
            }
        }

    /**
     * Accumulates per-chunk byte deltas into a whole-plan [DownloadProgress], throttled and rate-smoothed.
     *
     * Kept out of [PackageDownloader] on purpose: that object downloads a *list of files* and does not
     * know it is assembling a package, so the plan-level totals and the throttle belong here, with the
     * plan.
     */
    private class ProgressReporter(
        private val filesTotal: Int,
        private val bytesTotal: Long?,
        private val emit: (DownloadProgress) -> Unit,
    ) {
        private val bytesDone = AtomicLong(0)
        private var filesDone = 0
        private var path = ""
        private var phase = DownloadProgress.Phase.Downloading

        private var lastEmitMs = 0L
        private var lastEmitBytes = 0L
        private var smoothedBps = 0.0

        fun onBytes(path: String, delta: Long) {
            this.path = path
            bytesDone.addAndGet(delta)
            maybeEmit()
        }

        fun onPhase(path: String, phase: DownloadProgress.Phase) {
            this.path = path
            // Verifying a 3 GB file takes long enough to look like a hang of its own, so a phase
            // change always emits rather than waiting for the throttle window.
            if (this.phase != phase) {
                this.phase = phase
                emitNow(phase)
            }
        }

        fun onFileDone(done: Int, path: String) {
            filesDone = done
            this.path = path
            emitNow(phase)
        }

        private fun maybeEmit() {
            val now = System.currentTimeMillis()
            if (now - lastEmitMs < PROGRESS_INTERVAL_MS) return
            val done = bytesDone.get()
            if (lastEmitMs > 0L) {
                val seconds = (now - lastEmitMs) / 1000.0
                val instant = if (seconds > 0) (done - lastEmitBytes) / seconds else 0.0
                // Exponential smoothing: a raw per-window rate over mobile radio swings by an order
                // of magnitude between samples, which makes the ETA unreadable.
                smoothedBps = if (smoothedBps == 0.0) instant else 0.7 * smoothedBps + 0.3 * instant
            }
            lastEmitMs = now
            lastEmitBytes = done
            emit(snapshot(phase))
        }

        fun emitNow(phase: DownloadProgress.Phase) {
            this.phase = phase
            lastEmitMs = System.currentTimeMillis()
            lastEmitBytes = bytesDone.get()
            emit(snapshot(phase))
        }

        private fun snapshot(phase: DownloadProgress.Phase) =
            DownloadProgress(
                phase = phase,
                filesDone = filesDone,
                filesTotal = filesTotal,
                path = path,
                bytesDone = bytesDone.get().coerceAtLeast(0L),
                bytesTotal = bytesTotal,
                bytesPerSecond = smoothedBps.coerceAtLeast(0.0),
            )
    }
}

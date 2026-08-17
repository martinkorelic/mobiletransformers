package com.martinkorelic.mobiletransformers.app

import android.content.Context
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.MobileTransformers
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.hub.DownloadJob
import com.martinkorelic.mobiletransformers.hub.PackageDownloadWorker
import com.martinkorelic.mobiletransformers.packages.CacheIndex
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.mapNotNull
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

/**
 * The one loaded [MobileTransformerModel], shared by every screen.
 *
 * ### Why a single holder
 *
 * A model owns a native inference session and, when training, a native training session. Loading the
 * same package twice from two screens would open two sessions over one set of weights, which on device
 * means either wasted RAM or a mutating package underneath a live reader. So the app loads once and
 * every screen observes [state].
 *
 * ### Why this is the app's only load path
 *
 * Everything here goes through the public facade — `MobileTransformers.fromPretrained`,
 * `MobileTransformers.installed`. No `ORT*`, `*Native` or `*Repository` type appears anywhere in this
 * module; `tests/unit/test_guards.py::test_the_sample_app_uses_only_the_public_facade` enforces that,
 * which is what makes "the sample app is a worked example of the public API" checkable rather than
 * claimed.
 */
object ModelHolder {

    /** Guards load/unload so two screens cannot race two native sessions into existence. */
    private val lock = Mutex()

    private val _state = MutableStateFlow<ModelState>(ModelState.None)
    val state: StateFlow<ModelState> = _state.asStateFlow()

    /** The installed packages, refreshed by the Models screen. */
    private val _installed = MutableStateFlow<List<CacheIndex.InstalledPackage>>(emptyList())
    val installed: StateFlow<List<CacheIndex.InstalledPackage>> = _installed.asStateFlow()

    /**
     * The in-flight pull, if any — held here rather than in the Models screen's ViewModel.
     *
     * A download is the app's longest-running operation and the user is free to navigate away from
     * the screen that started it. Keeping the progress in that screen's state meant the model bar
     * (and every other screen) had nothing to show, so leaving Models during a pull looked exactly
     * like no pull running.
     */
    private val _download = MutableStateFlow<DownloadUi?>(null)
    val download: StateFlow<DownloadUi?> = _download.asStateFlow()

    /**
     * What the model is doing right now, for the one indicator every screen shows.
     *
     * The status dot used to be painted from [ModelState] alone, which only distinguishes loaded from
     * not-loaded — so a model mid-generation and a model sitting idle looked identical, and the
     * question the dot exists to answer ("can I ask it something now?") had no answer anywhere in the
     * app. The work itself happens in three different ViewModels, so the flag has to live with the
     * model rather than with any one of them.
     */
    private val _activity = MutableStateFlow(ModelActivity.Idle)
    val activity: StateFlow<ModelActivity> = _activity.asStateFlow()

    /**
     * Run [block] with [activity] set, restoring it afterwards on every path.
     *
     * Nested and concurrent work is not modelled: only one native session exists, and two screens
     * cannot drive it at once. The `finally` matters more than the nesting — a cancelled generation
     * that left the dot red would be worse than no dot.
     */
    suspend fun <T> withActivity(activity: ModelActivity, block: suspend () -> T): T {
        _activity.value = activity
        return try {
            block()
        } finally {
            _activity.value = ModelActivity.Idle
        }
    }

    fun refreshInstalled(context: Context) {
        _installed.value = MobileTransformers.installed(context)
    }

    /**
     * The Hub credentials to pull with, or `null` for an anonymous pull.
     *
     * `null` rather than `HubConfig(token = "")`: an empty token is not "no token", and sending an
     * empty `Authorization: Bearer` header is a different request from sending none. [HubResolver]
     * already treats blank as absent, and this keeps that decision in one place.
     *
     * See `BuildConfig.HF_TOKEN` in the app's `build.gradle.kts` for where the value comes from and
     * why baking one into an APK is a development affordance rather than a shipping pattern.
     */
    private fun hubConfig(): HubConfig? =
        BuildConfig.HF_TOKEN.takeIf { it.isNotBlank() }?.let { HubConfig(token = it) }

    /** Whether this build carries a Hub token — surfaced by the Models screen, never the token itself. */
    val hasHfToken: Boolean get() = BuildConfig.HF_TOKEN.isNotBlank()

    /**
 * Pull [repoId] in the background via WorkManager, then load it.
     *
 * ### Why this exists beside [load]
 *
 * [load] downloads inside the caller's coroutine, so the pull dies with the Activity — on a
 * multi-gigabyte package that is most of an hour of transfer lost to a task switch.
 * `PackageDownloadWorker` was written to solve exactly that and had **no caller**, so the capability
 * shipped and was unreachable.
 *
 * The worker only downloads and installs. Loading still goes through
 * `MobileTransformers.fromPretrained`, which finds the package already in the cache and opens it
 * without touching the network — so there is one load path, not two.
 *
 * @param wifiOnly the worker's `requireUnmetered` constraint. **Default true, and visible in the
 * UI on purpose**: a pull that silently never starts on mobile data is indistinguishable from a
 * hang, which is the trap this switch exists to make legible.
 */
    suspend fun loadInBackground(
        context: Context,
        repoId: String,
        engine: InferenceEngine = InferenceEngine.NATIVE,
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
        wifiOnly: Boolean = true,
) {
        val already = MobileTransformers.installed(context).any { it.repoId == repoId }
        if (already) {
            // Nothing to download; go straight to the shared load path rather than enqueueing a
            // worker that would resolve a manifest and find every file already present.
            load(context, repoId, engine, features)
            return
        }

        _state.value = ModelState.Loading(repoId)
        _activity.value = ModelActivity.Loading
        AppSnackbar.info("Queued $repoId for download")

        PackageDownloadWorker.enqueue(
                context = context,
                repoId = repoId,
            cacheDir = File(context.filesDir.absolutePath),
                features = features,
            genai = engine == InferenceEngine.GENAI,
            token = hubConfig()?.token,
            requireUnmetered = wifiOnly,
                    )

        // `first { it.isTerminal }` rather than a plain collect: the flow stays open for the work's
        // whole retained history, so a collector without a terminal condition never returns.
        val finished = try {
            PackageDownloadWorker.observe(context, repoId)
.onEach { jobs -> jobs.lastOrNull()?.let { publish(it) } }
.mapNotNull { jobs -> jobs.lastOrNull()?.takeIf { it.isTerminal } }
            .first()
        } finally {
            _download.value = null
        }

        when (finished.state) {
            DownloadJob.State.Finished -> load(context, repoId, engine, features)
            DownloadJob.State.Cancelled -> {
            _state.value = ModelState.None
            _activity.value = ModelActivity.Idle
                AppSnackbar.info("Download cancelled — it will resume where it stopped")
        }
            else -> {
                val reason = finished.error ?: "download failed"
            _state.value = ModelState.Failed(repoId, reason)
            _activity.value = ModelActivity.Idle
            AppSnackbar.error(reason)
        }
        }
        }

    /** Stop a background pull. Safe to call when none is running. */
    fun cancelBackgroundDownload(context: Context, repoId: String) {
        PackageDownloadWorker.cancel(context, repoId)
        }

    private fun publish(job: DownloadJob) {
                    _download.value = DownloadUi(
            // `WaitingForConstraints` is the state worth naming: with wifiOnly it means "waiting for
            // Wi-Fi", an indefinite and entirely normal wait that otherwise reads as a stall.
            phase = job.phase ?: job.state.name,
            waitingForConstraints = job.state == DownloadJob.State.WaitingForConstraints,
            filesDone = job.filesDone,
            filesTotal = job.filesTotal,
            path = "",
            fraction = job.fraction?.toFloat(),
            bytesDone = job.bytesDone,
            bytesTotal = job.bytesTotal,
            bytesPerSecond = job.bytesPerSecond.toDouble(),
                    )
        }

    /**
 * Load [repoId], pulling it from the Hub **in the caller's coroutine** when it is not installed.
 *
 * Requests Training as well as Inference only when the package can provide it — asking for a
     * feature the package lacks fails closed at construction (that is the point of
     * `FeatureNotInstalledException`), and a user who pulled an inference-only package should still
     * get a working Chat screen rather than an error.
 *
 * Prefer [loadInBackground] when a download is possible: this one dies with its caller's scope.
     */
    suspend fun load(
        context: Context,
        repoId: String,
        engine: InferenceEngine = InferenceEngine.NATIVE,
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
    ) = lock.withLock {
        _state.value = ModelState.Loading(repoId)
        _activity.value = ModelActivity.Loading
        AppSnackbar.info("Loading $repoId…")
        try {
            unlockedClose()
            val model = MobileTransformers.fromPretrained(
                context = context,
                repoId = repoId,
                engine = engine,
                features = features,
                // Without this the app could only ever pull PUBLIC packages: the facade has always
                // taken a HubConfig, and this screen never passed one, so a private or gated repo was
                // unreachable from the UI even though the whole download stack supported it.
                hubConfig = hubConfig(),
                onDownloadProgress = { p ->
                    _download.value = DownloadUi(
                        phase = p.phase.name,
                        filesDone = p.filesDone,
                        filesTotal = p.filesTotal,
                        path = p.path,
                        fraction = p.fraction,
                        bytesDone = p.bytesDone,
                        bytesTotal = p.bytesTotal,
                        bytesPerSecond = p.bytesPerSecond,
                        etaSeconds = p.etaSeconds,
                    )
                },
            )
            _state.value = ModelState.Loaded(model)
            AppSnackbar.success("Loaded $repoId")
            refreshInstalled(context)
        } catch (e: CancellationException) {
            // The user cancelled the pull. Not a load failure and not an error banner: leave the
            // holder empty and let the caller explain what survived on disk.
            _state.value = ModelState.None
            throw e
        } catch (e: Throwable) {
            // Surfaced verbatim: the SDK's exceptions name the missing feature/artifact, and
            // replacing that with "failed to load" is how an integrator loses the diagnosis.
            val reason = e.message ?: e::class.java.simpleName
            _state.value = ModelState.Failed(repoId, reason)
            AppSnackbar.error(reason)
        } finally {
            _download.value = null
            _activity.value = ModelActivity.Idle
        }
    }

    suspend fun close() = lock.withLock {
        val had = _state.value is ModelState.Loaded
        unlockedClose()
        if (had) AppSnackbar.info("Model unloaded")
    }

    private fun unlockedClose() {
        (_state.value as? ModelState.Loaded)?.model?.close()
        _state.value = ModelState.None
    }
}

/**
 * What the model is occupied with.
 *
 * Separate from [ModelState] on purpose: state answers "which model", activity answers "is it free".
 * A dot painted from state alone can only say loaded/not-loaded, and the question a user actually has
 * in front of a status light is the second one.
 */
enum class ModelActivity(val label: String) {
    Idle("ready"),
    Loading("loading"),
    Generating("generating"),
    Training("training"),
    Merging("merging"),
    Ingesting("ingesting"),
    ;

    /** Whether the native session is occupied — the whole reason this enum exists. */
    val isBusy: Boolean get() = this != Idle
}

/** What the app knows about the model right now. Every screen renders one of these four. */
sealed interface ModelState {
    /** No model loaded — the first thing a new user sees, and the reason Models is the first screen. */
    data object None : ModelState

    data class Loading(val repoId: String) : ModelState

    data class Loaded(val model: MobileTransformerModel) : ModelState

    data class Failed(val repoId: String, val reason: String) : ModelState
}

/**
 * UI-side mirror of the facade's `DownloadProgress`, so composables need no SDK import.
 *
 * Carries bytes and rate, not only a file count. A package's weights are one or two files of
 * gigabytes, so "0 / 6 files" is what a working download looks like for most of its life — the same
 * thing a hung one looks like.
 */
data class DownloadUi(
    val phase: String,
    /**
     * The worker is enqueued and waiting on a constraint — in practice, Wi-Fi.
     *
     * A boolean rather than a magic phase string, because that is exactly how this broke: the phase
     * was set to the human sentence `"waiting for Wi-Fi"`, and `downloadPhaseLabel` — which matches
     * `Resolving`/`Verifying`/`Installing` and sends everything else to `"Downloading"` — swallowed
     * it. The app then showed an active download that never advanced, which is precisely the state
     * the sentence existed to distinguish it from. Two correct halves, one unverified seam.
     */
    val waitingForConstraints: Boolean = false,
    val filesDone: Int,
    val filesTotal: Int,
    val path: String,
    val fraction: Float?,
    val bytesDone: Long = 0L,
    val bytesTotal: Long? = null,
    val bytesPerSecond: Double = 0.0,
    val etaSeconds: Long? = null,
) {
    private fun mb(bytes: Long): String = "%.0f MB".format(bytes / 1_048_576.0)

    /** e.g. `"412 MB / 1,320 MB · 8.4 MB/s · ~2m left"`, degrading as each part becomes unknown. */
    val summary: String
        get() = buildString {
            append(mb(bytesDone))
            bytesTotal?.let { append(" / ${mb(it)}") }
            if (bytesPerSecond > 0) append(" · %.1f MB/s".format(bytesPerSecond / 1_048_576.0))
            etaSeconds?.let { append(" · ~${humanDuration(it)} left") }
        }

    private fun humanDuration(seconds: Long): String = when {
        seconds < 60 -> "${seconds}s"
        seconds < 3600 -> "${seconds / 60}m"
        else -> "${seconds / 3600}h ${(seconds % 3600) / 60}m"
    }
}

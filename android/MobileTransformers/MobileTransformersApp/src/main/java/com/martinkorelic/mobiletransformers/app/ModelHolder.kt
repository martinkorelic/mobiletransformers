package com.martinkorelic.mobiletransformers.app

import android.content.Context
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.MobileTransformers
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.packages.CacheIndex
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

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
     * Load [repoId], pulling it from the Hub first when it is not installed.
     *
     * Requests Training as well as Inference **only when the package can provide it** — asking for a
     * feature the package lacks fails closed at construction (that is the point of
     * `FeatureNotInstalledException`), and a user who pulled an inference-only package should still
     * get a working Chat screen rather than an error.
     */
    suspend fun load(
        context: Context,
        repoId: String,
        engine: InferenceEngine = InferenceEngine.NATIVE,
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
    ) = lock.withLock {
        _state.value = ModelState.Loading(repoId)
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

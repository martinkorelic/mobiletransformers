package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.AppSnackbar
import com.martinkorelic.mobiletransformers.app.DownloadUi
import com.martinkorelic.mobiletransformers.app.ModelCatalog
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * #21/#13 — the Models / Hub screen.
 *
 * **This screen exists because the old sample app could not be used by anyone.** It assumed a package
 * already `adb push`ed into place, which no real user can do, so on a clean install every other screen
 * was dead. Pulling a package by repo id is therefore the app's entry point, not a convenience.
 */
class ModelsViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(ModelsUiState())
    val ui: StateFlow<ModelsUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    /** Held by [ModelHolder], not here, so a pull stays visible after navigating away from Models. */
    val download: StateFlow<DownloadUi?> = ModelHolder.download

    /**
     * Whether this build carries an `HF_TOKEN`, so a private-repo pull that 401s is diagnosable.
     *
     * The boolean, never the token: "you have no credentials" and "your credentials were rejected" are
     * different problems with the same symptom, and a screen that shows neither leaves the user
     * guessing. Rendering the token itself would put it on screen and in screenshots for no benefit.
     */
    val hasHfToken: Boolean get() = ModelHolder.hasHfToken

    init {
        refresh()
    }

    fun refresh() {
        ModelHolder.refreshInstalled(getApplication())
        _ui.value = _ui.value.copy(
            installed = ModelHolder.installed.value.map {
                InstalledRow(
                    repoId = it.repoId,
                    sanitizedRepoId = it.sanitizedRepoId,
                    baseModelId = it.baseModelId,
                    variantIds = it.variantIds,
                    installedVariantId = it.installedVariantId,
                    requestedFeatures = it.requestedFeatures,
                    sizeBytes = it.sizeBytes,
                    hasManifest = it.hasManifest,
                )
            },
        )
    }

    fun onRepoIdChanged(value: String) {
        _ui.value = _ui.value.copy(repoId = value)
    }

    fun onTrainingRequestedChanged(value: Boolean) {
        _ui.value = _ui.value.copy(requestTraining = value)
    }

    /**
     * RAG is a **download-time** decision, not a runtime toggle.
     *
     * The embedding encoder lives in its own `rag` feature group (~91 MB), and `DownloadPlanner` only
     * fetches that group when the feature is requested. Without this the Chat screen's RAG switch was
     * structurally dead: no encoder was ever downloaded, so ingest had nothing to embed with and every
     * grounded query returned zero sources. Asking here, where the cost is visible next to the other
     * groups, is the honest place for it.
     */
    fun onRagRequestedChanged(value: Boolean) {
        _ui.value = _ui.value.copy(requestRag = value)
    }

    fun onWifiOnlyChanged(value: Boolean) {
        _ui.value = _ui.value.copy(wifiOnly = value)
    }

    /**
     * The in-flight pull, so it can be cancelled.
     *
     * Without a handle there was no way to stop a download at all: `viewModelScope.launch` outlives
     * every screen the user can navigate to, so starting a 4 GB pull by mistake meant waiting it out
     * or killing the app. The partial files survive cancellation and `Range`-resume picks them up.
     */
    private var pullJob: Job? = null

    /** Pull-if-absent then load, reporting download progress through the facade's new callback. */
    fun loadSelected(repoId: String = _ui.value.repoId) {
        if (repoId.isBlank()) {
            _ui.value = _ui.value.copy(message = "enter a repo id first, e.g. HuggingFaceTB/SmolLM2-135M-Instruct")
            return
        }
        if (pullJob?.isActive == true) {
            _ui.value = _ui.value.copy(message = "a pull is already running — cancel it first")
            return
        }
        pullJob = viewModelScope.launch {
            _ui.value = _ui.value.copy(message = null)
            val features = buildSet {
                add(ModelFeature.Inference)
                if (_ui.value.requestTraining) add(ModelFeature.Training)
                if (_ui.value.requestRag) add(ModelFeature.Rag)
            }
            try {
                // Background rather than in-scope: the pull now survives leaving the app, which is
                // the whole reason PackageDownloadWorker exists. Loading still happens through the
                // one shared path once the worker reports Finished.
                ModelHolder.loadInBackground(
                    context = getApplication(),
                    repoId = repoId,
                    engine = _ui.value.engine,
                    features = features,
                    wifiOnly = _ui.value.wifiOnly,
                )
            } catch (cancellation: CancellationException) {
                // Cancelling is a user action with a normal outcome, not an error: say what survived.
                _ui.value = _ui.value.copy(
                    message = "download cancelled — partial files are kept, so pulling again resumes " +
                        "from where it stopped",
                )
                AppSnackbar.info("Download cancelled — it will resume where it stopped")
                throw cancellation
            } finally {
                refresh()
            }
        }
    }

    /**
     * Install a catalog entry: request exactly the feature groups it declares, then load it.
     *
     * The entry's own `features` drive the request rather than whatever the manual toggles happen to
     * be set to — a catalog row that says it ships a train stage should install one without the user
     * first discovering that a switch on another tab governs it.
     */
    fun installFromCatalog(entry: ModelCatalog.Entry) {
        _ui.value = _ui.value.copy(
            repoId = entry.repoId,
            requestTraining = entry.supportsTraining,
            requestRag = entry.supportsRag,
        )
        loadSelected(entry.repoId)
    }

    /**
 * Stop the in-flight pull, keeping the `.partial` files so a retry resumes.
 *
 * Cancels the WORKER as well as the coroutine observing it. Cancelling only the coroutine would
 * detach the UI from a download that kept running — which is the failure mode background work
 * introduces and the reason this is not just `pullJob?.cancel` any more.
 */
    fun cancelDownload() {
        ModelHolder.cancelBackgroundDownload(getApplication(), _ui.value.repoId)
        pullJob?.cancel()
        pullJob = null
    }

    fun onEngineChanged(engine: InferenceEngine) {
        _ui.value = _ui.value.copy(engine = engine)
    }

    fun unload() {
        viewModelScope.launch {
            ModelHolder.close()
            refresh()
        }
    }
}

data class ModelsUiState(
    val repoId: String = "HuggingFaceTB/SmolLM2-135M-Instruct",
    val requestTraining: Boolean = false,
    val requestRag: Boolean = false,
    /**
 * Constrain the background pull to unmetered networks.
 *
 * Visible rather than implicit: `PackageDownloadWorker` defaults `requireUnmetered` to true, so on
 * mobile data the work sits in `ENQUEUED` indefinitely and looks exactly like a hang. A switch is
 * the difference between "waiting for Wi-Fi" and "broken".
 */
    val wifiOnly: Boolean = true,
    /**
     * The engine to load with. Was hardcoded to `NATIVE` at the call site, which made the Chat
     * screen's engine picker decorative — GenAI could be *reported* as available and never selected.
     */
    val engine: InferenceEngine = InferenceEngine.NATIVE,
    val installed: List<InstalledRow> = emptyList(),
    val message: String? = null,
) {
    /** The empty state the whole app hangs off: nothing installed, nothing to do but pull. */
    val isEmpty: Boolean get() = installed.isEmpty()
}

data class InstalledRow(
    /**
     * The repo id to load this row with.
     *
     * Emphatically NOT [baseModelId]. This screen used to load `baseModelId ?: sanitizedRepoId`, and
     * the manifest's `baseModelId` names the model a package was exported *from*, not the repo it was
     * pulled *from* — so tapping Load on `mobiletransformers/functiongemma-270m-it` asked for
     * `google/functiongemma-270m-it`, resolved to a cache directory that does not exist, and reported
     * the package as not installed while it sat one directory over. `CacheIndex` now records the
     * installing repo id, and this is it.
     */
    val repoId: String,
    val sanitizedRepoId: String,
    val baseModelId: String?,
    val variantIds: List<String>,
    val installedVariantId: String? = null,
    val requestedFeatures: List<String> = emptyList(),
    val sizeBytes: Long,
    val hasManifest: Boolean,
) {
    val sizeMb: Long get() = sizeBytes / (1024 * 1024)

    /**
     * A package without a manifest is a legacy or hand-pushed directory. It still loads, but variant
     * selection and capability reporting have nothing to read — worth showing rather than hiding,
     * because it explains why such a package offers no variants.
     */
    val subtitle: String
        get() = buildString {
            append("${sizeMb} MB")
            baseModelId?.let { append(" · base: $it") }
            when {
                installedVariantId != null -> append(" · variant: $installedVariantId")
                variantIds.isNotEmpty() -> append(" · variants: ${variantIds.joinToString(", ")}")
            }
            if (requestedFeatures.isNotEmpty()) append(" · pulled with: ${requestedFeatures.joinToString(", ")}")
            if (!hasManifest) append(" · no manifest (legacy layout)")
        }
}

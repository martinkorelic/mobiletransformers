package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.DownloadUi
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
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

    init {
        refresh()
    }

    fun refresh() {
        ModelHolder.refreshInstalled(getApplication())
        _ui.value = _ui.value.copy(
            installed = ModelHolder.installed.value.map {
                InstalledRow(
                    sanitizedRepoId = it.sanitizedRepoId,
                    baseModelId = it.baseModelId,
                    variantIds = it.variantIds,
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

    /** Pull-if-absent then load, reporting download progress through the facade's new callback. */
    fun loadSelected(repoId: String = _ui.value.repoId) {
        if (repoId.isBlank()) {
            _ui.value = _ui.value.copy(message = "enter a repo id first, e.g. HuggingFaceTB/SmolLM2-135M-Instruct")
            return
        }
        viewModelScope.launch {
            _ui.value = _ui.value.copy(message = null, download = null)
            val features = buildSet {
                add(ModelFeature.Inference)
                if (_ui.value.requestTraining) add(ModelFeature.Training)
            }
            ModelHolder.load(
                context = getApplication(),
                repoId = repoId,
                engine = InferenceEngine.NATIVE,
                features = features,
                onDownloadProgress = { p -> _ui.value = _ui.value.copy(download = p) },
            )
            _ui.value = _ui.value.copy(download = null)
            refresh()
        }
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
    val installed: List<InstalledRow> = emptyList(),
    val download: DownloadUi? = null,
    val message: String? = null,
) {
    /** The empty state the whole app hangs off: nothing installed, nothing to do but pull. */
    val isEmpty: Boolean get() = installed.isEmpty()
}

data class InstalledRow(
    val sanitizedRepoId: String,
    val baseModelId: String?,
    val variantIds: List<String>,
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
            append(baseModelId ?: "unknown base model")
            append(" · ${sizeMb} MB")
            if (variantIds.isNotEmpty()) append(" · variants: ${variantIds.joinToString(", ")}")
            if (!hasManifest) append(" · no manifest (legacy layout)")
        }
}

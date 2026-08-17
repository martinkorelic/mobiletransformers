package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.ModelActivity
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.runtime.LabelScore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * The encoder story's payoff: ask a fine-tuned classifier something and see the class it picks.
 *
 * The encoder path is complete — `text-classification` export, a trainable head, `classify()` on
 * the facade, `ClassifierSession` under it — and then had nowhere to show it. A classifier could be
 * pulled and fine-tuned on device and never asked a single question, which made the encoder work
 * unfalsifiable from the app.
 *
 * ### Why the screen keeps the previous result
 *
 * [ClassifyUiState.previous] holds the last run's scores while a new one is in flight. The interesting
 * comparison for this screen is *before versus after training the head*, and a screen that blanks on
 * every submit makes the user hold both distributions in their head. Keeping the last one on screen is
 * the cheap version of the before/after the encoder story wants.
 */
class ClassifyViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(ClassifyUiState())
    val ui: StateFlow<ClassifyUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    fun onTextChanged(value: String) {
        _ui.value = _ui.value.copy(text = value)
    }

    fun submit() {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        val text = _ui.value.text.trim()
        if (text.isEmpty() || _ui.value.running) return

        viewModelScope.launch {
            _ui.value = _ui.value.copy(
                running = true,
                error = null,
                // Demote rather than discard, so the comparison survives the next run.
                previous = _ui.value.scores.takeIf { it.isNotEmpty() },
                previousText = _ui.value.classifiedText,
            )
            try {
                val result = ModelHolder.withActivity(ModelActivity.Generating) {
                    model.classify(text = text, topK = TOP_K)
                }
                _ui.value = _ui.value.copy(
                    scores = result.top,
                    classifiedText = text,
                )
            } catch (e: Throwable) {
                // Named, not swallowed: the most likely failure here is a package whose head has no
                // id2label, and `classify()` says exactly that. Paraphrasing loses the diagnosis.
                _ui.value = _ui.value.copy(error = e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(running = false)
            }
        }
    }

    fun clear() {
        _ui.value = ClassifyUiState(text = _ui.value.text)
    }

    companion object {
        /** Enough to show a distribution rather than just a winner; a head with fewer returns fewer. */
        const val TOP_K = 5
    }
}

data class ClassifyUiState(
    val text: String = "The battery lasts all day and the screen is gorgeous.",
    val running: Boolean = false,
    /** Highest probability first — `ClassificationResult.top` is already sorted. */
    val scores: List<LabelScore> = emptyList(),
    /** The input [scores] describes, so the result cannot silently re-label edited text. */
    val classifiedText: String = "",
    val previous: List<LabelScore>? = null,
    val previousText: String = "",
    val error: String? = null,
) {
    val best: LabelScore? get() = scores.firstOrNull()
}

package com.martinkorelic.mobiletransformers.app

import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

/**
 * One place for "something just happened", shown as a transient banner at the top of every screen.
 *
 * ### Why the app needed one
 *
 * Every outcome used to land in a card somewhere down the screen the user happened to be on — and
 * most of the interesting ones happen *while they are somewhere else*. Starting a pull on Models and
 * switching to Chat meant the download finishing, or failing, in silence. Ingest, merge, schedule and
 * every SDK refusal had the same shape: state written into a `ui.message` field that only one screen
 * renders, often below the fold.
 *
 * A shared flow rather than per-screen state because the events outlive the screen that caused them.
 *
 * ### Why events are dropped rather than queued
 *
 * A snackbar is a courtesy, never the only record: every one of these outcomes is also visible in the
 * durable UI — the model bar, the events list, the error card. Suspending a training loop because a
 * banner has nowhere to go would be exactly backwards, so the buffer drops its oldest entry instead.
 */
object AppSnackbar {

    private val _events = MutableSharedFlow<SnackbarEvent>(
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: SharedFlow<SnackbarEvent> = _events.asSharedFlow()

    fun info(message: String) = emit(SnackbarEvent(message, Severity.Info))

    fun success(message: String) = emit(SnackbarEvent(message, Severity.Success))

    /**
     * A failure the user should see wherever they are.
     *
     * The SDK's exception messages name the missing feature or artifact, so they are passed through
     * verbatim; paraphrasing them to fit a banner is how the diagnosis gets lost.
     */
    fun error(message: String) = emit(SnackbarEvent(message, Severity.Error))

    private fun emit(event: SnackbarEvent) {
        _events.tryEmit(event)
    }

    enum class Severity { Info, Success, Error }

    data class SnackbarEvent(
        val message: String,
        val severity: Severity,
        /** Optional label for a single action, e.g. "Cancel". */
        val actionLabel: String? = null,
        val onAction: (() -> Unit)? = null,
    )
}

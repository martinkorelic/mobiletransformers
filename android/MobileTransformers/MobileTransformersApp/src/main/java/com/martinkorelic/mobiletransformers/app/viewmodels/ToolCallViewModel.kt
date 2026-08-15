package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.agent.ActionSpec
import com.martinkorelic.mobiletransformers.agent.FunctionCallValidator
import com.martinkorelic.mobiletransformers.agent.ToolCallResult
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * #37 — the screen that carries the differentiation argument.
 *
 * instruction → `generateToolCall` → **Accepted or Rejected as a first-class outcome** → dry-run intent
 * with `willExecute = false` visible.
 *
 * ### Two things this screen must not do
 *
 * **It must not execute.** `IntentBinder.dryRun` returns an intent the app *would* fire; nothing here
 * calls `startActivity`. The safety property is structural, not a promise: only `FunctionCallValidator`
 * can build a `ValidatedCall`, only a `ValidatedCall` reaches the binder, and the intent string comes
 * from the app's own [ActionSpec] — **a model selects an action, it cannot name an intent** — so the
 * reachable intent set is fixed when the allowlist below is written.
 *
 * **It must not hide a refusal.** A rejection is the expected answer for untrusted output and is
 * rendered as a result, not an error — and on a base model that has not been fine-tuned on this
 * allowlist it is the *expected* answer, not a bug in the screen. Run the Train tab first.
 *
 * *(This block previously said the device gate "still fails … expect this screen to show Rejected
 * until that is fixed". That is now false: `ToolCallDeviceTest` PASSES as of 2026-08-14 — 2 tests /
 * 0 failures / 754 s on an S21 FE — once the merge-transpose defect was fixed. The model had been
 * learning the task all along and the merge was corrupting the result on the way out.)*
 */
class ToolCallViewModel(app: Application) : AndroidViewModel(app) {

    /**
     * The app's declaration — the only source of intent strings. Generating a training set from this
     * same object is what makes the corpus and the boundary provably one value
     * (`mobiletransformers agent-dataset` writes it as `action_schema.json`).
     */
    val allowlist = listOf(
        ActionSpec(
            actionName = "set_alarm",
            parameters = mapOf("time" to "string"),
            allowedIntent = "android.intent.action.SET_ALARM",
            validationRules = mapOf("time" to "HH:mm"),
            privacyClass = "harmless-demo",
        ),
        ActionSpec(
            actionName = "set_timer",
            parameters = mapOf("seconds" to "string"),
            allowedIntent = "android.intent.action.SET_TIMER",
            validationRules = mapOf("seconds" to "/[0-9]{1,4}/"),
            privacyClass = "harmless-demo",
        ),
        ActionSpec(
            actionName = "open_wifi_settings",
            parameters = emptyMap(),
            allowedIntent = "android.settings.WIFI_SETTINGS",
            privacyClass = "harmless-demo",
        ),
    )

    private val validator = FunctionCallValidator(allowlist)

    private val _ui = MutableStateFlow(ToolCallUiState(allowedActions = validator.allowedActions))
    val ui: StateFlow<ToolCallUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    fun onInstructionChanged(value: String) {
        _ui.value = _ui.value.copy(instruction = value)
    }

    fun submit() {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        val instruction = _ui.value.instruction.trim()
        if (instruction.isEmpty() || _ui.value.running) return

        viewModelScope.launch {
            _ui.value = _ui.value.copy(running = true, outcome = null, error = null)
            try {
                when (val result = model.generateToolCall(
                    instruction = instruction,
                    validator = validator,
                    config = AppConfig.generation.value,
                )) {
                    is ToolCallResult.Accepted -> {
                        val intended = result.dryRun()
                        _ui.value = _ui.value.copy(
                            outcome = Outcome.Accepted(
                                raw = result.raw,
                                actionName = result.call.actionName,
                                parameters = result.call.parameters,
                                intentAction = intended.intent.action ?: "(none)",
                                willExecute = intended.willExecute,
                            ),
                        )
                    }
                    is ToolCallResult.Rejected ->
                        _ui.value = _ui.value.copy(
                            outcome = Outcome.Rejected(raw = result.raw, reason = result.reason),
                        )
                }
            } catch (e: Throwable) {
                _ui.value = _ui.value.copy(error = e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(running = false)
            }
        }
    }
}

data class ToolCallUiState(
    val instruction: String = "wake me at 07:30",
    val allowedActions: Set<String> = emptySet(),
    val running: Boolean = false,
    val outcome: Outcome? = null,
    val error: String? = null,
)

/** Accepted and Rejected are peers. A refusal is a result, not a failure. */
sealed interface Outcome {
    data class Accepted(
        val raw: String,
        val actionName: String,
        val parameters: Map<String, String>,
        val intentAction: String,
        val willExecute: Boolean,
    ) : Outcome

    data class Rejected(val raw: String, val reason: String) : Outcome
}

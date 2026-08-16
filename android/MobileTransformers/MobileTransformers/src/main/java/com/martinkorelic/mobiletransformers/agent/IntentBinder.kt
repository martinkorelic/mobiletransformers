package com.martinkorelic.mobiletransformers.agent

import android.content.Intent

/**
 * The intent an accepted call *would* produce, and whether anything is allowed to run it.
 *
 * @property willExecute always `false` from [IntentBinder.dryRun]. The flag exists so a caller that
 *   chooses to execute has to read it and act on it, rather than executing because the object happened
 *   to contain an [Intent].
 */
data class IntendedAction(
    val intent: Intent,
    val willExecute: Boolean = false,
    /**
 * Permissions the caller must hold before starting [intent] — copied from the app's own
 * [ActionSpec], never from model output.
 *
 * Carried on the result so a caller can check and request them *before* firing. Without it the
 * only way to discover a missing permission was to call `startActivity` and catch
 * `SecurityException`, which is an exception used as a question.
 */
    val requiredPermissions: List<String> = emptyList(),
)

/**
 * Builds an Android [Intent] from a [ValidatedCall] **without executing it**.
 *
 * #37's hard requirement is "no arbitrary model output is ever executed". Two things enforce it here,
 * neither of them a convention:
 *
 * 1. **The type.** `dryRun` accepts only [ValidatedCall], which nothing but [FunctionCallValidator]
 *    can construct. Raw model text cannot reach this function at all.
 * 2. **The action string.** It is read from `spec.allowedIntent` — the APP's declaration — never from
 *    the model's output. A model cannot name an intent, only select an action the app already
 *    permitted, so the reachable set of intents is fixed at allowlist-construction time.
 *
 * This class does not hold a `Context` and never calls `startActivity`. Executing an intent is the
 * caller's decision, made with the caller's own `Context`, after reading [IntendedAction.willExecute]
 * — deliberately not offered here as a convenience, because the convenience is the risk.
 */
object IntentBinder {

    /**
     * @return the intent this call describes, marked as not-to-be-executed.
     *
     * Parameters become string extras under their declared names. They are already checked against
     * `validationRules`, and the key set is exactly what the [ActionSpec] declares — the validator
     * rejects both unknown and missing parameters — so no model-chosen key reaches the extras bundle.
     */
    fun dryRun(call: ValidatedCall): IntendedAction {
        val intent = Intent(call.allowedIntent)
        for ((key, value) in call.parameters) {
            intent.putExtra(key, value)
        }
        return IntendedAction(
            intent = intent,
            willExecute = false,
            requiredPermissions = call.requiredPermissions,
)
    }
}

package com.martinkorelic.mobiletransformers.agent

import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.martinkorelic.mobiletransformers.MobileTransformersException

/**
 * Rejected because the model asked for something the app did not declare, or asked for it wrongly.
 *
 * A distinct type because rejection is the **expected** outcome for untrusted output, not a defect —
 * callers route it to "I can't do that" rather than to error reporting.
 */
class RejectedCallException(message: String) : MobileTransformersException(message)

/**
 * What an app declares a model is allowed to ask for. One row per action.
 *
 * `allowedIntent` is the ONLY intent this action can ever produce — it comes from the app's own
 * declaration, never from model output. That is what makes the validator a boundary rather than a
 * formatter: a model cannot name an intent, only an action the app already permitted.
 *
 * @property validationRules parameter name -> rule. Two forms are supported, both deliberately small:
 *   `"HH:mm"` (a literal time-of-day format) and `"/<regex>/"`. Anything richer belongs in the caller's
 *   own check after validation, not in a mini-language here — an expressive rule DSL parsed from app
 *   config is another place for a mistake to hide.
 * @property privacyClass documentation for the app's own review (e.g. `"harmless-demo"`); it is not
 *   interpreted here, and is present so an allowlist can be audited without reading code.
 */
data class ActionSpec(
    val actionName: String,
    val parameters: Map<String, String> = emptyMap(),
    val allowedIntent: String,
    val validationRules: Map<String, String> = emptyMap(),
    val privacyClass: String = "unspecified",
)

/** The raw shape parsed out of model output. Untrusted until [FunctionCallValidator] accepts it. */
internal data class ToolCall(
    val actionName: String? = null,
    val parameters: Map<String, String>? = null,
)

/**
 * A call that passed every check, paired with the app's own spec for it.
 *
 * Construction is the proof: nothing else in this package produces one, so holding a [ValidatedCall]
 * means the allowlist and the rules were satisfied. [IntentBinder] takes only this type, which is what
 * makes "no arbitrary model output is ever executed" a property of the types rather than of a habit.
 */
data class ValidatedCall(
    val spec: ActionSpec,
    val parameters: Map<String, String>,
) {
    val actionName: String get() = spec.actionName
    val allowedIntent: String get() = spec.allowedIntent
}

/**
 * Turns raw model output into a [ValidatedCall], or refuses.
 *
 * #37's safety contract, stated as one rule: **the model chooses among what the app already declared;
 * it never introduces anything.** Every field that reaches Android — the intent action above all —
 * comes from [ActionSpec], and the only thing taken from the model is *which* action and *which*
 * parameter values, both checked before they are handed on.
 *
 * Gson because it is the module's single JSON library (per the typed fail-closed parsing decision) and
 * already a dependency.
 */
class FunctionCallValidator(allowlist: List<ActionSpec>) {

    private val byName: Map<String, ActionSpec> = allowlist.associateBy { it.actionName }

    init {
        // A duplicated action name means two rows disagree about what is permitted and `associateBy`
        // silently keeps the last. Fail at construction, where the allowlist is visible.
        require(byName.size == allowlist.size) {
            val dupes = allowlist.map { it.actionName }.groupBy { it }.filterValues { it.size > 1 }.keys
            "duplicate action names in the allowlist: $dupes"
        }
    }

    /** Action names this validator will accept, for diagnostics and tests. */
    val allowedActions: Set<String> get() = byName.keys

    /**
     * @throws RejectedCallException on anything that is not a well-formed, allowlisted, rule-satisfying
     *   call. The message names the offending entity — an error that says only "invalid" costs an
     *   export→push→run cycle to diagnose.
     */
    fun validate(raw: String): ValidatedCall {
        val call = try {
            Gson().fromJson(raw, ToolCall::class.java)
        } catch (e: JsonSyntaxException) {
            throw RejectedCallException("model output is not valid JSON: ${e.message}")
        } ?: throw RejectedCallException("model output is empty")

        val name = call.actionName
            ?: throw RejectedCallException("model output has no 'actionName' field")

        // The allowlist check comes BEFORE anything is done with the parameters, so an unknown action
        // cannot reach any other code path.
        val spec = byName[name]
            ?: throw RejectedCallException(
                "action not allowlisted: '$name' (allowed: ${byName.keys.sorted()})"
            )

        val supplied = call.parameters ?: emptyMap()

        val unknown = supplied.keys - spec.parameters.keys
        if (unknown.isNotEmpty()) {
            throw RejectedCallException(
                "action '$name' does not declare parameter(s) ${unknown.sorted()} " +
                    "(declared: ${spec.parameters.keys.sorted()})"
            )
        }

        val missing = spec.parameters.keys - supplied.keys
        if (missing.isNotEmpty()) {
            throw RejectedCallException("action '$name' is missing parameter(s) ${missing.sorted()}")
        }

        for ((param, rule) in spec.validationRules) {
            val value = supplied[param] ?: continue
            if (!matches(rule, value)) {
                throw RejectedCallException(
                    "action '$name' parameter '$param' value '$value' does not satisfy rule '$rule'"
                )
            }
        }

        return ValidatedCall(spec = spec, parameters = supplied)
    }

    private fun matches(rule: String, value: String): Boolean = when {
        rule == "HH:mm" -> HH_MM.matches(value)
        rule.length >= 2 && rule.startsWith("/") && rule.endsWith("/") ->
            // An unparseable regex in the APP's own allowlist is a bug in the app, not untrusted input,
            // so it surfaces rather than silently rejecting every call.
            Regex(rule.substring(1, rule.length - 1)).matches(value)
        // An unrecognised rule must NOT pass by default: a typo in the allowlist would otherwise
        // silently disable the check it was written to perform.
        else -> false
    }

    private companion object {
        val HH_MM = Regex("^([01]\\d|2[0-3]):[0-5]\\d$")
    }
}

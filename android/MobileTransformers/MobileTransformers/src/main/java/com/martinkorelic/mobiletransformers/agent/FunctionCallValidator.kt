package com.martinkorelic.mobiletransformers.agent

import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.martinkorelic.mobiletransformers.MobileTransformersException
import java.io.File

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
    /**
     * Parameters that MUST be present. `null` means "all declared ones", which is what a hand-written
     * allowlist means and what this validator enforced before optional parameters existed.
     *
     * Real tool schemas separate the two. In `google/mobile-actions`, `send_email` declares
     * `subject`/`body`/`to` but requires only `to`/`subject`; `create_contact` requires 2 of 4.
     * Treating every declared parameter as required would reject calls that corpus considers correct
     * — the model would be trained toward targets its own validator refuses.
     */
    val requiredParameters: Set<String>? = null,
) {
    /** The effective required set: all declared parameters unless [requiredParameters] narrows it. */
    val required: Set<String> get() = requiredParameters ?: parameters.keys
}

/**
 * The raw shape parsed out of model output. Untrusted until [FunctionCallValidator] accepts it.
 *
 * Public because the format a model speaks is a property of the model, not of the boundary: a
 * [ToolCallParser] produces one of these from JSON, from FunctionGemma's `call:` grammar, or from
 * whatever a future model emits, and the validator then judges it identically. Holding one means
 * nothing has been checked yet — [ValidatedCall] is the type that carries a decision.
 */
data class ToolCall(
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
class FunctionCallValidator(
    /**
     * What the app permits — readable so the same object can be *declared to the model*.
     *
     * [ToolPromptBuilder] renders it into the tool declaration a prompt carries. Generating the
     * declaration from the enforcement list, rather than writing it out beside it, is the same
     * argument that generates the training corpus from it: three copies of the boundary would be
     * three chances for them to disagree, and the disagreement surfaces as an unexplained refusal.
     */
    val allowlist: List<ActionSpec>,
) {

    companion object {
        private val HH_MM = Regex("^([01]\\d|2[0-3]):[0-5]\\d$")

        /** The filename `mobiletransformers agent-dataset` writes beside the training JSONL. */
        const val ACTION_SCHEMA_FILENAME = "action_schema.json"

        /**
         * Build a validator from the action schema emitted next to the training set.
         *
         * The point of loading rather than hard-coding: the schema comes out of the SAME command that
         * produced the training rows, so the boundary the model was trained toward and the boundary
         * enforced here are one artifact. A hand-written allowlist beside a generated dataset is a
         * drift waiting to happen.
         */
        @JvmStatic
        fun fromSchema(file: File): FunctionCallValidator {
            if (!file.isFile) throw RejectedCallException("no action schema at ${file.path}")
            val specs = try {
                Gson().fromJson(file.readText(Charsets.UTF_8), Array<ActionSpec>::class.java)
            } catch (e: JsonSyntaxException) {
                throw RejectedCallException("${file.path} is not a valid action schema: ${e.message}")
            } ?: throw RejectedCallException("${file.path} parsed to null")
            return FunctionCallValidator(specs.toList())
        }
    }

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
        return validate(call)
    }

    /**
     * Judge an already-parsed [call].
     *
     * Every check below runs on this path, so which [ToolCallParser] produced the call cannot change
     * what is permitted — only whether a call was recognised at all. That separation is what let
     * FunctionGemma's non-JSON grammar be supported without touching the boundary.
     *
     * @throws RejectedCallException on anything that is not an allowlisted, rule-satisfying call.
     */
    fun validate(call: ToolCall): ValidatedCall {
        // Names the parsed field rather than a JSON key: the same check now runs on calls that never
        // were JSON, and telling a FunctionGemma user their output lacks an "actionName" field would
        // point them at a field their model has no way to emit.
        val name = call.actionName
            ?: throw RejectedCallException("model output names no action (ToolCall.actionName is null)")

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

        // Against `required`, not every declared parameter — an optional one may legitimately be absent.
        val missing = spec.required - supplied.keys
        if (missing.isNotEmpty()) {
            throw RejectedCallException("action '$name' is missing required parameter(s) ${missing.sorted()}")
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

}

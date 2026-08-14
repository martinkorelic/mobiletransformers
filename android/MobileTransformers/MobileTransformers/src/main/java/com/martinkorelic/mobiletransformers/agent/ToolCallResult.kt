package com.martinkorelic.mobiletransformers.agent

/**
 * The outcome of asking a model for a tool call (#37).
 *
 * Sealed, with the intent reachable **only** through [Accepted], so the safety contract is a property
 * of the type rather than of caller discipline: there is no path from raw model output to an
 * `IntendedAction` that does not pass through [FunctionCallValidator].
 *
 * [Rejected] is a first-class outcome, not an error. Refusing is the *expected* answer for untrusted
 * output — a UI shows "I can't do that", it does not show a crash — and modelling it as a value keeps
 * the raw text available for display and debugging instead of losing it inside an exception.
 */
sealed interface ToolCallResult {

    /** Exactly what the model emitted, before any extraction or validation. Always available. */
    val raw: String

    /** The call passed the allowlist and every `validationRules` check. */
    data class Accepted(
        override val raw: String,
        val call: ValidatedCall,
    ) : ToolCallResult {

        /**
         * The Android intent this call describes, marked not-to-be-executed.
         *
         * A method rather than a field so nothing constructs an `Intent` on a JVM unit-test classpath
         * that stubs it — the accepted/rejected logic stays testable without Robolectric, and only a
         * caller that actually wants the intent pays for the framework.
         */
        fun dryRun(): IntendedAction = IntentBinder.dryRun(call)
    }

    /**
     * The output was not a call this app permits.
     *
     * @property reason the validator's message, which names the offending entity (an unknown action,
     *   an undeclared parameter, a value failing its rule) rather than saying only "invalid".
     */
    data class Rejected(
        override val raw: String,
        val reason: String,
    ) : ToolCallResult
}

/**
 * Pull the first balanced JSON object out of free-form model text.
 *
 * A fine-tuned model still commonly wraps its answer in prose or a code fence. Handing the whole
 * string to Gson would fail as a *syntax* error and report "not valid JSON" for output that contains a
 * perfectly good call — a diagnosis that costs a device run to see through.
 *
 * **This does not weaken the boundary.** Extraction only chooses which substring to validate; every
 * allowlist and rule check then runs on it unchanged, so no text this finds can produce an action the
 * app did not declare. It is brace-counting, not parsing: string literals are respected (so a `}` inside
 * a value does not end the object early) along with their escapes. Returns the input untouched when
 * there is no balanced object, letting the validator report the real problem.
 */
internal fun extractFirstJsonObject(text: String): String {
    val start = text.indexOf('{')
    if (start < 0) return text
    var depth = 0
    var inString = false
    var escaped = false
    for (i in start until text.length) {
        val c = text[i]
        when {
            escaped -> escaped = false
            c == '\\' && inString -> escaped = true
            c == '"' -> inString = !inString
            inString -> Unit
            c == '{' -> depth++
            c == '}' -> {
                depth--
                if (depth == 0) return text.substring(start, i + 1)
            }
        }
    }
    return text
}

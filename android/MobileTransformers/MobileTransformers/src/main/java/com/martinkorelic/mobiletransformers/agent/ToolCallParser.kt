package com.martinkorelic.mobiletransformers.agent

import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.martinkorelic.mobiletransformers.packages.ToolCallDialect

/**
 * Turns raw model text into a candidate [ToolCall], for [FunctionCallValidator] to judge.
 *
 * ### Why parsing is separate from validation
 *
 * [FunctionCallValidator] parsed JSON itself, which quietly made "the model emits JSON" part of the
 * safety boundary's contract. It is not — it is a property of one model family. **FunctionGemma, the
 * model this app's tool-calling story is built around, does not emit JSON at all**: it emits
 * `<start_function_call>call:name{key:<escape>value<escape>}<end_function_call>`. Handed to a JSON
 * parser that is a syntax error, so a correct, well-formed call from a correctly fine-tuned model was
 * reported as "model output is not valid JSON" — and no amount of further fine-tuning could have
 * changed that.
 *
 * Splitting the two makes the format a parameter and leaves the boundary exactly where it was.
 *
 * ### This does not weaken the boundary
 *
 * A parser only decides **which candidate** to check. Every allowlist and `validationRules` check then
 * runs on it unchanged, and the intent string still comes from the app's own [ActionSpec] — so no
 * parser, however wrong, can produce an action the app did not declare. The worst a bad parser can do
 * is fail to recognise a call.
 */
fun interface ToolCallParser {

    /** The call named by [raw], or `null` when it contains none. */
    fun parse(raw: String): ToolCall?

    companion object {
        /**
         * `{"actionName": "...", "parameters": {...}}`, optionally wrapped in prose or a code fence.
         *
         * The historical default, and the right one for a model fine-tuned on this repo's
         * `mobile_actions` corpus, whose completions are exactly this shape.
         */
        @JvmField
        val Json: ToolCallParser = JsonToolCallParser

        /** Google's FunctionGemma tool-call grammar. */
        @JvmField
        val FunctionGemma: ToolCallParser = FunctionGemmaToolCallParser

        /**
         * The parser for a dialect detected from the package itself.
         *
         * Prefer this over [forModel]. The dialect comes from
         * [com.martinkorelic.mobiletransformers.packages.ToolCallSupport], which reads the model's own
         * chat template rather than pattern-matching a name.
         */
        @JvmStatic
        fun forDialect(dialect: ToolCallDialect): ToolCallParser = when (dialect) {
            ToolCallDialect.FUNCTION_GEMMA -> FunctionGemma
            ToolCallDialect.JSON -> Json
        }

        /**
         * The parser suited to a package, guessed from any names known for the model.
         *
         * A guess, and the weaker of the two signals — [forDialect] reads the artifact. Kept because
         * a package whose chat template did not survive export has nothing else to go on.
         *
         * **Takes every hint, not one.** The single-argument version was called as
         * `forModel(task.modelType ?: repoId)`, and `modelType` is the *architecture*
         * (`gemma3_text`) — non-null for every modern package, so the repo id that actually carries
         * the family name was never reached. FunctionGemma got the JSON parser and every well-formed
         * call it made was reported as "no tool call found".
         */
        @JvmStatic
        fun forModel(vararg hints: String?): ToolCallParser =
            if (hints.any { it?.contains("functiongemma", ignoreCase = true) == true }) {
                FunctionGemma
            } else {
                Json
            }
    }
}

/** Extracts a balanced JSON object and reads `actionName` / `parameters` off it. */
private object JsonToolCallParser : ToolCallParser {
    override fun parse(raw: String): ToolCall? {
        val candidate = extractFirstJsonObject(raw)
        return try {
            Gson().fromJson(candidate, ToolCall::class.java)?.takeIf { it.actionName != null }
        } catch (e: JsonSyntaxException) {
            null
        }
    }
}

/**
 * Reads FunctionGemma's call grammar.
 *
 * ```
 * <start_function_call>call:set_alarm{time:<escape>07:30<escape>}<end_function_call>
 * ```
 *
 * Three things make this more than a regex:
 *
 * - **`<escape>` delimits string values**, and exists precisely so a value may contain the `,` and `}`
 *   that would otherwise end it. A naive split on `,` mangles `location:<escape>Tokyo, Japan<escape>`
 *   into two parameters, one of them named ` Japan`.
 * - **The end token may be missing.** Generation stops at `maxNewTokens`, and a call truncated after
 *   its closing brace is complete information; refusing it would report a model failure for a
 *   configuration choice.
 * - **Bare values are unquoted** (`temperature:15`), so the parser cannot require `<escape>`.
 *
 * Values are surfaced as strings because that is what [ActionSpec] declares and what
 * `validationRules` match against; a numeric literal keeps its text form (`15`), which is what a
 * `"/[0-9]{1,4}/"` rule expects.
 */
private object FunctionGemmaToolCallParser : ToolCallParser {

    private const val CALL_MARKER = "call:"
    private const val ESCAPE = "<escape>"

    override fun parse(raw: String): ToolCall? {
        val markerAt = raw.indexOf(CALL_MARKER).takeIf { it >= 0 } ?: return null
        val braceAt = raw.indexOf('{', markerAt).takeIf { it >= 0 } ?: return null

        val name = raw.substring(markerAt + CALL_MARKER.length, braceAt).trim()
        if (name.isEmpty()) return null

        val body = balancedBody(raw, braceAt) ?: return null
        return ToolCall(actionName = name, parameters = parseFields(body))
    }

    /**
     * The text between [openBrace] and its matching `}`, ignoring braces inside `<escape>` spans.
     *
     * Returns everything to the end of the input when the closing brace never arrives — a truncated
     * generation, where what was emitted is still the model's answer.
     */
    private fun balancedBody(raw: String, openBrace: Int): String? {
        var depth = 0
        var i = openBrace
        var inEscape = false
        while (i < raw.length) {
            if (raw.startsWith(ESCAPE, i)) {
                inEscape = !inEscape
                i += ESCAPE.length
                continue
            }
            if (!inEscape) {
                when (raw[i]) {
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) return raw.substring(openBrace + 1, i)
                    }
                }
            }
            i++
        }
        return raw.substring(openBrace + 1)
    }

    /** Split `key:value` pairs on top-level commas, then unwrap each value. */
    private fun parseFields(body: String): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        for (field in splitTopLevel(body)) {
            val colon = firstTopLevelColon(field)
            if (colon < 0) continue
            val key = field.substring(0, colon).trim()
            if (key.isEmpty()) continue
            out[key] = unwrap(field.substring(colon + 1).trim())
        }
        return out
    }

    private fun splitTopLevel(body: String): List<String> {
        val parts = mutableListOf<String>()
        val current = StringBuilder()
        var depth = 0
        var inEscape = false
        var i = 0
        while (i < body.length) {
            if (body.startsWith(ESCAPE, i)) {
                inEscape = !inEscape
                current.append(ESCAPE)
                i += ESCAPE.length
                continue
            }
            val c = body[i]
            when {
                inEscape -> current.append(c)
                c == '{' || c == '[' -> { depth++; current.append(c) }
                c == '}' || c == ']' -> { depth--; current.append(c) }
                c == ',' && depth == 0 -> {
                    parts += current.toString()
                    current.clear()
                }
                else -> current.append(c)
            }
            i++
        }
        if (current.isNotBlank()) parts += current.toString()
        return parts
    }

    /** The `:` that separates key from value — never one inside an escaped value such as `07:30`. */
    private fun firstTopLevelColon(field: String): Int {
        var i = 0
        while (i < field.length) {
            if (field.startsWith(ESCAPE, i)) return -1 // the value started before any key separator
            if (field[i] == ':') return i
            i++
        }
        return -1
    }

    private fun unwrap(value: String): String {
        val trimmed = value.trim()
        return if (trimmed.startsWith(ESCAPE) && trimmed.endsWith(ESCAPE) && trimmed.length >= 2 * ESCAPE.length) {
            trimmed.substring(ESCAPE.length, trimmed.length - ESCAPE.length)
        } else {
            trimmed
        }
    }
}

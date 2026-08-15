package com.martinkorelic.mobiletransformers.agent

/**
 * Renders an app's [ActionSpec] allowlist into the tool declaration a model expects to be shown.
 *
 * ### Why this had to exist
 *
 * `generateToolCall` sent the user's instruction **and nothing else** — no list of available
 * functions, no output format, no schema. A model was asked to emit a call to one of a set of actions
 * it had never been told about. That works only for a model fine-tuned on exactly this app's
 * allowlist, which is why the Tool calls screen's documented answer was "expect Rejected until you
 * train"; for every off-the-shelf tool-calling model the omission alone guaranteed failure.
 *
 * The allowlist is already the single source of truth for what is permitted. Declaring it to the model
 * from the same object keeps the boundary and the prompt from drifting apart — the same argument that
 * generates the training corpus from it.
 */
object ToolPromptBuilder {

    private const val ESCAPE = "<escape>"

    /** The instruction preamble Google documents for FunctionGemma. */
    private const val PREAMBLE = "You are a model that can do function calling with the following functions"

    /**
     * A declaration block in the dialect [parser] reads back.
     *
     * Prompt and parser are chosen together on purpose: declaring functions in FunctionGemma's grammar
     * and then parsing the reply as JSON is the mismatch this whole seam exists to prevent.
     */
    @JvmStatic
    fun declarations(allowlist: List<ActionSpec>, parser: ToolCallParser): String =
        if (parser === ToolCallParser.FunctionGemma) {
            functionGemmaDeclarations(allowlist)
        } else {
            jsonDeclarations(allowlist)
        }

    /**
     * `<start_function_declaration>declaration:name{…}<end_function_declaration>`, one per action.
     *
     * String values are wrapped in `<escape>` because the grammar requires it — the delimiter is what
     * lets a description contain the `,` and `}` that would otherwise end the field.
     */
    private fun functionGemmaDeclarations(allowlist: List<ActionSpec>): String = buildString {
        append(PREAMBLE).append('\n')
        for (spec in allowlist) {
            append("<start_function_declaration>declaration:").append(spec.actionName).append('{')
            append("description:").append(ESCAPE).append(describe(spec)).append(ESCAPE)
            if (spec.parameters.isNotEmpty()) {
                append(",parameters:{properties:{")
                append(
                    spec.parameters.entries.joinToString(",") { (name, type) ->
                        val hint = spec.validationRules[name]?.let { " (format: $it)" }.orEmpty()
                        "$name:{description:$ESCAPE$name$hint$ESCAPE," +
                            "type:$ESCAPE${type.uppercase()}$ESCAPE}"
                    },
                )
                append("},required:[")
                append(spec.required.joinToString(",") { "$ESCAPE$it$ESCAPE" })
                append("],type:${ESCAPE}OBJECT$ESCAPE}")
            }
            append("}<end_function_declaration>\n")
        }
    }

    /**
     * A plain-language schema for models with no tool-call grammar of their own, naming the exact
     * output shape [ToolCallParser.Json] reads.
     */
    private fun jsonDeclarations(allowlist: List<ActionSpec>): String = buildString {
        append(PREAMBLE).append(". Reply with one JSON object and nothing else, shaped ")
        append("{\"actionName\": <name>, \"parameters\": {<name>: <value>}}.\n")
        for (spec in allowlist) {
            append("- ").append(spec.actionName)
            if (spec.parameters.isEmpty()) {
                append(" (no parameters)")
            } else {
                append(": ")
                append(
                    spec.parameters.keys.joinToString(", ") { name ->
                        val rule = spec.validationRules[name]
                        val required = if (name in spec.required) "" else ", optional"
                        if (rule != null) "$name (format $rule$required)" else "$name (string$required)"
                    },
                )
            }
            append('\n')
        }
    }

    /**
     * What an action does, in words.
     *
     * Derived from the intent it is permitted to fire, because that is the only description an
     * [ActionSpec] carries — the type deliberately holds a *permission*, not documentation. A caller
     * wanting better wording writes it into the action name, which is what the model selects on.
     */
    private fun describe(spec: ActionSpec): String =
        "Performs '${spec.actionName}' on the device (${spec.allowedIntent})."

    /**
     * A tool result to feed back for a second turn, in FunctionGemma's response grammar.
     *
     * The half of the loop that makes a tool call useful: the model calls, the app answers, the model
     * says something about the answer. Without it a call is a dead end.
     */
    @JvmStatic
    fun functionResponse(actionName: String, values: Map<String, String>): String = buildString {
        append("<start_function_response>response:").append(actionName).append('{')
        append(values.entries.joinToString(",") { (k, v) -> "$k:$ESCAPE$v$ESCAPE" })
        append("}<end_function_response>")
    }
}

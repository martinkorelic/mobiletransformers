package com.martinkorelic.mobiletransformers.packages

import java.io.File

/**
 * Which tool-call grammar a package speaks, read from the package rather than guessed from its name.
 *
 * ### Why this exists
 *
 * `ToolCallParser.forModel` took a single string and looked for `"functiongemma"` in it, and
 * `generateToolCall` called it as `forModel(capabilities.task.modelType ?: repoId)`. For a
 * FunctionGemma package `task.modelType` is `"gemma3_text"` — non-null, so `repoId` was never
 * consulted, and the *architecture family* was searched for a *model name* that is never in it. The
 * JSON parser was therefore selected for the one model family that provably does not emit JSON, and
 * every well-formed call it made came back as
 *
 *     no tool call found in the model's output
 *
 * which reads as a model failure and is not one. Two correct fixes were shipped in sequence — the
 * FunctionGemma parser, then the tool declarations — and neither could take effect, because the
 * selector in front of them never chose the parser they were written for.
 *
 * ### Why the chat template is the signal
 *
 * A name match is a guess that a rename breaks. The chat template is what the model was trained
 * against: a model that can be *asked* for a tool call has the grammar for one written into its
 * template, and the template ships in the package. FunctionGemma's contains
 * `<start_function_call>`; Qwen's, Llama-3.1's and Mistral's contain `tool_call`/`tools`. That is a
 * property of the artifact, checkable on device, and it answers both questions at once: whether to
 * offer tool calling at all, and which dialect to parse.
 *
 * The name hints remain as a fallback for packages whose template did not survive the export.
 */
enum class ToolCallDialect {
    /** `<start_function_call>call:name{key:<escape>value<escape>}<end_function_call>`. */
    FUNCTION_GEMMA,

    /** `{"actionName": …, "parameters": {…}}` — the shape this repo's `mobile_actions` corpus teaches. */
    JSON,
}

/**
 * What a package can do with tools.
 *
 * @property supported the model was trained with a tool-call grammar, so asking it for one is
 *   reasonable. False does **not** forbid tool calls — a model fine-tuned on this repo's corpus
 *   learns the JSON shape without its template ever mentioning tools — it only means the app should
 *   not advertise the capability.
 * @property dialect which parser reads this model's output. Always meaningful: [ToolCallDialect.JSON]
 *   is the fallback, and it is the right one for anything fine-tuned here.
 */
data class ToolCallSupport(
    val supported: Boolean,
    val dialect: ToolCallDialect,
) {
    companion object {
        /** The default for a package that says nothing: parse JSON, advertise nothing. */
        @JvmField
        val NONE = ToolCallSupport(supported = false, dialect = ToolCallDialect.JSON)

        // FunctionGemma's grammar tokens, which appear in its template and nowhere else.
        private val FUNCTION_GEMMA_MARKERS = listOf("<start_function_call>", "<start_function_declaration>")

        // The generic signal: a template that renders a `tools` argument at all.
        private val GENERIC_MARKERS = listOf("tool_call", "tool_calls", "<tools>", "available_tools")

        /**
         * Pure detection over a chat template and whatever names are known for the model.
         *
         * @param chatTemplate the package's Jinja chat template, or null when it ships none.
         * @param hints repo id, base model id, architecture — anything that might name the family.
         *   Checked only after the template, because a name is the weaker evidence.
         */
        @JvmStatic
        fun detect(chatTemplate: String?, hints: List<String?> = emptyList()): ToolCallSupport {
            val template = chatTemplate.orEmpty()
            if (FUNCTION_GEMMA_MARKERS.any { template.contains(it, ignoreCase = true) }) {
                return ToolCallSupport(supported = true, dialect = ToolCallDialect.FUNCTION_GEMMA)
            }
            val named = hints.filterNotNull()
            if (named.any { it.contains("functiongemma", ignoreCase = true) }) {
                return ToolCallSupport(supported = true, dialect = ToolCallDialect.FUNCTION_GEMMA)
            }
            if (GENERIC_MARKERS.any { template.contains(it, ignoreCase = true) }) {
                return ToolCallSupport(supported = true, dialect = ToolCallDialect.JSON)
            }
            return NONE
        }

        /** The tokenizer stage's chat template, from either place the exporter may have put it. */
        @JvmStatic
        fun readChatTemplate(tokenizerDir: File): String? {
            // #15 writes it standalone; older packages keep it inside tokenizer_config.json.
            val jinja = File(tokenizerDir, "chat_template.jinja")
            if (jinja.isFile) return runCatching { jinja.readText(Charsets.UTF_8) }.getOrNull()
            val config = File(tokenizerDir, "tokenizer_config.json")
            if (!config.isFile) return null
            return runCatching {
                val text = config.readText(Charsets.UTF_8)
                // Deliberately a substring check, not a parse: this file is over a megabyte for a
                // large-vocabulary tokenizer (FunctionGemma's is 1.1 MB of added_tokens_decoder), and
                // all we need to know is whether the grammar appears in it.
                text.takeIf { it.contains("chat_template") }
            }.getOrNull()
        }

        /** [detect] against an installed package's tokenizer stage. Never throws. */
        @JvmStatic
        fun read(tokenizerDir: File, hints: List<String?> = emptyList()): ToolCallSupport =
            detect(readChatTemplate(tokenizerDir), hints)
    }
}

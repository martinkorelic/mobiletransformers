package com.martinkorelic.mobiletransformers.rag

import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch

/**
 * #27: assembles the grounded prompt from a query + retrieved matches. Pure (JVM-testable). The default
 * template is overridable per call via a [PromptStrategy]; the assembled prompt is always surfaced on
 * `GroundedResult.prompt` so the grounded flow is inspectable.
 */
fun interface PromptStrategy {
    fun assemble(query: String, matches: List<RetrievalMatch>): String
}

object PromptAssembler {
    val DEFAULT: PromptStrategy =
        PromptStrategy { query, matches ->
            buildString {
                appendLine("Use the following context to answer the question.")
                appendLine()
                appendLine("Context:")
                matches.forEach { appendLine("- ${it.text}") }
                appendLine()
                appendLine("Question: $query")
                append("Answer:")
            }
        }

    fun assemble(
        query: String,
        matches: List<RetrievalMatch>,
        strategy: PromptStrategy = DEFAULT,
    ): String = strategy.assemble(query, matches)
}

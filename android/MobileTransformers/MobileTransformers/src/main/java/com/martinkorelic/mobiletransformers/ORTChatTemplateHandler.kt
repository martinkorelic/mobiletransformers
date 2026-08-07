package com.martinkorelic.mobiletransformers

import android.util.Log
import io.pebbletemplates.pebble.PebbleEngine
import io.pebbletemplates.pebble.error.PebbleException
import io.pebbletemplates.pebble.loader.StringLoader
import io.pebbletemplates.pebble.template.EvaluationContext
import io.pebbletemplates.pebble.template.PebbleTemplate
import java.io.StringWriter


class ORTChatTemplateHandler(
    chatTemplate: String,
) {

    private val LOG_TAG = "ORTChatTemplateHandler"

    private val pebbleEngine: PebbleEngine = PebbleEngine.Builder().extension(ChatTemplatePebbleExtension()).extension(TypeSafeComparisonExtension()).loader(StringLoader()).autoEscaping(false).build()

    private val chatTemplate = prepareChatTemplate(chatTemplate)

    fun prepareChatTemplate(originalTemplate: String) : String {

        var template = originalTemplate

        // Replace Python type template keywords
        template = replaceTemplateKeywords(template)

        template = prepareChatTemplateArrays(template)
        template = prepareChatTemplateContains(template)
        // Replace comparison operators with type-safe versions
        template = replaceComparisonOperators(template)

        Log.d(LOG_TAG, template)
        return template
    }

    /**
     * Replaces == and != operators with type-safe eq and neq operators
     * This fixes type comparison issues in Pebble templates
     */
    fun replaceComparisonOperators(template: String): String {
        var result = template

        // Replace != with neq (must be done first to avoid conflicts)
        result = result.replace("!=", "neq")

        // Replace == with eq
        result = result.replace("==", "eq")

        return result
    }

    fun replaceTemplateKeywords(template: String): String {
        // Regex pattern to match full words only, ensuring exact word boundaries
        val regex = Regex("""\b(?:elif|none)\b""")

        // Replace only full words
        return regex.replace(template) { matchResult ->
            when (matchResult.value) {
                "elif" -> "elseif"
                "none" -> "empty"
                else -> matchResult.value
            }
        }
    }

    fun prepareChatTemplateArrays(template: String): String {
        // Regex to match slicing patterns like [x:], [:x], [x:y]
        val regex = Regex("""(\w+)\[\s*(\d*)\s*:\s*(\d*)\s*]""")

        // Replace only the slicing patterns
        return regex.replace(template) { matchResult ->
            val variableName = matchResult.groupValues[1] // Variable name (e.g., messages)
            val start = matchResult.groupValues[2] // Start index
            val end = matchResult.groupValues[3] // End index

            // Determine slice parameters
            when {
                start.isNotEmpty() && end.isEmpty() -> {
                    // Case: [x:]
                    "$variableName | slice($start)"
                }
                start.isEmpty() && end.isNotEmpty() -> {
                    // Case: [:x]
                    "$variableName | slice($end)"
                }
                start.isNotEmpty() && end.isNotEmpty() -> {
                    // Case: [x:y]
                    "$variableName | slice($start,$end)"
                }
                else -> {
                    // Default case (should not occur in valid slicing syntax)
                    matchResult.value
                }
            }
        }
    }

    fun prepareChatTemplateContains(template: String): String {
        // Regex to match expressions with 'in' operator inside {% %} or {{ }} blocks
        val regex = """'([^']+)'\s+in\s+(\w+)""".toRegex()

        return template.replace(regex) { matchResult ->
            val (quotedString, variable) = matchResult.destructured
            "$variable contains '$quotedString'"
        }
    }

    /**
     * Build the input string for tokenization based on the provided messages.
     *
     * @param context
     */
    fun buildInput(context: Map<String, Any>): String {
        // Prepare the context for the template
        val template = pebbleEngine.getTemplate(chatTemplate)
        val writer = StringWriter()
        template.evaluate(writer, context)
        return writer.toString()
    }
}
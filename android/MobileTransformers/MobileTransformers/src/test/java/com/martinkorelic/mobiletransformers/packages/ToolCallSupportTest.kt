package com.martinkorelic.mobiletransformers.packages

import com.martinkorelic.mobiletransformers.agent.ToolCallParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Which tool-call grammar a package speaks, and which parser that selects.
 *
 * The regression underneath these: `generateToolCall` chose its parser with
 * `ToolCallParser.forModel(capabilities.task.modelType ?: repoId)`. `task.modelType` is the
 * architecture — `"gemma3_text"` — and it is non-null for every modern package, so the `?:` never
 * fired and the repo id carrying the word "functiongemma" was never looked at. The JSON parser was
 * selected for the one family that does not emit JSON, and every well-formed call it made came back
 * as "no tool call found in the model's output".
 */
class ToolCallSupportTest {

    @get:Rule
    val temp = TemporaryFolder()

    /** The two markers that appear in FunctionGemma's template and nowhere else. */
    private val functionGemmaTemplate = """
        {%- if tools -%}
            {{- '<start_function_declaration>' -}}
            {{- format_function_declaration(tool) | trim }}
            {{- '<end_function_declaration>' -}}
        {%- endif -%}
        {{- '<start_function_call>call:' + function['name'] + '{' -}}
    """.trimIndent()

    private val qwenStyleTemplate = """
        {%- for tool_call in message.tool_calls %}{{ tool_call.function.name }}{%- endfor %}
    """.trimIndent()

    private val plainChatTemplate = """
        {% for message in messages %}<|im_start|>{{ message['role'] }}
        {{ message['content'] }}<|im_end|>{% endfor %}
    """.trimIndent()

    @Test
    fun aFunctionGemmaTemplateIsDetectedFromTheTemplateAlone() {
        // No name hint at all: the artifact is enough, which is the point of reading it.
        val support = ToolCallSupport.detect(functionGemmaTemplate, hints = emptyList())

        assertTrue(support.supported)
        assertEquals(ToolCallDialect.FUNCTION_GEMMA, support.dialect)
    }

    @Test
    fun theArchitectureNameAloneNoLongerDecidesTheParser() {
        // The exact call the facade used to make. It must not resolve to the JSON parser.
        val support = ToolCallSupport.detect(
            functionGemmaTemplate,
            hints = listOf("gemma3_text"),
        )

        assertEquals(
            "a gemma3_text package with FunctionGemma's grammar must get the FunctionGemma parser",
            ToolCallParser.FunctionGemma,
            ToolCallParser.forDialect(support.dialect),
        )
    }

    @Test
    fun theRepoIdIsTheFallbackWhenNoTemplateSurvivedTheExport() {
        val support = ToolCallSupport.detect(
            chatTemplate = null,
            hints = listOf("gemma3_text", "mobiletransformers/functiongemma-270m-it"),
        )

        assertTrue(support.supported)
        assertEquals(ToolCallDialect.FUNCTION_GEMMA, support.dialect)
    }

    @Test
    fun everyHintIsConsideredNotJustTheFirst() {
        // `forModel(modelType ?: repoId)` stopped at the first non-null and that was the bug.
        assertEquals(
            ToolCallParser.FunctionGemma,
            ToolCallParser.forModel("gemma3_text", "mobiletransformers/functiongemma-270m-it"),
        )
    }

    @Test
    fun aGenericToolCallingTemplateGetsTheJsonParser() {
        val support = ToolCallSupport.detect(qwenStyleTemplate, hints = listOf("Qwen/Qwen2-0.5B"))

        assertTrue(support.supported)
        assertEquals(ToolCallDialect.JSON, support.dialect)
    }

    @Test
    fun aPlainChatModelAdvertisesNothingButStillParsesJson() {
        // Not supported != cannot be asked: a model fine-tuned on this repo's mobile_actions corpus
        // learns the JSON shape without its template ever mentioning tools.
        val support = ToolCallSupport.detect(plainChatTemplate, hints = listOf("HuggingFaceTB/SmolLM2-135M"))

        assertFalse(support.supported)
        assertEquals(ToolCallDialect.JSON, support.dialect)
        assertEquals(ToolCallParser.Json, ToolCallParser.forDialect(support.dialect))
    }

    @Test
    fun aPackageThatSaysNothingIsUnsupportedRatherThanAFailure() {
        val support = ToolCallSupport.detect(chatTemplate = null, hints = emptyList())

        assertFalse(support.supported)
        assertEquals(ToolCallSupport.NONE, support)
    }

    @Test
    fun itReadsTheStandaloneJinjaFileTheExporterWrites() {
        val dir = temp.newFolder("tokenizer")
        File(dir, "chat_template.jinja").writeText(functionGemmaTemplate)

        val support = ToolCallSupport.read(dir, hints = listOf("gemma3_text"))

        assertEquals(ToolCallDialect.FUNCTION_GEMMA, support.dialect)
    }

    @Test
    fun anAbsentTokenizerStageIsNotAnError() {
        // Detection decides which chips to show. A package that cannot be read must still load.
        val support = ToolCallSupport.read(File(temp.root, "does-not-exist"), hints = emptyList())

        assertEquals(ToolCallSupport.NONE, support)
    }
}

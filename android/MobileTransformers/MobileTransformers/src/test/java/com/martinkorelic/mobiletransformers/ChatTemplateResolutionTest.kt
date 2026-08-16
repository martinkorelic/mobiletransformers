package com.martinkorelic.mobiletransformers

import com.google.gson.Gson
import com.google.gson.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

/**
 * Where `ORTTokenizerNative` looks for a chat template, and whether it trusts what it finds.
 *
 * ### The defect this pins
 *
 * `loadTokenizerConfiguration` read `chat_template` out of `tokenizer_config.json` and nowhere else.
 * The exporter has never written it there: `export/pipeline.py::_emit_chat_template` writes a sibling
 * `chat_template.jinja`, which the installers flatten into `tokenizer/`. Measured across every package
 * on this machine — `build/pkg`, `pkg-ab`, `pkg-functiongemma`, `pkg-gemma3` — the sibling file is
 * present in all of them and the key appears in **none**.
 *
 * So `chatTemplate` was null for every package ever shipped, [ORTConversationState] was never
 * constructed, and no plain-chat prompt was wrapped in the model's turn format. A model trained on
 * `<start_of_turn>user … <start_of_turn>model` was handed a bare string.
 * [siblingFileIsReadWhenTheConfigCarriesNoKey] is the case that was broken; it fails against the
 * key-only lookup.
 *
 * ### Why the probe half matters just as much
 *
 * Resolving more templates is only an improvement if the ones resolved actually work. Pebble is not
 * Jinja, and FunctionGemma's template alone uses `namespace()`, `dictsort` and macros it does not
 * implement. Without [ORTTokenizerNative.validatedChatTemplate], such a template would throw
 * mid-generation once per turn instead of once at load — strictly worse than the null it replaced.
 * [aTemplatePebbleCannotCompileIsDiscarded] and [aTemplateRenderingEmptyIsDiscarded] pin the fallback.
 *
 * Robolectric because both functions log.
 */
@RunWith(RobolectricTestRunner::class)
class ChatTemplateResolutionTest {

    @get:Rule
    val temp = TemporaryFolder()

    /** SmolLM2's shipped template, verbatim from `shared/tokenizer/chat_template.jinja`. */
    private val workingTemplate =
        "{% for message in messages %}" +
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" +
            "{% endfor %}" +
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    private val specialTokens = mapOf(
        "bos_token" to "<|im_start|>",
        "eos_token" to "<|im_end|>",
    )

    private fun config(json: String): JsonObject =
        Gson().fromJson(json, JsonObject::class.java)

    private fun dirWithSibling(template: String): File {
        val dir = temp.newFolder("tokenizer-${template.hashCode()}")
        File(dir, ORTTokenizerNative.CHAT_TEMPLATE_FILE_NAME).writeText(template)
        return dir
    }

    // ---------------------------------------------------------------- resolution

    /**
     * **The regression.** Config carries no `chat_template`; the template is in the sibling file.
     * This is the shape of every package the exporter produces, and the key-only lookup returned null
     * for all of them.
     */
    @Test
    fun siblingFileIsReadWhenTheConfigCarriesNoKey() {
        val dir = dirWithSibling(workingTemplate)
        val resolved = ORTTokenizerNative.resolveChatTemplate(config("""{"model_max_length": 2048}"""), dir)

        assertEquals(workingTemplate, resolved)
    }

    /** Packages predating the exporter change inline it, and must keep working. */
    @Test
    fun inlineKeyIsReadWhenThereIsNoSiblingFile() {
        val dir = temp.newFolder("tokenizer-inline-only")
        val resolved = ORTTokenizerNative.resolveChatTemplate(
            config("""{"chat_template": "{{ 'inline' }}"}"""),
            dir,
        )

        assertEquals("{{ 'inline' }}", resolved)
    }

    /** A package shipping both is stating a deliberate override, so the key wins. */
    @Test
    fun inlineKeyWinsOverTheSiblingFile() {
        val dir = dirWithSibling(workingTemplate)
        val resolved = ORTTokenizerNative.resolveChatTemplate(
            config("""{"chat_template": "{{ 'inline' }}"}"""),
            dir,
        )

        assertEquals("{{ 'inline' }}", resolved)
    }

    @Test
    fun neitherSourcePresentResolvesToNull() {
        assertNull(
            ORTTokenizerNative.resolveChatTemplate(config("""{"model_max_length": 2048}"""), temp.newFolder("empty")),
        )
    }

    /**
     * Some tokenizer configs carry a LIST of named templates (`[{name, template}]`) rather than a
     * string. `asString` throws on that shape, which would take down the whole of
     * `loadTokenizerConfiguration` — including the special-token parsing above it — via its catch-all.
     * The non-primitive must be stepped over so the sibling file is still found.
     */
    @Test
    fun aNonStringChatTemplateFallsThroughToTheSiblingInsteadOfThrowing() {
        val dir = dirWithSibling(workingTemplate)
        val resolved = ORTTokenizerNative.resolveChatTemplate(
            config("""{"chat_template": [{"name": "default", "template": "x"}]}"""),
            dir,
        )

        assertEquals(workingTemplate, resolved)
    }

    @Test
    fun aBlankTemplateIsTreatedAsAbsent() {
        val dir = dirWithSibling("   \n  ")
        assertNull(ORTTokenizerNative.resolveChatTemplate(config("{}"), dir))
    }

    @Test
    fun aMissingDirectoryResolvesToNullRatherThanThrowing() {
        assertNull(ORTTokenizerNative.resolveChatTemplate(config("{}"), File("/does/not/exist")))
        assertNull(ORTTokenizerNative.resolveChatTemplate(null, null))
    }

    // ---------------------------------------------------------------- probe render

    @Test
    fun aWorkingTemplateSurvivesTheProbe() {
        val kept = ORTTokenizerNative.validatedChatTemplate(workingTemplate, specialTokens)

        assertNotNull("SmolLM2's own template must survive the probe", kept)
        assertEquals(workingTemplate, kept)
    }

    /**
     * The probe must reject rather than propagate. Unclosed `{% if %}` is a compile error in any
     * Jinja-alike, so this is deterministic regardless of which constructs Pebble happens to support.
     */
    @Test
    fun aTemplatePebbleCannotCompileIsDiscarded() {
        assertNull(ORTTokenizerNative.validatedChatTemplate("{% if true %}never closed", specialTokens))
    }

    /** A template that evaluates to nothing would hand generation an empty prompt every turn. */
    @Test
    fun aTemplateRenderingEmptyIsDiscarded() {
        assertNull(ORTTokenizerNative.validatedChatTemplate("{% if false %}x{% endif %}", specialTokens))
    }

    @Test
    fun aNullCandidateStaysNull() {
        assertNull(ORTTokenizerNative.validatedChatTemplate(null, specialTokens))
    }

    /**
     * The probe renders a real two-turn conversation, so a template that only references
     * `messages` still produces both turns and the assistant opener — i.e. the probe exercises the
     * same path a live turn does, rather than a degenerate empty-message render that would pass
     * templates which break on real input.
     */
    @Test
    fun theProbeExercisesBothTurnsAndTheGenerationPrompt() {
        // Rendering the same context the probe uses, to assert on what it produced.
        val rendered = ORTChatTemplateHandler(workingTemplate).buildInput(
            mutableMapOf<String, Any>(
                "messages" to listOf(
                    mapOf("role" to "user", "content" to "ping"),
                    mapOf("role" to "assistant", "content" to "pong"),
                ),
                "add_generation_prompt" to true,
            ).also { it.putAll(specialTokens) },
        )

        assertTrue("probe must render the user turn", rendered.contains("ping"))
        assertTrue("probe must render the assistant turn", rendered.contains("pong"))
        assertTrue("probe must reach the generation opener", rendered.trimEnd().endsWith("assistant"))
    }
}

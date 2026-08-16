package com.martinkorelic.mobiletransformers

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * #37: the train/inference **prompt-shape seam** (JVM, Robolectric — the template handler logs).
 *
 * ### What this pins, and what it does not
 *
 * `ORTDataCurator` tokenized whatever the preprocessor returned, verbatim, while
 * `ORTGeneratorNative.generate` tokenizes its first turn with `prependBos = true` and — when the
 * package's tokenizer loaded a chat template — wraps the prompt via
 * [ORTConversationState.addUserMessage]. So the two sides could disagree on the token sequence for
 * one and the same instruction. `TaskPreprocessor.formatsPromptForGeneration` closes that.
 *
 * **Scope, honestly stated.** The BOS half of the mismatch was real and is fixed. The chat-template
 * half *was* latent for the package this was measured on, and is no longer: `ORTTokenizerNative` read
 * the template only from `tokenizer_config.json` while SmolLM2's export writes it to a sibling
 * `chat_template.jinja`, so `chatTemplate` was null and neither side templated. The device log said so
 * in as many words — `W/ORTTokenizerNative: Chat template not found … No chat template will be used`.
 * `ORTTokenizerNative.resolveChatTemplate` now reads the sibling file, so both sides of this seam
 * template for real; see `ChatTemplateResolutionTest`. The measurements below predate that fix and
 * describe the untemplated behaviour — re-measure before citing them.
 *
 * **This did not fix the #37 gate.** With BOS parity and EOS in place, training reaches a loss of
 * ~0.006, the merge completes over all 60 adapted tensors, 60 merged initializers load at inference
 * with no base-weight fallback, and generation *still* collapses to token 198 (newline) on the very
 * prompt it was trained on. That leaves merge numerics — not convergence, and not prompt format — as
 * where the remaining defect lives. See the #37 self-check.
 *
 * ### Why these assertions can fail
 *
 * [promptFormatMatchingIsOptInAndMobileActionsOptsIn] fails if the opt-in is dropped or leaks to other
 * tasks. [aRenderedFirstTurnEndsAtTheAssistantOpener] fails if the render stops short of the opener,
 * which is what decides whether the completion is trained as the *assistant's* turn or glued onto the
 * user's. [renderingChangesTheTokenizedPrompt] fails if the render becomes a no-op.
 */
@RunWith(RobolectricTestRunner::class)
class ChatFormattedTrainingPromptTest {

    /** SmolLM2's shipped template, verbatim from `shared/tokenizer/chat_template.jinja`. */
    private val smolLm2Template =
        "{% for message in messages %}" +
            "{% if loop.first and messages[0]['role'] != 'system' %}" +
            "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}" +
            "{% endif %}" +
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" +
            "{% endfor %}" +
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    /**
     * The one expression both sides of the seam use: a **fresh** conversation state, no explicit system
     * prompt (matching `GenerationConfig.systemPrompt`'s null default), rendering one first turn.
     */
    private fun renderFirstTurn(content: String): String =
        ORTConversationState(ORTChatTemplateHandler(smolLm2Template), emptyMap(), null)
            .addUserMessage(content)

    @Test
    fun promptFormatMatchingIsOptInAndMobileActionsOptsIn() {
        assertTrue(
            "mobile_actions is generated through generateToolCall, which goes through the chat " +
                "template — its training data must be rendered the same way",
            MobileActionsPreprocessor.formatsPromptForGeneration(),
        )

        // The default must stay false: every pre-existing task was trained and evaluated on raw
        // prompts, and flipping them here would silently invalidate those runs.
        assertFalse(CoLAPreprocessor.formatsPromptForGeneration())
        assertFalse(BoolqPreprocessor.formatsPromptForGeneration())
        assertFalse(LogiqaPreprocessor.formatsPromptForGeneration())
        assertFalse(MiniPersonalQAPreprocessor.formatsPromptForGeneration())

        // A caller's own preprocessor inherits the raw default rather than being opted in behind its back.
        val custom = object : TaskPreprocessor {
            override fun preprocess(json: org.json.JSONObject) = "in" to "out"
        }
        assertFalse(custom.formatsPromptForGeneration())
    }

    @Test
    fun aRenderedFirstTurnEndsAtTheAssistantOpener() {
        val rendered = renderFirstTurn("wake me at 07:30")

        // The completion is concatenated directly after this string during training, so the prompt has
        // to stop exactly where the assistant's turn begins. Ending anywhere else trains the JSON as
        // part of the user's message.
        assertTrue(
            "a training prompt must end at the assistant opener, not mid-turn; got: '$rendered'",
            rendered.endsWith("<|im_start|>assistant\n"),
        )
        assertTrue("the instruction must survive the render", rendered.contains("wake me at 07:30"))
        assertTrue(
            "the template injects its default system turn when none is supplied",
            rendered.contains("<|im_start|>system"),
        )
    }

    @Test
    fun renderingChangesTheTokenizedPrompt() {
        val raw = "wake me at 07:30"
        assertNotEquals(
            "if the render were a no-op the seam would be back and this test would pass vacuously",
            raw,
            renderFirstTurn(raw),
        )
    }

    @Test
    fun everyRowIsRenderedAsAFirstTurn() {
        // The curator builds a new state per row. Were it to share one, row 2 would render only the
        // delta (`buildNewUserMessageTemplate`) and train on a fragment with no system turn.
        assertEquals(renderFirstTurn("timer for 30 seconds"), renderFirstTurn("timer for 30 seconds"))

        val shared = ORTConversationState(ORTChatTemplateHandler(smolLm2Template), emptyMap(), null)
        val first = shared.addUserMessage("timer for 30 seconds")
        val second = shared.addUserMessage("timer for 30 seconds")
        assertNotEquals(
            "a shared state must NOT be what the curator uses — proving the per-row state matters",
            first,
            second,
        )
    }
}

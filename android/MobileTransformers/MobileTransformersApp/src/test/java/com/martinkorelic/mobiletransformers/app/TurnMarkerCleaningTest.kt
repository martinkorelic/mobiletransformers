package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.viewmodels.ChatViewModel
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * What a reader sees when a chat model emits its own scaffolding.
 *
 * Every template in this app's catalog ends a turn with a marker, and for ChatML models that marker
 * (`<|im_end|>`) *is* the eos token — so it arrives as the last token of a completely ordinary
 * reply. The engines suppress it at the emit site now; this is the net under that, and the layer
 * that also handles a model which keeps talking past its turn and starts playing both parts.
 *
 * SmolLM2 showed `<|im_end|>` in the Chat bubble, which is what these assertions are about.
 */
class TurnMarkerCleaningTest {

    private fun clean(raw: String) = ChatViewModel.cleanTurnMarkers(raw)

    @Test
    fun theChatMlEndMarkerNeverReachesTheBubble() {
        assertEquals("Paris is the capital of France.", clean("Paris is the capital of France.<|im_end|>"))
    }

    @Test
    fun aModelPlayingBothPartsIsCutAtItsOwnTurnEnd() {
        assertEquals(
            "The Eiffel Tower is in Paris.",
            clean("The Eiffel Tower is in Paris.<|im_end|>\n<|im_start|>user\nAnd Rome?<|im_end|>"),
        )
        assertEquals(
            "It is 4.",
            clean("It is 4.<end_of_turn><start_of_turn>user\nwhat about 3+3<end_of_turn>"),
        )
    }

    @Test
    fun aRoleLabelTheModelCompletedForItselfIsStripped() {
        // Gemma's prompt ends with `<start_of_turn>model\n` and the model often re-emits the label.
        assertEquals("Done.", clean("model\nDone.<end_of_turn>"))
        assertEquals("Done.", clean("assistant\nDone.<|im_end|>"))
        // ...but only as its own line: "model" is an ordinary word, and a reply that opens with it
        // must survive intact. `removePrefix("model")` on its own got this wrong.
        assertEquals("model weights are stored on device.", clean("model weights are stored on device."))
        assertEquals("models are exported by optimum.", clean("models are exported by optimum."))
    }

    @Test
    fun ordinaryTextIsUntouched() {
        assertEquals("2 + 2 = 4", clean("2 + 2 = 4"))
        assertEquals("", clean("   "))
        // A partially streamed marker is still incomplete text, not a marker — cutting early here
        // would make the streaming bubble flicker on every token that starts with `<`.
        assertEquals("almost <|im_en", clean("almost <|im_en"))
    }

    /**
     * Cleaning is for DISPLAY; the accumulator behind it must stay raw.
     *
     * Feeding `clean()` its own output token by token — `clean(shown + token)` — drops any newline a
     * token ends with, because `trim()` sees it as trailing. Every list item and paragraph break in a
     * streamed answer ends a token that way, so the next token runs straight onto the previous line.
     * This asserts the shape the view models actually use.
     */
    @Test
    fun streamingCleansForDisplayWithoutEatingTheAccumulatorsNewlines() {
        val tokens = listOf("Steps:\n", "1. pull\n", "2. train\n", "3. merge", "<|im_end|>")

        val raw = StringBuilder()
        var shown = ""
        for (token in tokens) {
            raw.append(token)
            shown = clean(raw.toString())
        }
        assertEquals("Steps:\n1. pull\n2. train\n3. merge", shown)

        // The shape that loses them, kept here so the difference is visible rather than asserted about.
        var wrong = ""
        for (token in tokens) wrong = clean(wrong + token)
        assertEquals("Steps:1. pull2. train3. merge", wrong)
    }

    @Test
    fun everyMarkerInTheListIsActuallyStripped() {
        // A list that silently stopped matching would pass every case above except this one.
        for (marker in ChatViewModel.TURN_MARKERS) {
            assertEquals("answer for $marker", "answer", clean("answer$marker trailing"))
        }
    }
}

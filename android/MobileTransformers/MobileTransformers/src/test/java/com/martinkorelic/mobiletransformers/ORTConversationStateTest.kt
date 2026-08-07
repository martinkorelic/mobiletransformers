package com.martinkorelic.mobiletransformers

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #23: conversation-state lifecycle (JVM, no template engine / device).
 *
 * With a null template handler the class exercises its pure history/reset bookkeeping without touching
 * Pebble or `android.util.Log`. The rendered-offset fix in [ORTConversationState.addAssistantMessage]
 * (which needs a real chat template) is validated by the device conversation-reset smoke; here we pin
 * that [ORTConversationState.resetForNewConversation] fully clears state so a new conversation never
 * inherits the previous turn's prepend state.
 */
class ORTConversationStateTest {

    @Test
    fun resetClearsHistoryAndFirstMessageFlag() {
        val state = ORTConversationState(null, emptyMap(), systemPrompt = "sys")
        state.addUserMessage("hello") // first message -> system + user added to history
        state.addAssistantMessage("hi there") // + assistant
        assertEquals(3, state.getConversationHistory().size)

        state.resetForNewConversation()
        assertTrue(state.getConversationHistory().isEmpty())

        // After reset the next user message is treated as the first again (system prompt re-applied).
        state.addUserMessage("again")
        assertEquals(listOf("system", "user"), state.getConversationHistory().map { it["role"] })
    }

    @Test
    fun systemPromptAppliedOnFirstMessage() {
        val state = ORTConversationState(null, emptyMap(), systemPrompt = null)
        state.setSystemPrompt("you are helpful")
        state.addUserMessage("q")
        assertEquals("you are helpful", state.getConversationHistory().first()["content"])
    }
}

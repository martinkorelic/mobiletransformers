package com.martinkorelic.mobiletransformers


class ORTConversationState(
    private var templateHandler: ORTChatTemplateHandler?,
    private val specialTokens: Map<String, String>,
    private var systemPrompt: String? = null
) {
    private val conversationHistory = mutableListOf<Map<String, String>>()
    private var currentConversationLength : Int = 0
    private var isFirstMessage = true

    fun setSystemPrompt(prompt: String?) {
        systemPrompt = prompt
    }

    /**
     * Adds a new user message and the content that will be processed into the LLM.
     * Takes in account the states and previous context that is already computed in KV cache.
     */
    fun addUserMessage(content: String): String {

        if (isFirstMessage) {
            // First message: create full template with system prompt + user message
            isFirstMessage = false
            return buildInitialTemplate(content)
        } else {
            // Subsequent messages: get only the new user message part
            return buildNewUserMessageTemplate(content)
        }
    }

    /**
     * Adds a new assistant message after the LLM has stopped computing.
     *
     * #23: advance the consumed-prefix marker by the assistant content's RENDERED offset, not the
     * decoded `content.length`. `currentConversationLength` is an index into the chat-template's
     * rendered text (used by [buildNewUserMessageTemplate] to slice the next-turn delta); chat templates
     * routinely add/trim whitespace around a turn's content, so the decoded length under/over-counts the
     * rendered position and the next delta starts mid-content — the "one token from the previous
     * assistant message keeps prepending" bug. Locating the content inside the re-rendered history
     * anchors the marker exactly at the end of the rendered content (before the turn's closing markup),
     * so the next user delta cleanly re-feeds the closer the KV cache does not yet hold.
     */
    fun addAssistantMessage(content: String) {
        // `currentConversationLength` here is the sent prefix ending at the assistant generation opener.
        val openerPrefixLen = currentConversationLength

        conversationHistory.add(mapOf("role" to "assistant", "content" to content))

        val rendered = renderHistory(addGenerationPrompt = false)
        currentConversationLength = if (rendered.length > openerPrefixLen) {
            val tail = rendered.substring(openerPrefixLen)
            val idx = tail.indexOf(content)
            if (idx >= 0) openerPrefixLen + idx + content.length
            else openerPrefixLen + content.length // fallback: template transformed the content
        } else {
            openerPrefixLen + content.length
        }
    }

    private fun renderHistory(addGenerationPrompt: Boolean): String {
        val context = mutableMapOf<String, Any>(
            "messages" to conversationHistory,
            "add_generation_prompt" to addGenerationPrompt
        )
        context.putAll(specialTokens)
        return templateHandler?.buildInput(context) ?: ""
    }

    /**
     * Generates the full starting template from the conversation history.
     * Adds the system prompt and the user message right after.
     */
    private fun buildInitialTemplate(content: String): String {

        systemPrompt?.takeIf { it.isNotBlank() }?.let {
            conversationHistory.add(mapOf("role" to "system", "content" to it))
        }

        conversationHistory.add(mapOf("role" to "user", "content" to content))

        val context = mutableMapOf(
            "messages" to conversationHistory,
            "add_generation_prompt" to true
        )
        context.putAll(specialTokens)

        val output = templateHandler?.buildInput(context) ?: ""

        // Store the current conversation length
        currentConversationLength = output.length

        return output
    }

    private fun buildNewUserMessageTemplate(userContent: String): String {
        // Build template for just the new user message
        conversationHistory.add(mapOf("role" to "user", "content" to userContent))

        val context = mutableMapOf(
            "messages" to conversationHistory,
            "add_generation_prompt" to true
        )
        context.putAll(specialTokens)

        // Get the full template with the new user message
        val userMessageTemplate = templateHandler?.buildInput(context) ?: ""

        // Extract only the new part that hasn't been processed by KV cache yet
        val newPart = if (currentConversationLength < userMessageTemplate.length) {
            userMessageTemplate.substring(currentConversationLength)
        } else {
            ""
        }

        // Update with the new length
        currentConversationLength = userMessageTemplate.length

        return newPart
    }

    fun resetForNewConversation() {
        conversationHistory.clear()
        currentConversationLength = 0
        isFirstMessage = true
    }

    fun getConversationHistory(): List<Map<String, String>> {
        return conversationHistory.toList()
    }
}
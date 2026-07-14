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
     * Adds a new assistant message after the LLM has stopped computing
     */
    fun addAssistantMessage(content: String) {

        // Add what the newly produced message length
        currentConversationLength += content.length

        // Create the full conversation history
        conversationHistory.add(mapOf("role" to "assistant", "content" to content))

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
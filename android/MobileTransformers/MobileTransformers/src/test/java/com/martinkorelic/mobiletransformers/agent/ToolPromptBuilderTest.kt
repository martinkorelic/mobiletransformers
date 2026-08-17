package com.martinkorelic.mobiletransformers.agent

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The prompt a tool-calling turn actually sends.
 *
 * ### Why the turn markers matter
 *
 * `generateToolCall` used to send `declarations + "\n" + instruction` with no turn structure at all.
 * FunctionGemma is trained on `<start_of_turn>developer … <end_of_turn><start_of_turn>user …
 * <end_of_turn><start_of_turn>model`, and the framing normally comes from the tokenizer's chat
 * template — but the exporter writes that template to a sibling `chat_template.jinja` which
 * `ORTTokenizerNative` does not read, so `chatTemplate` is null on device and **nothing** wrapped the
 * prompt. The model was handed a bare instruction in a shape it has never been trained on and asked
 * for a grammar it only emits inside a model turn.
 */
class ToolPromptBuilderTest {

    private val allowlist = listOf(
        ActionSpec(
            actionName = "set_alarm",
            parameters = mapOf("time" to "string"),
            requiredParameters = setOf("time"),
            validationRules = mapOf("time" to "^([01][0-9]|2[0-3]):[0-5][0-9]$"),
            allowedIntent = "android.intent.action.SET_ALARM",
        ),
        ActionSpec(
            actionName = "toggle_wifi",
            parameters = emptyMap(),
            allowedIntent = "android.settings.WIFI_SETTINGS",
        ),
    )

    @Test
    fun theFunctionGemmaPromptCarriesTheTurnStructureTheModelWasTrainedOn() {
        val prompt = ToolPromptBuilder.prompt(allowlist, ToolCallParser.FunctionGemma, "wake me at 07:30")

        assertTrue("declarations belong in the developer turn", prompt.contains("<start_of_turn>developer"))
        assertTrue(prompt.contains("<start_function_declaration>declaration:set_alarm"))
        assertTrue(prompt.contains("<end_of_turn>"))
        assertTrue("the instruction belongs in the user turn", prompt.contains("<start_of_turn>user\nwake me at 07:30"))
        assertTrue("the floor must be handed to the model", prompt.trimEnd().endsWith("<start_of_turn>model"))
    }

    @Test
    fun theDeveloperTurnClosesBeforeTheUserTurnOpens() {
        val prompt = ToolPromptBuilder.prompt(allowlist, ToolCallParser.FunctionGemma, "turn on wifi")

        val developerAt = prompt.indexOf("<start_of_turn>developer")
        val userAt = prompt.indexOf("<start_of_turn>user")
        val closeAt = prompt.indexOf("<end_of_turn>")

        assertTrue(developerAt in 0 until closeAt)
        assertTrue("the developer turn must close before the user turn opens", closeAt < userAt)
    }

    @Test
    fun everyAllowlistedActionIsDeclared() {
        // The declaration and the boundary come from one object on purpose: a model asked for an
        // action it was never shown cannot produce it, and one shown an action the validator does
        // not permit is being taught to be refused.
        val prompt = ToolPromptBuilder.prompt(allowlist, ToolCallParser.FunctionGemma, "hi")

        for (spec in allowlist) {
            assertTrue("'${spec.actionName}' was not declared", prompt.contains(spec.actionName))
        }
    }

    @Test
    fun theJsonDialectIsNotGivenGemmaTurnMarkers() {
        // Framing is dialect-specific for the same reason the parser is. A SmolLM2 package has never
        // seen <start_of_turn>developer and would treat it as content.
        val prompt = ToolPromptBuilder.prompt(allowlist, ToolCallParser.Json, "wake me at 07:30")

        assertFalse(prompt.contains("<start_of_turn>"))
        assertTrue(prompt.contains("actionName"))
        assertTrue(prompt.contains("wake me at 07:30"))
    }

    @Test
    fun theInstructionIsNeverLost() {
        for (parser in listOf(ToolCallParser.FunctionGemma, ToolCallParser.Json)) {
            val prompt = ToolPromptBuilder.prompt(allowlist, parser, "set an alarm for quarter past six")
            assertTrue(prompt.contains("set an alarm for quarter past six"))
        }
    }

    @Test
    fun aFunctionResponseRoundTripsThroughTheParsersGrammar() {
        // The second half of the loop: what the app feeds back must be in the same dialect it read.
        val response = ToolPromptBuilder.functionResponse("set_alarm", mapOf("status" to "ok"))

        assertTrue(response.startsWith("<start_function_response>response:set_alarm{"))
        assertTrue(response.contains("status:<escape>ok<escape>"))
        assertTrue(response.endsWith("<end_function_response>"))
    }
}

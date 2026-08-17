package com.martinkorelic.mobiletransformers.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #37: the accept/reject seam and the JSON extraction that feeds it.
 *
 * No Robolectric — that is the point of `Accepted.dryRun()` being a method rather than a field. The
 * decision logic is framework-free and testable here; only a caller that actually wants an `Intent`
 * pays for the framework (`IntentBinderTest`, `MobileActionsParityTest`).
 */
class ToolCallResultTest {

    private val alarm = ActionSpec(
        actionName = "set_alarm",
        parameters = mapOf("time" to "string"),
        allowedIntent = "android.intent.action.SET_ALARM",
        validationRules = mapOf("time" to "HH:mm"),
    )
    private val validator = FunctionCallValidator(listOf(alarm))

    /** What `generateToolCall` does, minus the model — kept in step with it by the assertions below. */
    private fun classify(raw: String, extractJson: Boolean = true): ToolCallResult {
        val candidate = if (extractJson) extractFirstJsonObject(raw) else raw
        return try {
            ToolCallResult.Accepted(raw, validator.validate(candidate))
        } catch (e: RejectedCallException) {
            ToolCallResult.Rejected(raw, e.message ?: "rejected")
        }
    }

    // --- extraction -----------------------------------------------------------

    @Test
    fun extractsTheCallOutOfSurroundingProse() {
        val raw = """Sure! Here you go:
            |```json
            |{"actionName": "set_alarm", "parameters": {"time": "07:30"}}
            |```
            |Anything else?
        """.trimMargin()
        assertEquals(
            """{"actionName": "set_alarm", "parameters": {"time": "07:30"}}""",
            extractFirstJsonObject(raw),
        )
        assertTrue(classify(raw) is ToolCallResult.Accepted)
    }

    @Test
    fun bracesInsideStringValuesDoNotEndTheObjectEarly() {
        // Brace-counting that ignored string literals would cut this off mid-object and report a
        // syntax error for output that is perfectly well-formed.
        val raw = """{"actionName": "set_alarm", "parameters": {"time": "07:30", "note": "a } brace"}}"""
        assertEquals(raw, extractFirstJsonObject(raw))
    }

    @Test
    fun escapedQuotesInsideStringsAreRespected() {
        val raw = """prefix {"actionName": "x", "parameters": {"s": "say \"hi\" }"}} suffix"""
        assertEquals(
            """{"actionName": "x", "parameters": {"s": "say \"hi\" }"}}""",
            extractFirstJsonObject(raw),
        )
    }

    @Test
    fun textWithNoBalancedObjectIsReturnedUnchangedSoTheValidatorReportsTheRealProblem() {
        assertEquals("I'm not sure", extractFirstJsonObject("I'm not sure"))
        assertEquals("{\"unclosed\": 1", extractFirstJsonObject("{\"unclosed\": 1"))
    }

    @Test
    fun extractionCanBeTurnedOffToDemandBareJson() {
        val raw = """Sure: {"actionName": "set_alarm", "parameters": {"time": "07:30"}}"""
        assertTrue(classify(raw, extractJson = true) is ToolCallResult.Accepted)
        assertTrue(classify(raw, extractJson = false) is ToolCallResult.Rejected)
    }

    // --- the boundary is not weakened by extraction ---------------------------

    @Test
    fun extractionCannotAdmitAnActionTheAppNeverDeclared() {
        // The whole safety question for extraction, asked directly: choosing a substring must not
        // change *which* actions are reachable.
        val raw = """Of course. {"actionName": "wipe_device", "parameters": {}}"""
        val result = classify(raw)
        assertTrue(result is ToolCallResult.Rejected)
        assertTrue((result as ToolCallResult.Rejected).reason.contains("not allowlisted"))
    }

    @Test
    fun extractionCannotBypassAValidationRule() {
        val raw = """Here: {"actionName": "set_alarm", "parameters": {"time": "25:99"}}"""
        val result = classify(raw)
        assertTrue(result is ToolCallResult.Rejected)
        assertTrue((result as ToolCallResult.Rejected).reason.contains("does not satisfy rule"))
    }

    // --- the result type ------------------------------------------------------

    @Test
    fun bothOutcomesKeepTheRawTextForDisplayAndDebugging() {
        val good = """{"actionName": "set_alarm", "parameters": {"time": "07:30"}}"""
        val bad = "no idea"
        assertEquals(good, classify(good).raw)
        assertEquals(bad, classify(bad).raw)
    }

    @Test
    fun anAcceptedCallCarriesTheAppsIntentNotTheModelsText() {
        val result = classify(
            """{"actionName": "set_alarm", "parameters": {"time": "07:30", }}"""
                .replace(", }", " }"),
        )
        assertTrue(result is ToolCallResult.Accepted)
        // Read off the ActionSpec, so it is knowable without reasoning about the model at all.
        assertEquals("android.intent.action.SET_ALARM", (result as ToolCallResult.Accepted).call.allowedIntent)
    }

    @Test
    fun rejectionNamesTheOffendingEntity() {
        val result = classify("""{"actionName": "set_alarm", "parameters": {"time": "07:30", "x": "1"}}""")
        val reason = (result as ToolCallResult.Rejected).reason
        assertTrue("message must name the parameter, not just say 'invalid': $reason", reason.contains("[x]"))
    }
}

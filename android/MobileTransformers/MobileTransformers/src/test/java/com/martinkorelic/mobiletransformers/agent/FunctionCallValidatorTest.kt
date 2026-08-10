package com.martinkorelic.mobiletransformers.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #37 self-check 2: "Does it **never** execute raw model output (allowlist + dry-run + validated tool
 * calls)?"
 *
 * These are written as the shapes a wrong-but-plausible model output actually takes, not as a happy
 * path plus one negative. A validator that only rejects malformed JSON provides no safety at all —
 * the dangerous input is well-formed JSON naming something the app never declared.
 */
class FunctionCallValidatorTest {

    private val alarm = ActionSpec(
        actionName = "set_alarm",
        parameters = mapOf("time" to "string", "label" to "string"),
        allowedIntent = "android.intent.action.SET_ALARM",
        validationRules = mapOf("time" to "HH:mm"),
        privacyClass = "harmless-demo",
    )
    private val timer = ActionSpec(
        actionName = "set_timer",
        parameters = mapOf("seconds" to "string"),
        allowedIntent = "android.intent.action.SET_TIMER",
        validationRules = mapOf("seconds" to "/[0-9]{1,4}/"),
    )
    private val validator = FunctionCallValidator(listOf(alarm, timer))

    @Test
    fun acceptsAnAllowlistedCallThatSatisfiesItsRules() {
        val call = validator.validate(
            """{"actionName": "set_alarm", "parameters": {"time": "07:30", "label": "gym"}}"""
        )

        assertEquals("set_alarm", call.actionName)
        assertEquals("android.intent.action.SET_ALARM", call.allowedIntent)
        assertEquals(mapOf("time" to "07:30", "label" to "gym"), call.parameters)
    }

    @Test
    fun rejectsAnActionTheAppNeverDeclared() {
        // The dangerous input: perfectly well-formed JSON asking for something else entirely.
        val error = assertThrows(RejectedCallException::class.java) {
            validator.validate("""{"actionName": "wipe_device", "parameters": {}}""")
        }
        assertTrue(error.message!!.contains("not allowlisted"))
        assertTrue("the message must name the offending action", error.message!!.contains("wipe_device"))
    }

    @Test
    fun rejectsAnIntentSmuggledInAsAParameter() {
        // A model cannot introduce an intent: `allowedIntent` is read from the app's spec, and an
        // undeclared parameter is refused outright rather than ignored.
        val error = assertThrows(RejectedCallException::class.java) {
            validator.validate(
                """{"actionName": "set_alarm", "parameters": {"time": "07:30", "label": "x",
                   "allowedIntent": "android.intent.action.CALL"}}"""
            )
        }
        assertTrue(error.message!!.contains("does not declare parameter"))
    }

    @Test
    fun rejectsAValueThatBreaksItsRule() {
        val error = assertThrows(RejectedCallException::class.java) {
            validator.validate("""{"actionName": "set_alarm", "parameters": {"time": "25:99", "label": "x"}}""")
        }
        assertTrue(error.message!!.contains("does not satisfy rule"))
        assertTrue(error.message!!.contains("25:99"))
    }

    @Test
    fun rejectsAMissingParameterRatherThanDefaultingIt() {
        val error = assertThrows(RejectedCallException::class.java) {
            validator.validate("""{"actionName": "set_alarm", "parameters": {"time": "07:30"}}""")
        }
        assertTrue(error.message!!.contains("missing parameter"))
        assertTrue(error.message!!.contains("label"))
    }

    @Test
    fun rejectsMalformedOrEmptyOutput() {
        for (raw in listOf("not json at all {", "", "   ")) {
            assertThrows(
                "raw output '$raw' must be rejected",
                RejectedCallException::class.java,
            ) { validator.validate(raw) }
        }
    }

    @Test
    fun rejectsJsonWithoutAnActionName() {
        val error = assertThrows(RejectedCallException::class.java) {
            validator.validate("""{"parameters": {"time": "07:30"}}""")
        }
        assertTrue(error.message!!.contains("actionName"))
    }

    @Test
    fun anUnrecognisedRuleRejectsRatherThanPassingByDefault() {
        // A typo in the app's allowlist must not silently disable the check it was written to perform.
        val typo = FunctionCallValidator(
            listOf(alarm.copy(validationRules = mapOf("time" to "HH:MM")))  // wrong case
        )

        assertThrows(RejectedCallException::class.java) {
            typo.validate("""{"actionName": "set_alarm", "parameters": {"time": "07:30", "label": "x"}}""")
        }
    }

    @Test
    fun aRegexRuleIsAnchoredSoAPrefixDoesNotSlipThrough() {
        assertEquals(
            "60",
            validator.validate("""{"actionName": "set_timer", "parameters": {"seconds": "60"}}""")
                .parameters["seconds"],
        )
        // `Regex.matches` is whole-string; a trailing payload must not be accepted.
        assertThrows(RejectedCallException::class.java) {
            validator.validate("""{"actionName": "set_timer", "parameters": {"seconds": "60; rm -rf /"}}""")
        }
    }

    @Test
    fun aDuplicatedActionNameFailsAtConstructionNotSilently() {
        // `associateBy` keeps the last silently, so two rows disagreeing about what is permitted would
        // resolve to whichever came last in the list.
        val error = assertThrows(IllegalArgumentException::class.java) {
            FunctionCallValidator(listOf(alarm, alarm.copy(allowedIntent = "android.intent.action.CALL")))
        }
        assertTrue(error.message!!.contains("duplicate action names"))
    }

    @Test
    fun theReachableIntentSetIsFixedByTheAllowlist() {
        // The property that makes this a boundary: every intent any accepted call can produce is one
        // the app declared, so the set is knowable without reasoning about the model at all.
        assertEquals(setOf("set_alarm", "set_timer"), validator.allowedActions)
    }
}

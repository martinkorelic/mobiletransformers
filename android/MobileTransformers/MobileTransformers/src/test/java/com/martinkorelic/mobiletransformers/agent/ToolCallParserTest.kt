package com.martinkorelic.mobiletransformers.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #37: reading a tool call out of whatever dialect the model speaks.
 *
 * The case that forced this to exist: `FunctionCallValidator` parsed JSON, and **FunctionGemma does
 * not emit JSON**. Its calls look like
 * `<start_function_call>call:set_alarm{time:<escape>07:30<escape>}<end_function_call>`, which a JSON
 * parser reports as a syntax error — so a correctly fine-tuned FunctionGemma had every well-formed
 * call rejected, and no further training could have changed that.
 *
 * The samples below are Google's documented ones, kept verbatim so the parser is checked against the
 * specification rather than against our own idea of it.
 */
class ToolCallParserTest {

    private val gemma = ToolCallParser.FunctionGemma
    private val json = ToolCallParser.Json

    // --- FunctionGemma ---------------------------------------------------------

    @Test
    fun parsesTheDocumentedFunctionGemmaCall() {
        val raw = "<start_function_call>call:get_current_weather" +
            "{location:<escape>Tokyo, Japan<escape>}<end_function_call>"

        val call = gemma.parse(raw)
        assertNotNull(call)
        assertEquals("get_current_weather", call!!.actionName)
        // The comma is INSIDE the value. `<escape>` exists precisely so it can be, and a parser that
        // splits fields on every comma turns this into two parameters, the second named " Japan".
        assertEquals(mapOf("location" to "Tokyo, Japan"), call.parameters)
    }

    @Test
    fun aColonInsideAValueIsNotAFieldSeparator() {
        val call = gemma.parse("<start_function_call>call:set_alarm{time:<escape>07:30<escape>}<end_function_call>")
        assertEquals(mapOf("time" to "07:30"), call?.parameters)
    }

    @Test
    fun parsesSeveralParametersIncludingBareValues() {
        val call = gemma.parse(
            "<start_function_call>call:set_timer{seconds:90,label:<escape>tea, strong<escape>}" +
                "<end_function_call>",
        )
        // A bare numeric keeps its text form: `validationRules` are regexes over strings, so
        // "/[0-9]{1,4}/" must see "90" rather than a parsed number.
        assertEquals(mapOf("seconds" to "90", "label" to "tea, strong"), call?.parameters)
    }

    @Test
    fun anActionWithNoParametersParses() {
        val call = gemma.parse("<start_function_call>call:open_wifi_settings{}<end_function_call>")
        assertEquals("open_wifi_settings", call?.actionName)
        assertEquals(emptyMap<String, String>(), call?.parameters)
    }

    /**
     * Generation stops at `maxNewTokens`, so a call whose end token never arrived is routine. What was
     * emitted is still the model's answer, and refusing it reports a model failure for what is really
     * a configuration choice.
     */
    @Test
    fun aTruncatedCallStillParses() {
        val call = gemma.parse("<start_function_call>call:set_alarm{time:<escape>07:30<escape>}")
        assertEquals("set_alarm", call?.actionName)
        assertEquals(mapOf("time" to "07:30"), call?.parameters)
    }

    @Test
    fun aClosingBraceInsideAValueDoesNotEndTheCall() {
        val call = gemma.parse(
            "<start_function_call>call:note{body:<escape>use {braces} freely<escape>,tag:x}<end_function_call>",
        )
        assertEquals(mapOf("body" to "use {braces} freely", "tag" to "x"), call?.parameters)
    }

    @Test
    fun proseAroundTheCallIsIgnored() {
        val call = gemma.parse(
            "Sure, I'll do that.\n<start_function_call>call:set_alarm{time:<escape>08:00<escape>}" +
                "<end_function_call>\nAnything else?",
        )
        assertEquals("set_alarm", call?.actionName)
    }

    @Test
    fun textWithNoCallYieldsNull() {
        assertNull(gemma.parse("I'm sorry, I can't help with that."))
        assertNull(gemma.parse(""))
    }

    // --- JSON ------------------------------------------------------------------

    @Test
    fun jsonParserReadsAFencedObject() {
        val call = json.parse("""Sure! ```json {"actionName":"set_alarm","parameters":{"time":"07:30"}} ```""")
        assertEquals("set_alarm", call?.actionName)
        assertEquals(mapOf("time" to "07:30"), call?.parameters)
    }

    @Test
    fun jsonParserYieldsNullOnFunctionGemmaOutput() {
        // The regression itself: this is a perfectly good call that the JSON reader cannot see.
        assertNull(json.parse("<start_function_call>call:set_alarm{time:<escape>07:30<escape>}<end_function_call>"))
    }

    @Test
    fun jsonParserYieldsNullOnProseWithNoObject() {
        assertNull(json.parse("I can't do that."))
    }

    // --- selection -------------------------------------------------------------

    @Test
    fun theParserIsChosenFromTheModelFamily() {
        assertEquals(ToolCallParser.FunctionGemma, ToolCallParser.forModel("mobiletransformers/functiongemma-270m-it"))
        assertEquals(ToolCallParser.FunctionGemma, ToolCallParser.forModel("google/FunctionGemma-270M-IT"))
        assertEquals(ToolCallParser.Json, ToolCallParser.forModel("HuggingFaceTB/SmolLM2-135M-Instruct"))
        assertEquals(ToolCallParser.Json, ToolCallParser.forModel(null))
    }

    // --- the boundary is unchanged ---------------------------------------------

    /**
     * The point of the whole seam: a new parser must widen what can be *recognised*, never what can be
     * *permitted*. A FunctionGemma call naming an action the app never declared is still refused.
     */
    @Test
    fun anUndeclaredActionIsStillRejectedWhicheverDialectItArrivesIn() {
        val validator = FunctionCallValidator(
            listOf(ActionSpec(actionName = "set_alarm", parameters = mapOf("time" to "string"), allowedIntent = "X")),
        )
        val call = gemma.parse("<start_function_call>call:wipe_device{}<end_function_call>")!!
        val error = assertThrows(RejectedCallException::class.java) { validator.validate(call) }
        assertTrue(error.message!!.contains("not allowlisted"))
    }

    @Test
    fun aParsedCallStillHasToSatisfyItsValidationRules() {
        val validator = FunctionCallValidator(
            listOf(
                ActionSpec(
                    actionName = "set_alarm",
                    parameters = mapOf("time" to "string"),
                    allowedIntent = "X",
                    validationRules = mapOf("time" to "HH:mm"),
                ),
            ),
        )
        val bad = gemma.parse("<start_function_call>call:set_alarm{time:<escape>quarter past<escape>}<end_function_call>")!!
        assertThrows(RejectedCallException::class.java) { validator.validate(bad) }

        val good = gemma.parse("<start_function_call>call:set_alarm{time:<escape>07:30<escape>}<end_function_call>")!!
        assertEquals("set_alarm", validator.validate(good).actionName)
    }
}

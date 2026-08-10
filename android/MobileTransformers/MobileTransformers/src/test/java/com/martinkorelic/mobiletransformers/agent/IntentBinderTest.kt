package com.martinkorelic.mobiletransformers.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * #37 self-check 2, the binding half: the intended action is produced, and nothing runs it.
 *
 * Robolectric because `android.content.Intent` is a real framework class; the plain unit-test classpath
 * stubs it, and `isReturnDefaultValues = false` makes that stub throw rather than quietly return null —
 * which is what would otherwise let these assertions "pass" against a method that never ran.
 */
@RunWith(RobolectricTestRunner::class)
class IntentBinderTest {

    private val alarm = ActionSpec(
        actionName = "set_alarm",
        parameters = mapOf("time" to "string", "label" to "string"),
        allowedIntent = "android.intent.action.SET_ALARM",
        validationRules = mapOf("time" to "HH:mm"),
        privacyClass = "harmless-demo",
    )
    private val validator = FunctionCallValidator(listOf(alarm))

    private fun accepted() = validator.validate(
        """{"actionName": "set_alarm", "parameters": {"time": "07:30", "label": "gym"}}"""
    )

    @Test
    fun buildsTheDeclaredIntentWithTheValidatedParametersAsExtras() {
        val action = IntentBinder.dryRun(accepted())

        assertEquals("android.intent.action.SET_ALARM", action.intent.action)
        assertEquals("07:30", action.intent.getStringExtra("time"))
        assertEquals("gym", action.intent.getStringExtra("label"))
    }

    @Test
    fun dryRunIsNeverMarkedExecutable() {
        // The flag exists so a caller that chooses to execute must read it and act on it, rather than
        // executing because the object happened to contain an Intent.
        assertFalse(IntentBinder.dryRun(accepted()).willExecute)
    }

    @Test
    fun theIntentActionComesFromTheAppsSpecNotFromModelOutput() {
        // The property that makes this safe: a model selects an ACTION, it does not name an INTENT.
        // Even a spec whose action name looks hostile yields only the intent the app declared.
        val odd = ActionSpec(
            actionName = "android.intent.action.CALL",  // a name, not an intent
            parameters = emptyMap(),
            allowedIntent = "android.intent.action.SET_ALARM",
        )
        val call = FunctionCallValidator(listOf(odd))
            .validate("""{"actionName": "android.intent.action.CALL", "parameters": {}}""")

        assertEquals("android.intent.action.SET_ALARM", IntentBinder.dryRun(call).intent.action)
    }

    @Test
    fun onlyDeclaredParameterKeysReachTheExtras() {
        // The validator refuses undeclared parameters, so no model-chosen key can appear here. Asserting
        // it across the seam rather than trusting the validator's own test.
        val action = IntentBinder.dryRun(accepted())

        assertNull(action.intent.getStringExtra("allowedIntent"))
        assertNull(action.intent.getStringExtra("uri"))
    }

    @Test
    fun theBinderCarriesNoContextSoItCannotStartAnything() {
        // Structural, not behavioural: `IntentBinder` is an object with no Context field and no
        // startActivity call site. Executing is the caller's decision with the caller's own Context —
        // deliberately not offered here, because the convenience is the risk.
        val fields = IntentBinder::class.java.declaredFields.map { it.type.name }

        assertFalse(
            "IntentBinder must not hold a Context",
            fields.any { it.contains("android.content.Context") },
        )
    }
}

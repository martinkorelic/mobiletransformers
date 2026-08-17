package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.agent.ActionSpec

/**
 * Everything this app will let a model ask for, and the permission each one needs.
 *
 * This object *is* the security boundary. A model selects an action by name; it can never name an
 * intent, because intent strings appear only here. So the set of intents any model output can reach
 * is fixed when this list is written, and widening it is a deliberate edit to this file rather than
 * anything a model can talk its way into.
 *
 * It also feeds the training corpus: `mobiletransformers agent-dataset` writes this same declaration
 * out as `action_schema.json`, so the examples a model is fine-tuned on and the boundary that judges
 * its output are provably one value rather than two lists that agree today.
 *
 * ### Why permissions live here
 *
 * An action and the permission its intent needs are one fact. `set_alarm` without
 * `com.android.alarm.permission.SET_ALARM` is an entry that always fails: the model emits a correct
 * call, the validator accepts it, and `startActivity` throws `SecurityException` at the last possible
 * moment — which a user reads as "the feature is broken", not as "a line is missing from the
 * manifest". That is exactly what happened before these were declared.
 *
 * Android has two kinds and they behave differently:
 *
 * - **Install-time (`normal`)** — `SET_ALARM`, the only kind used here. Granted when the app is
 *   installed, provided the manifest declares it. There is no dialog and there never will be, so a UI
 *   that promises to "ask" for it is lying, and a missing one is a manifest bug rather than a user
 *   decision.
 * - **Runtime (`dangerous`)** — a system dialog the user can refuse. The app handles these generically
 *   ([PermissionGate]) so adding such an action needs no new plumbing.
 *
 * **No action here needs a runtime permission, and that is not an oversight.** Intent-based actions
 * delegate the sensitive work to the target app, which enforces its own permissions behind its own UI
 * — that is the point of the design. So the benign actions worth showcasing (an alarm, a timer, a
 * settings screen) are all install-time, and manufacturing a dangerous one purely to make a dialog
 * appear would be a demo of nothing. The app's real runtime-permission prompt is `POST_NOTIFICATIONS`,
 * requested when a training run starts.
 *
 * Every permission named here must also be declared in `AndroidManifest.xml`; the manifest is what
 * grants the install-time kind and what makes the runtime kind requestable at all.
 */
object ActionAllowlist {

    /** Required by both `SET_ALARM` and `SET_TIMER`. Install-time: declared, never prompted for. */
    const val ALARM_PERMISSION = "com.android.alarm.permission.SET_ALARM"

    val ENTRIES: List<ActionSpec> = listOf(
        ActionSpec(
            actionName = "set_alarm",
            parameters = mapOf("time" to "string"),
            allowedIntent = "android.intent.action.SET_ALARM",
            validationRules = mapOf("time" to "HH:mm"),
            privacyClass = "harmless-demo",
            requiredPermissions = listOf(ALARM_PERMISSION),
        ),
        ActionSpec(
            actionName = "set_timer",
            parameters = mapOf("seconds" to "string"),
            allowedIntent = "android.intent.action.SET_TIMER",
            validationRules = mapOf("seconds" to "/[0-9]{1,4}/"),
            privacyClass = "harmless-demo",
            requiredPermissions = listOf(ALARM_PERMISSION),
        ),
        ActionSpec(
            actionName = "open_wifi_settings",
            parameters = emptyMap(),
            allowedIntent = "android.settings.WIFI_SETTINGS",
            privacyClass = "harmless-demo",
        ),
    )

    /** Every permission any allowed action can need — what the manifest must declare. */
    val ALL_PERMISSIONS: Set<String> = ENTRIES.flatMap { it.requiredPermissions }.toSet()
}

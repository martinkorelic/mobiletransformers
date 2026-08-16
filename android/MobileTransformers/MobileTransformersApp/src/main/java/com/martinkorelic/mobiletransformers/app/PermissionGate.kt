package com.martinkorelic.mobiletransformers.app

import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat

/**
 * Whether the app may actually start an allowed action's intent, and what to do when it may not.
 *
 * Firing an intent used to be a `runCatching { startActivity(...) }`, so a missing permission arrived
 * as a `SecurityException` after the user had already tapped Run — an exception used to answer a
 * question that could have been asked first. This asks first.
 *
 * ### Why the two outcomes are not the same
 *
 * A missing permission means opposite things depending on its protection level, and telling a user the
 * wrong one wastes their time:
 *
 * - **Install-time** (`com.android.alarm.permission.SET_ALARM`) is granted at install *if the manifest
 *   declares it*. If it is missing at runtime, no dialog will ever fix that — it is a build defect,
 *   and the honest message says so rather than inviting the user to grant something they cannot.
 * - **Runtime** is the user's decision, and the system dialog is the right response.
 *
 * The two are distinguished by asking the platform for the permission's own protection level rather
 * than by keeping a list here, so a permission added to [ActionAllowlist] is classified correctly
 * without this file changing.
 */
object PermissionGate {

    /** Permissions in [required] that are not currently granted. */
    fun missing(context: Context, required: List<String>): List<String> =
        required.filter {
            ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
        }

    /**
     * Whether [permission] is one the system will show a dialog for.
     *
     * Reads the platform's own `protectionLevel`. An unknown permission — one the device has never
     * heard of — reports `false`: a dialog for it would be dismissed instantly, and calling it a
     * configuration problem is both true and actionable.
     */
    fun isRuntimePermission(context: Context, permission: String): Boolean = runCatching {
        val info = context.packageManager.getPermissionInfo(permission, 0)
        @Suppress("DEPRECATION")
        val level = info.protectionLevel and android.content.pm.PermissionInfo.PROTECTION_MASK_BASE
        level == android.content.pm.PermissionInfo.PROTECTION_DANGEROUS
    }.getOrDefault(false)

    /**
     * Split [missing] into the ones worth prompting for and the ones that are a build defect.
     *
     * @return `requestable` — show the system dialog for these; `undeclared` — no dialog can help,
     *   the manifest is wrong.
     */
    fun classify(context: Context, missing: List<String>): Pair<List<String>, List<String>> =
        missing.partition { isRuntimePermission(context, it) }

    /** A message naming what is wrong, for permissions no dialog can resolve. */
    fun undeclaredMessage(permissions: List<String>): String =
        "this build is missing ${permissions.joinToString()} in its AndroidManifest — an install-time " +
            "permission cannot be granted from here, so the app has to declare it and be reinstalled"
}

package com.martinkorelic.mobiletransformers.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat

/**
 * Ask for `POST_NOTIFICATIONS` once, on first composition.
 *
 * ### Why this had to be added
 *
 * The SDK declares the permission in its own manifest — deliberately, so a consumer that calls
 * `TrainingScheduler.schedule()` inherits it — and `TrainingWorker` posts a foreground notification
 * with a Cancel action. But **nothing ever requested it**. Since API 33 the permission is runtime-
 * granted and denied by default, so on every modern device the training notification was constructed,
 * handed to the system, and silently dropped. The worker still ran; the user simply had no way to see
 * that it was running or to stop it — which is the whole point of promoting it to the foreground.
 *
 * Declaring a permission and requesting it are different acts, and only one of them was happening.
 *
 * Below API 33 the permission does not exist and is granted implicitly, so this is a no-op there.
 * A denial is not fatal and not re-prompted here: Android stops showing the dialog after two refusals
 * anyway, and the About screen links to system settings for anyone who changes their mind.
 */
@Composable
fun RequestNotificationPermissionOnce() {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return

    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { /* granted or not, the app works either way — only visibility changes */ }

    LaunchedEffect(Unit) {
        if (!hasNotificationPermission(context)) {
            launcher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }
}

/**
 * Whether background progress can actually be shown.
 *
 * Surfaced so a screen offering scheduled training can say "this will run without a visible
 * notification" rather than implying an ongoing notification the system will discard.
 */
fun hasNotificationPermission(context: Context): Boolean =
    Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) ==
        PackageManager.PERMISSION_GRANTED

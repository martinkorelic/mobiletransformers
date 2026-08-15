package com.martinkorelic.mobiletransformers.app.views

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.hasNotificationPermission

/**
 * What this app is, in what order to use it, and the two device settings that change what it can show.
 *
 * The tour used to live in a `Guide` card on the Models screen, which meant the one explanation of how
 * the app fits together was reachable only from the screen a user had already worked out. It belongs
 * somewhere addressable — and the drawer now makes "somewhere" cheap.
 */
@Composable
fun AboutScreen(onGoToModels: () -> Unit) {
    val context = LocalContext.current
    val notificationsOn = hasNotificationPermission(context)

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        ScreenIntro(
            "MobileTransformers runs, fine-tunes and retrieves with language models entirely on this " +
                "device. Nothing you type, ingest or train on is uploaded — the only network traffic " +
                "is pulling a model package from the Hub.",
        )

        Section("The order things work in") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Step(
                    "1 · Models",
                    "Pick a package from the catalog or enter any Hub repo id. Nothing else works " +
                        "until one is installed. Tick Training if you want the Training and Tool " +
                        "calls screens; tick RAG for grounding in Chat — each is a separate download " +
                        "group and cannot be added later without re-pulling.",
                )
                Step(
                    "2 · Chat",
                    "Type and send; tokens stream as they arrive. A small model on a phone is slow " +
                        "and not very fluent — that is the hardware and the model size, not a defect.",
                )
                Step(
                    "3 · Training",
                    "Install the sample dataset first: packages ship no training data by design, " +
                        "because the task belongs with the data and the data is yours. Then Start, " +
                        "and watch the loss curve. Merge writes what was learned into the inference " +
                        "graph.",
                )
                Step(
                    "4 · Tool calls",
                    "Turn an instruction into a validated call bound to a real Android intent — in " +
                        "dry-run only, always. On a model you have not fine-tuned, Rejected is the " +
                        "correct answer rather than a bug.",
                )
                Button(onClick = onGoToModels) { Text("Start at Models") }
            }
        }

        Section("Device settings that affect what you see") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    if (notificationsOn) {
                        "Notifications are allowed, so scheduled training and background downloads " +
                            "show their progress and can be cancelled from the shade."
                    } else {
                        "Notifications are blocked. Training still runs in the background, but its " +
                            "ongoing notification — the only place its progress and Cancel button " +
                            "appear while you are outside the app — will not be shown."
                    },
                    style = MaterialTheme.typography.bodySmall,
                )
                if (!notificationsOn && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    OutlinedButton(onClick = {
                        context.startActivity(
                            Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                                .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }) { Text("Open notification settings") }
                }
                Text(
                    "Scheduled training runs only while charging and idle, and re-checks that on " +
                        "every chunk — so unplugging pauses a run instead of failing it.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Section("Bringing your own model") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Any package exported with `mobiletransformers export` and pushed to the Hub can " +
                        "be pulled here by repo id. The catalog on the Models screen is a bundled " +
                        "JSON file — adding an entry to it is editing one asset, not writing code.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "Every screen in this app talks to the SDK only through " +
                        "MobileTransformers.fromPretrained and MobileTransformerModel. A build guard " +
                        "fails if any screen names an engine-layer type, which is what makes this a " +
                        "worked example of the public API rather than a claim about one.",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedButton(onClick = {
                    context.startActivity(
                        Intent(
                            Intent.ACTION_VIEW,
                            Uri.parse("https://github.com/martinkorelic/mobiletransformers"),
                        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                    )
                }) { Text("Documentation and source") }
            }
        }
    }
}

@Composable
private fun Step(title: String, body: String) {
    Column(Modifier.padding(bottom = 4.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge)
        Text(body, style = MaterialTheme.typography.bodySmall)
    }
}

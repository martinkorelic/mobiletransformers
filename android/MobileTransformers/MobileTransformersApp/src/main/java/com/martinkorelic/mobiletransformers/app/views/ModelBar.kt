package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.DownloadUi
import com.martinkorelic.mobiletransformers.app.ModelState

/**
 * A one-line answer to "what is loaded and what can it do", pinned under the app bar on every screen.
 *
 * ### Why the app needed this
 *
 * The loaded model was reported in exactly one place — a "Current model" card partway down the Models
 * screen. Everywhere else the model was invisible, so the answer to "why did Chat just refuse me" or
 * "is this the package I trained" required navigating away from the thing that raised the question.
 * Worse, a pull in progress was equally invisible: leaving Models mid-download looked exactly like no
 * download running.
 *
 * Every value here already existed on `RuntimeCapabilities`; none of it was ever put in front of the
 * user. The bar is deliberately dense and tappable rather than complete — the sheet behind it carries
 * the full detail.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelBar(state: ModelState, download: DownloadUi?, onUnload: () -> Unit, onGoToModels: () -> Unit) {
    var sheetOpen by remember { mutableStateOf(false) }

    Column(
        Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .clickable { sheetOpen = true }
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            StatusDot(state)
            Text(
                text = when (state) {
                    is ModelState.None -> "No model loaded"
                    is ModelState.Loading -> state.repoId
                    is ModelState.Failed -> state.repoId
                    is ModelState.Loaded -> state.model.repoId
                },
                style = MaterialTheme.typography.titleSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            if (state is ModelState.Loaded) {
                Text(state.model.capabilities.engine.name, style = MaterialTheme.typography.labelSmall)
            }
        }

        when (state) {
            is ModelState.Loaded -> {
                val c = state.model.capabilities
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    // Only what this package can actually do. A chip that is present but greyed out
                    // would say "almost" about a capability that is simply absent.
                    c.task.taskType?.let { CapabilityChip(it.wire) }
                    if (c.supportsTraining) CapabilityChip("train")
                    if (c.supportsRag) CapabilityChip("rag")
                    if (c.supportsClassification) CapabilityChip("classify")
                }
            }

            is ModelState.Loading -> {
                // The download that used to be visible only on the screen that started it.
                if (download != null) {
                    Text(
                        "${downloadPhaseLabel(download.phase)} · ${download.summary}",
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                val fraction = download?.fraction
                if (fraction != null) {
                    LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
                } else {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
            }

            is ModelState.Failed -> Text(
                state.reason,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )

            is ModelState.None -> Text(
                "Tap to see how to load one",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
    HorizontalDivider()

    if (sheetOpen) {
        ModalBottomSheet(onDismissRequest = { sheetOpen = false }) {
            ModelDetail(
                state = state,
                onUnload = { sheetOpen = false; onUnload() },
                onGoToModels = { sheetOpen = false; onGoToModels() },
            )
        }
    }
}

@Composable
private fun StatusDot(state: ModelState) {
    val color = when (state) {
        is ModelState.Loaded -> MaterialTheme.colorScheme.primary
        is ModelState.Loading -> MaterialTheme.colorScheme.tertiary
        is ModelState.Failed -> MaterialTheme.colorScheme.error
        is ModelState.None -> MaterialTheme.colorScheme.outline
    }
    Column(Modifier.size(10.dp).clip(CircleShape).background(color)) {}
}

@Composable
private fun CapabilityChip(label: String) {
    AssistChip(
        onClick = { },
        label = { Text(label, style = MaterialTheme.typography.labelSmall) },
        colors = AssistChipDefaults.assistChipColors(labelColor = MaterialTheme.colorScheme.onSurface),
        border = null,
    )
}

/** Everything the bar had to abbreviate, plus the actions that belong with it. */
@Composable
private fun ModelDetail(state: ModelState, onUnload: () -> Unit, onGoToModels: () -> Unit) {
    Column(
        Modifier.fillMaxWidth().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        when (state) {
            is ModelState.Loaded -> {
                val c = state.model.capabilities
                Text(state.model.repoId, style = MaterialTheme.typography.titleMedium)
                DetailRow("task", c.task.declaredTask ?: "not declared by this package")
                DetailRow("architecture", c.task.modelType ?: "unknown")
                DetailRow("engine", c.engine.name)
                DetailRow("engines available here", c.availableEngines.joinToString())
                DetailRow("features installed", c.availableFeatures.joinToString().ifEmpty { "none" })
                DetailRow("training", if (c.supportsTraining) "yes" else "no train/ stage installed")
                DetailRow("retrieval", if (c.supportsRag) "yes" else "no embedding stage installed")
                DetailRow(
                    "classification",
                    when {
                        c.supportsClassification -> "${c.task.labelCount} labels"
                        c.isClassifier -> "classifier, but the package names no labels"
                        else -> "not a classification model"
                    },
                )
                DetailRow("scheduled training", if (c.supportsScheduledTraining) "yes" else "no")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onUnload) { Text("Unload") }
                    TextButton(onClick = onGoToModels) { Text("Manage models") }
                }
            }

            is ModelState.Loading -> {
                Text("Loading ${state.repoId}", style = MaterialTheme.typography.titleMedium)
                Text(
                    "A first pull is 1–4 GB and needs roughly its own size again free while it " +
                        "installs. Progress resumes if it is interrupted.",
                    style = MaterialTheme.typography.bodySmall,
                )
                TextButton(onClick = onGoToModels) { Text("Go to Models") }
            }

            is ModelState.Failed -> {
                Text("Could not load ${state.repoId}", style = MaterialTheme.typography.titleMedium)
                // Verbatim: the SDK's exceptions name the missing feature or artifact.
                Text(state.reason, style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = onGoToModels) { Text("Go to Models") }
            }

            is ModelState.None -> {
                Text("No model loaded", style = MaterialTheme.typography.titleMedium)
                Text(
                    "Nothing else in the app can do anything until a package is installed. Pick one " +
                        "from the catalog on the Models screen, or enter any Hub repo id.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                TextButton(onClick = onGoToModels) { Text("Go to Models") }
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(0.45f),
        )
        Text(value, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(0.55f))
    }
}

/** Mirrors `DownloadProgress.Phase` without importing it into the composable layer. */
internal fun downloadPhaseLabel(phase: String): String = when (phase) {
    "Resolving" -> "Resolving"
    "Verifying" -> "Verifying"
    "Installing" -> "Installing"
    else -> "Downloading"
}

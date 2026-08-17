package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.DownloadUi
import com.martinkorelic.mobiletransformers.app.viewmodels.peftDisplayName
import com.martinkorelic.mobiletransformers.app.ModelActivity
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.ui.theme.statusColors

/**
 * A one-line answer to "what is loaded, is it busy, and what can it do", pinned under the app bar on
 * every screen.
 *
 * ### Why the app needed this
 *
 * The loaded model was reported in exactly one place — a "Current model" card partway down the Models
 * screen. Everywhere else the model was invisible, so the answer to "why did Chat just refuse me" or
 * "is this the package I trained" required navigating away from the thing that raised the question.
 * Worse, a pull in progress was equally invisible: leaving Models mid-download looked exactly like no
 * download running.
 *
 * ### Three things this got wrong, all now fixed
 *
 * - **The dot was painted `primary` when loaded**, and `primary` in this theme is the project red. A
 *   healthy, idle model therefore showed the colour every user reads as "stop" — while a model that
 *   was genuinely busy showed exactly the same thing, because [ModelState] cannot tell those apart.
 *   It takes [ModelActivity] now: green when the model is free, red while it works.
 * - **The chips were `AssistChip`s**, which are 32dp tall touch targets with 16dp of internal padding
 *   each, built for actions. Four of them ate a third of the bar to say "text-generation, train,
 *   rag" — labels, not buttons, and nothing happened when you pressed one. They are flat badges now.
 * - **The repo id was ellipsised to one line.** `mobiletransformers/functiongemma-270m-it` is the
 *   answer to "which model am I talking to", and it was the part of the bar most likely to be cut.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ModelBar(
    state: ModelState,
    activity: ModelActivity,
    download: DownloadUi?,
    onUnload: () -> Unit,
    onGoToModels: () -> Unit,
) {
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
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            StatusDot(state, activity, Modifier.padding(top = 5.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    text = when (state) {
                        is ModelState.None -> "No model loaded"
                        is ModelState.Loading -> state.repoId
                        is ModelState.Failed -> state.repoId
                        is ModelState.Loaded -> state.model.repoId
                    },
                    style = MaterialTheme.typography.labelLarge,
                    // Wraps rather than truncates: a repo id cut at "mobiletransformers/functiong…"
                    // does not identify a model. Two lines is enough for every id in the catalog.
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    statusLine(state, activity),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (state is ModelState.Loaded) {
                Badge(state.model.capabilities.engine.name, Modifier.padding(top = 2.dp))
            }
        }

        when (state) {
            is ModelState.Loaded -> {
                val c = state.model.capabilities
                // FlowRow, so a package with five capabilities wraps instead of pushing the last one
                // off the right edge.
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    // Only what this package can actually do. A chip that is present but greyed out
                    // would say "almost" about a capability that is simply absent.
                    c.task.taskType?.let { Badge(it.wire) }
                    if (c.supportsToolCalling) Badge("tools")
                    if (c.supportsTraining) Badge("train")
                    if (c.supportsRag) Badge("rag")
                    if (c.supportsClassification) Badge("classify")
                    // Which fine-tuning technique this package carries. MARS is the project's own
                    // method, and until now nothing in the app said which one you were running.
                    c.primaryPeftMethod?.let { Badge(peftDisplayName(it)) }
                }
            }

            is ModelState.Loading -> {
                // The download that used to be visible only on the screen that started it.
                if (download != null) {
                    Text(
                        "${downloadPhaseLabel(download.phase, download.waitingForConstraints)} · ${download.summary}",
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
                maxLines = 3,
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
                activity = activity,
                onUnload = { sheetOpen = false; onUnload() },
                onGoToModels = { sheetOpen = false; onGoToModels() },
            )
        }
    }
}

/**
 * Green when the model can take work, red while it cannot.
 *
 * The rule the user asked for, and the only one a single dot can carry: *busy* covers loading,
 * generating, training and merging alike, because from the outside they are the same fact — asking
 * for something now will queue behind what is already running.
 */
@Composable
internal fun statusColorFor(state: ModelState, activity: ModelActivity): Color {
    val status = MaterialTheme.statusColors
    return when {
        state is ModelState.Failed -> status.failed
        state is ModelState.None -> status.idle
        state is ModelState.Loading -> status.busy
        activity.isBusy -> status.busy
        else -> status.ready
    }
}

/** The words beside the dot, so the colour is never the only carrier of the state. */
internal fun statusLine(state: ModelState, activity: ModelActivity): String = when (state) {
    is ModelState.None -> "nothing loaded"
    is ModelState.Loading -> "loading"
    is ModelState.Failed -> "failed to load"
    is ModelState.Loaded -> loadedStatusLine(activity)
}

/**
 * The loaded case, split out so it is reachable from a test.
 *
 * A `ModelState.Loaded` carries a real `MobileTransformerModel`, which owns a native session — there
 * is no way to construct one on the JVM, and `ModelState` is sealed to the main source set so it
 * cannot be faked either. The branch that matters most would otherwise be the one branch with no
 * coverage.
 */
internal fun loadedStatusLine(activity: ModelActivity): String =
    if (activity.isBusy) "busy · ${activity.label}" else "ready"

@Composable
private fun StatusDot(state: ModelState, activity: ModelActivity, modifier: Modifier = Modifier) {
    val color = statusColorFor(state, activity)
    val description = statusLine(state, activity)
    Box(
        modifier
            .size(10.dp)
            .clip(CircleShape)
            .background(color)
            // Colour alone is not a status for anyone who cannot distinguish red from green.
            .semantics { contentDescription = description },
    )
}

/**
 * A capability label.
 *
 * Not a chip: nothing happens when you press it. `AssistChip` gave each of these a 32dp height and
 * 16dp of horizontal padding — the geometry of a button — so four labels consumed a third of the bar
 * and invited a tap that does nothing.
 */
@Composable
private fun Badge(label: String, modifier: Modifier = Modifier) {
    Text(
        label,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = modifier
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(4.dp))
            .padding(horizontal = 6.dp, vertical = 1.dp),
    )
}

/** Everything the bar had to abbreviate, plus the actions that belong with it. */
@Composable
private fun ModelDetail(
    state: ModelState,
    activity: ModelActivity,
    onUnload: () -> Unit,
    onGoToModels: () -> Unit,
) {
    Column(
        Modifier.fillMaxWidth().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        when (state) {
            is ModelState.Loaded -> {
                val c = state.model.capabilities
                Text(state.model.repoId, style = MaterialTheme.typography.titleMedium)
                DetailRow("status", statusLine(state, activity))
                DetailRow("task", c.task.declaredTask ?: "not declared by this package")
                DetailRow("architecture", c.task.modelType ?: "unknown")
                DetailRow("engine", c.engine.name)
                DetailRow("engines available here", c.availableEngines.joinToString())
                DetailRow("features installed", c.availableFeatures.joinToString().ifEmpty { "none" })
                DetailRow("training", if (c.supportsTraining) "yes" else "no train/ stage installed")
                DetailRow(
                    "fine-tuning method",
                    c.peftMethods.joinToString { peftDisplayName(it) }
                        .ifEmpty { "not declared by this package" },
                )
                DetailRow(
                    "graph precision",
                    // Deliberately named separately from the variant id: `cpu-int4` ships an fp32
                    // graph, and the measured figure is the only honest one.
                    c.graphPrecision ?: "not measured by this export",
                )
                DetailRow("retrieval", if (c.supportsRag) "yes" else "no embedding stage installed")
                DetailRow(
                    "tool calling",
                    if (c.supportsToolCalling) {
                        "yes — ${c.toolCalling.dialect.name.lowercase()} grammar"
                    } else {
                        "this model has no tool-call grammar of its own; calls are still parsed as " +
                            "JSON, which is what fine-tuning here teaches"
                    },
                )
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
internal fun downloadPhaseLabel(phase: String, waitingForConstraints: Boolean = false): String = when {
    // Checked FIRST: an enqueued job still reports whatever phase it last reached, so matching on
    // the phase alone renders an indefinite wait as an active download.
    waitingForConstraints -> "Waiting for Wi-Fi"
    phase == "Resolving" -> "Resolving"
    phase == "Verifying" -> "Verifying"
    phase == "Installing" -> "Installing"
    else -> "Downloading"
}

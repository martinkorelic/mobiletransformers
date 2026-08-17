package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.app.viewmodels.StartDelay
import com.martinkorelic.mobiletransformers.app.viewmodels.TrainViewModel

/** The training lifecycle: status, charts, events, cancel, resume, merge, scheduling. */
@Composable
fun TrainScreen(vm: TrainViewModel) {
    val state by vm.modelState.collectAsState()
    var tab by remember { mutableIntStateOf(0) }

    ModelGate(state, needs = "Training needs a package exported with TRAIN=1.") { model ->
        Column(Modifier.fillMaxSize()) {
            SubTabs(listOf("Run", "Progress", "Schedule"), tab) { tab = it }
            when (tab) {
                0 -> RunTab(vm, model)
                1 -> ProgressTab(vm)
                else -> ScheduleTab(vm, model)
            }
        }
    }
}

@Composable
private fun RunTab(vm: TrainViewModel, model: MobileTransformerModel) {
    val ui by vm.ui.collectAsState()

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        ScreenIntro(
            "Fine-tune on this device. Order: install the sample dataset, Start, then Merge. Only the " +
                "LoRA adapter trains, so this is minutes rather than hours — but it is still minutes. " +
                "Watch the loss curve on the Progress tab.",
        )

        if (!model.capabilities.supportsTraining) {
            EmptyState(
                title = "This package cannot train",
                detail = "No train/ stage is installed. Pull or export one with TRAIN=1 — the buttons " +
                    "below would fail closed.",
            )
        }

        Section("Data") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Model packages ship no training data — the task belongs with the data, and the " +
                        "data is yours. This installs a small tool-call set generated from the same " +
                        "allowlist the Tool calls screen declares, so what the model learns is " +
                        "exactly what the validator there accepts.",
                    style = MaterialTheme.typography.bodySmall,
                )
                ActionRow {
                    Button(
                        onClick = vm::installSampleDataset,
                        enabled = !ui.running && model.capabilities.supportsTraining,
                    ) { Text("Install sample dataset") }
                }
                ui.datasetNote?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }
        }

        Section("Run") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(ui.status, style = MaterialTheme.typography.titleSmall)
                if (ui.canResume) {
                    Text(
                        "A checkpoint exists — starting again resumes from it while resumeFromState " +
                            "is on (Configuration → Training).",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                ActionRow {
                    Button(
                        onClick = vm::start,
                        enabled = !ui.running && model.capabilities.supportsTraining,
                    ) { Text(if (ui.running) "Training…" else "Start") }
                    OutlinedButton(onClick = vm::cancel, enabled = ui.running) { Text("Cancel") }
                    OutlinedButton(onClick = vm::merge, enabled = !ui.running) { Text("Merge") }
                }
                Text(
                    "Cancel is cooperative: the native loop breaks at the next step boundary and a " +
                        "checkpoint is written, so cancelling is resumable rather than lossy. Merge " +
                        "writes the learned adapter into the inference graph — until then, Chat is " +
                        "still generating from the base weights.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        ui.error?.let { Section("Error") { Text(it) } }
    }
}

/**
 * The curve and the log, in that order.
 *
 * The log is the chart's table view: every value the curve draws is also readable as a number, which
 * is what keeps the chart an enhancement rather than the only way to read the run.
 */
@Composable
private fun ProgressTab(vm: TrainViewModel) {
    val ui by vm.ui.collectAsState()

    Column(Modifier.fillMaxSize()) {
        RunStatusCard(ui)

        TrainingCharts(ui.points, Modifier.padding(top = 12.dp))

        Text(
            "Events",
            style = MaterialTheme.typography.titleSmall,
            modifier = Modifier.padding(start = 16.dp, top = 16.dp, bottom = 4.dp),
        )
        if (ui.events.isEmpty()) {
            Text(
                "Nothing yet. Start a run on the Run tab.",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth()) {
            items(ui.events) { e ->
                Text(
                    e,
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
                )
            }
        }
    }
}

/**
 * Where the run is, at the top of the tab that exists to answer that.
 *
 * The Progress tab opened straight onto a loss chart, and the status line — the one that says
 * "preparing", "step 42 of 108", "failed: …" — was a single `bodyMedium` on the *Run* tab, which is
 * the tab you leave to come here. So the screen dedicated to watching a run was the one place that
 * did not say what the run was doing, and an empty chart meant both "not started" and "starting".
 */
@Composable
private fun RunStatusCard(ui: com.martinkorelic.mobiletransformers.app.viewmodels.TrainUiState) {
    val last = ui.points.lastOrNull()
    Card(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (ui.error != null) {
                MaterialTheme.colorScheme.errorContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            },
        ),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                ui.error ?: ui.status,
                style = MaterialTheme.typography.titleMedium,
            )
            if (ui.running) {
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            Row(horizontalArrangement = Arrangement.spacedBy(20.dp)) {
                Stat("step", last?.step?.toString() ?: "—")
                Stat("loss", last?.let { "%.4f".format(it.loss) } ?: "—")
                Stat("lr", last?.let { "%.2e".format(it.learningRate) } ?: "—")
                Stat("ms/step", last?.stepDurationMs?.toString() ?: "—")
            }
        }
    }
}

@Composable
private fun Stat(label: String, value: String) {
    Column {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun ScheduleTab(vm: TrainViewModel, model: MobileTransformerModel) {
    val ui by vm.ui.collectAsState()
    val scheduled by vm.scheduledRuns.collectAsState()

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        ScreenIntro(
            "Hand the run to the system instead of running it now. Chunks execute only while the " +
                "device is charging and idle, and each chunk re-checks that before starting — so " +
                "unplugging pauses the run rather than failing it.",
        )

        Section("Schedule a run") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Each chunk re-enters the queue when it finishes, restoring globalStep, epoch and " +
                        "the LR schedule from training_state.json — the same mechanism that survives " +
                        "the app's process being killed.",
                    style = MaterialTheme.typography.bodySmall,
                )
                ChipPicker(
                    label = "Start",
                    options = StartDelay.entries,
                    selected = ui.startDelay,
                    optionLabel = { it.label },
                    onSelect = vm::onStartDelayChanged,
                )
                Text(
                    "A delay is a floor, not an appointment: Android batches deferrable work and Doze " +
                        "can hold it longer. Charging is still the real gate — the delay only moves " +
                        "the earliest moment it is checked.",
                    style = MaterialTheme.typography.bodySmall,
                )
                ActionRow {
                    Button(
                        onClick = vm::schedule,
                        enabled = model.capabilities.supportsScheduledTraining,
                    ) { Text("Schedule") }
                    OutlinedButton(
                        onClick = vm::cancelSchedule,
                        enabled = scheduled.isNotEmpty(),
                    ) { Text("Cancel scheduled") }
                }
                ui.scheduled?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }
        }

        Section("Queue") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (scheduled.isEmpty()) {
                    Text(
                        "Nothing queued. A scheduled run appears here with its state, so " +
                            "\"waiting for the charger\" is distinguishable from \"not scheduled\".",
                        style = MaterialTheme.typography.bodySmall,
                    )
                } else {
                    scheduled.forEach { run ->
                        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Text(run.stateLabel, style = MaterialTheme.typography.bodyMedium)
                            Text(run.detail, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }

        ui.error?.let { Section("Error") { Text(it) } }
    }
}

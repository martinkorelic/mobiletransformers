package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.TrainViewModel

/** #18/#19/#34 — the training lifecycle: status, events, cancel, resume, merge, and scheduling. */
@Composable
fun TrainScreen(vm: TrainViewModel) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    ModelGate(state, needs = "Training needs a package exported with TRAIN=1.") { model ->
        Column(Modifier.fillMaxSize()) {
            if (!model.capabilities.supportsTraining) {
                EmptyState(
                    title = "This package cannot train",
                    detail = "No train/ stage is installed. Pull or export one with TRAIN=1 — the " +
                        "buttons below would fail closed.",
                )
            }

            Section("Run") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("status: ${ui.status}", style = MaterialTheme.typography.bodyMedium)
                    if (ui.canResume) {
                        Text(
                            "A checkpoint exists — starting again resumes from it when " +
                                "resumeFromState is on.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = vm::start,
                            enabled = !ui.running && model.capabilities.supportsTraining,
                        ) { Text(if (ui.running) "Training…" else "Start") }
                        OutlinedButton(onClick = vm::cancel, enabled = ui.running) { Text("Cancel") }
                        OutlinedButton(onClick = vm::merge, enabled = !ui.running) { Text("Merge") }
                    }
                    Text(
                        "Cancel is cooperative: the native loop breaks at the next step boundary and " +
                            "a checkpoint is written, so cancelling is resumable rather than lossy.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Section("Schedule (#34)") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "Runs in charging + idle chunks via WorkManager. Each chunk re-enters the " +
                            "queue, so unplugging between chunks pauses the run instead of failing it.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Button(
                        onClick = vm::schedule,
                        enabled = model.capabilities.supportsScheduledTraining,
                    ) { Text("Schedule") }
                    ui.scheduled?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
            }

            ui.error?.let { Section("Error") { Text(it) } }

            Text(
                "Events",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.padding(start = 16.dp, top = 8.dp),
            )
            LazyColumn(Modifier.weight(1f).fillMaxWidth()) {
                items(ui.events) { e ->
                    Text(
                        e,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
                    )
                }
            }
        }
    }
}

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
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.ChatViewModel
import com.martinkorelic.mobiletransformers.app.viewmodels.EnginePickerState

/** #11/#24/#27 — generate + streaming, an honest engine picker, and RAG with source cards. */
@Composable
fun ChatScreen(vm: ChatViewModel) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    ModelGate(state, needs = "Chat needs an inference-capable package.") { model ->
        val picker = EnginePickerState(
            selected = model.capabilities.engine,
            available = model.capabilities.availableEngines,
        )

        Column(Modifier.fillMaxSize()) {
            ScreenIntro(
                "Generate text, optionally grounded in documents you ingest. Tokens stream as they " +
                    "arrive. A small model on a phone is slow and not very fluent — that is the " +
                    "hardware and the model size, not a defect.",
            )
            Section("Engine") {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        picker.available.forEach { engine ->
                            FilterChip(
                                selected = engine == picker.selected,
                                onClick = { /* selection applies at load; see the note below */ },
                                label = { Text(engine.name) },
                            )
                        }
                    }
                    picker.genAiNote?.let {
                        Text(it, style = MaterialTheme.typography.bodySmall)
                    }
                    Text(
                        "The engine is fixed when the model is loaded, so switching means reloading " +
                            "from the Models tab. Naming GenAI and being given Native would be a wrong " +
                            "answer, so the SDK refuses instead of falling back silently.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Section("Retrieval") {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    LabeledSwitch("Ground answers with RAG", ui.useRag, vm::onRagToggled)
                    if (ui.useRag && !model.capabilities.supportsRag) {
                        Text(
                            "This package has no embedding stage, so retrieval will fail closed. " +
                                "Pull a package exported with RAG=1.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    // Retrieval reads a store that only `ingest` fills. Without something here the
                    // toggle above is decorative: every grounded query returns zero sources.
                    Text(
                        "Retrieval searches documents you have ingested — the store starts empty, so " +
                            "grounding an answer before ingesting anything returns no sources.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Button(
                        onClick = vm::ingestSampleDocument,
                        enabled = !ui.ingesting && model.capabilities.supportsRag,
                    ) { Text(if (ui.ingesting) "Ingesting…" else "Ingest sample document") }
                    ui.ingestNote?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
            }

            LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                items(ui.messages) { m ->
                    Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 3.dp)) {
                        Column(Modifier.padding(10.dp)) {
                            Text(
                                if (m.fromUser) "you" else "model",
                                style = MaterialTheme.typography.labelSmall,
                            )
                            Text(m.text, style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
                if (ui.streaming.isNotEmpty()) {
                    item {
                        Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 3.dp)) {
                            Column(Modifier.padding(10.dp)) {
                                Text("model (streaming)", style = MaterialTheme.typography.labelSmall)
                                Text(ui.streaming, style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                }
                if (ui.sources.isNotEmpty()) {
                    item {
                        Section("Sources") {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                ui.sources.forEach { s ->
                                    Text(
                                        "%.3f — %s".format(s.score, s.text.take(240)),
                                        style = MaterialTheme.typography.bodySmall,
                                    )
                                }
                            }
                        }
                    }
                }
                ui.assembledPrompt?.let { p ->
                    item {
                        Section("Assembled prompt (what the model was actually asked)") {
                            Text(p, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
                ui.stats?.let { item { Section("Last generation") { Text(it) } } }
                ui.error?.let { item { Section("Error") { Text(it) } } }
            }

            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                TextField("Prompt", ui.prompt, vm::onPromptChanged)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = vm::send, enabled = !ui.generating) {
                        Text(if (ui.generating) "Generating…" else "Send")
                    }
                    OutlinedButton(onClick = vm::clear) { Text("Clear") }
                }
            }
        }
    }
}

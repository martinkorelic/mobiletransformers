package com.martinkorelic.mobiletransformers.app.views

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.ChatMessage
import com.martinkorelic.mobiletransformers.app.viewmodels.ChatViewModel
import com.martinkorelic.mobiletransformers.app.viewmodels.EnginePickerState
import com.martinkorelic.mobiletransformers.app.viewmodels.ToolCallCard

/**
 * #11/#24/#27/#37 — generate and stream, ground in ingested documents, and call tools, all in one
 * conversation.
 *
 * The retrieved sources and the assembled prompt hang off **the message they produced**, not off the
 * screen. Previously they lived in a screen-level "Sources" section detached from any answer and
 * cleared by the next question, so a conversation with several grounded answers showed one set of
 * sources belonging to none of them in particular.
 */
@Composable
fun ChatScreen(vm: ChatViewModel) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()
    var settingsOpen by remember { mutableStateOf(false) }

    ModelGate(state, needs = "Chat needs an inference-capable package.") { model ->
        val listState = rememberLazyListState()
        // Follow the conversation: without this a streaming answer scrolls out from under the reader.
        LaunchedEffect(ui.messages.size, ui.streaming) {
            val last = ui.messages.lastIndex
            if (last >= 0) listState.animateScrollToItem(last)
        }

        Column(Modifier.fillMaxSize()) {
            ChatToolbar(
                ui.useRag,
                ui.useTools,
                settingsOpen,
                supportsRag = model.capabilities.supportsRag,
                onRag = vm::onRagToggled,
                onTools = vm::onToolsToggled,
                onToggleSettings = { settingsOpen = !settingsOpen },
            )

            if (settingsOpen) {
                ChatSettings(vm, model)
            }

            LazyColumn(
                Modifier.weight(1f),
                state = listState,
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                if (ui.messages.isEmpty()) {
                    item {
                        EmptyState(
                            title = "Say something",
                            detail = "A small model on a phone is slow and not very fluent — that is " +
                                "the hardware and the model size, not a defect. Tokens appear as " +
                                "they are generated.",
                        )
                    }
                }
                items(ui.messages) { m -> MessageCard(m, onSimulate = vm::simulateToolResult) }

                if (ui.streaming.isNotEmpty()) {
                    item { Bubble("model", ui.streaming, streaming = true) }
                }
                ui.phase?.let {
                    // The grounded and tool-call paths do not stream, so without this a 20-second
                    // answer is a frozen screen.
                    item {
                        Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
                            Text(it, style = MaterialTheme.typography.bodySmall)
                            LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 4.dp))
                        }
                    }
                }
                ui.error?.let { item { Section("Error") { Text(it) } } }
            }

            HorizontalDivider()
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                TextField("Message", ui.prompt, vm::onPromptChanged)
                ActionRow {
                    Button(onClick = vm::send, enabled = !ui.generating) {
                        Text(if (ui.generating) "Generating…" else "Send")
                    }
                    OutlinedButton(onClick = vm::clear) { Text("Clear") }
                }
            }
        }
    }
}

@Composable
private fun ChatToolbar(
    useRag: Boolean,
    useTools: Boolean,
    settingsOpen: Boolean,
    supportsRag: Boolean,
    onRag: (Boolean) -> Unit,
    onTools: (Boolean) -> Unit,
    onToggleSettings: () -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        ModeChip("Ground with RAG", useRag, enabled = supportsRag) { onRag(!useRag) }
        ModeChip("Tool calls", useTools) { onTools(!useTools) }
        TextButton(onClick = onToggleSettings) { Text(if (settingsOpen) "Hide setup" else "Setup") }
    }
}

@Composable
private fun ModeChip(label: String, selected: Boolean, enabled: Boolean = true, onClick: () -> Unit) {
    androidx.compose.material3.FilterChip(
        selected = selected,
        onClick = onClick,
        enabled = enabled,
        label = { Text(label) },
    )
}

/** The things a conversation needs set up once — collapsed, because they are scaffolding. */
@Composable
private fun ChatSettings(vm: ChatViewModel, model: com.martinkorelic.mobiletransformers.MobileTransformerModel) {
    val ui by vm.ui.collectAsState()
    val picker = EnginePickerState(
        selected = model.capabilities.engine,
        available = model.capabilities.availableEngines,
    )
    // The document picker: retrieval over a file you chose is a different demo from retrieval over
    // ours, and `ingest` cannot open a content:// URI, so the ViewModel copies it in first.
    val pickDocument = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(vm::ingest) }

    Section("Engine") {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                "Running on ${picker.selected.name}. Available here: ${picker.available.joinToString()}.",
                style = MaterialTheme.typography.bodyMedium,
            )
            picker.genAiNote?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            Text(
                "The engine is fixed when the model is loaded, so switching means reloading from " +
                    "Models. Naming GenAI and being given Native would be a wrong answer, so the SDK " +
                    "refuses instead of falling back silently.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    Section("Documents to retrieve from") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (!model.capabilities.supportsRag) {
                Text(
                    "This package has no embedding stage, so retrieval fails closed. Re-pull it with " +
                        "the RAG feature requested — it is a separate download group.",
                    style = MaterialTheme.typography.bodySmall,
                )
            } else {
                Text(
                    "Retrieval searches only what you have ingested; the store starts empty, so " +
                        "grounding an answer before ingesting anything returns no sources.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            ActionRow {
                Button(
                    onClick = { vm.ingest() },
                    enabled = !ui.ingesting && model.capabilities.supportsRag,
                ) { Text(if (ui.ingesting) "Ingesting…" else "Ingest sample") }
                OutlinedButton(
                    onClick = { pickDocument.launch(arrayOf("text/*", "text/markdown", "application/json")) },
                    enabled = !ui.ingesting && model.capabilities.supportsRag,
                ) { Text("Pick a file…") }
            }
            ui.ingestNote?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            if (ui.ingestedDocuments.isNotEmpty()) {
                Text(
                    "In the store: ${ui.ingestedDocuments.joinToString(", ")}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }

    Section("Tool calls") {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                "With Tool calls on, each message asks for a call from this app's allowlist instead " +
                    "of prose. Nothing is executed — the card shows the intent that WOULD fire.",
                style = MaterialTheme.typography.bodySmall,
            )
            vm.allowlist.forEach { Text("• ${it.actionName}", style = MaterialTheme.typography.bodySmall) }
        }
    }
}

@Composable
private fun MessageCard(m: ChatMessage, onSimulate: (ToolCallCard) -> Unit) {
    when {
        m.toolCall != null -> ToolCallBubble(m.toolCall, onSimulate)
        else -> Bubble(
            who = if (m.fromUser) "you" else "model",
            text = m.text,
            sources = m.sources,
            assembledPrompt = m.assembledPrompt,
            stats = m.stats,
        )
    }
}

@Composable
private fun Bubble(
    who: String,
    text: String,
    streaming: Boolean = false,
    sources: List<com.martinkorelic.mobiletransformers.app.viewmodels.SourceCard> = emptyList(),
    assembledPrompt: String? = null,
    stats: String? = null,
) {
    var showSources by remember { mutableStateOf(false) }
    var showPrompt by remember { mutableStateOf(false) }

    Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 3.dp)) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                if (streaming) "$who (streaming)" else who,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(text, style = MaterialTheme.typography.bodyMedium)

            stats?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (sources.isNotEmpty()) {
                TextButton(onClick = { showSources = !showSources }) {
                    Text(if (showSources) "Hide sources" else "${sources.size} sources")
                }
                if (showSources) {
                    sources.forEach { s ->
                        Column(Modifier.padding(bottom = 6.dp)) {
                            Text(
                                "score %.3f".format(s.score),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Text(s.text, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }

            assembledPrompt?.let { p ->
                TextButton(onClick = { showPrompt = !showPrompt }) {
                    Text(if (showPrompt) "Hide the prompt" else "What the model was actually asked")
                }
                // The whole point of a grounded API returning its prompt: an app that cannot show
                // what was asked cannot debug a bad grounded answer.
                if (showPrompt) Text(p, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

/**
 * A tool call as a turn.
 *
 * Accepted and refused render as peers, because a refusal is the expected answer for untrusted output
 * — it is the safety property working, not a failure to display.
 */
@Composable
private fun ToolCallBubble(card: ToolCallCard, onSimulate: (ToolCallCard) -> Unit) {
    var showRaw by remember { mutableStateOf(false) }

    Card(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 3.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (card.accepted) {
                MaterialTheme.colorScheme.secondaryContainer
            } else {
                MaterialTheme.colorScheme.errorContainer
            },
        ),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                if (card.accepted) "tool call · accepted" else "tool call · refused",
                style = MaterialTheme.typography.labelSmall,
            )

            if (card.accepted) {
                Text(card.actionName.orEmpty(), style = MaterialTheme.typography.titleSmall)
                card.parameters.forEach { (k, v) ->
                    Text("$k = $v", style = MaterialTheme.typography.bodySmall)
                }
                Text("intent: ${card.intentAction}", style = MaterialTheme.typography.bodySmall)
                Text(
                    "willExecute = ${card.willExecute} — dry run. IntentBinder holds no Context and " +
                        "has no startActivity call site, so firing it is the caller's deliberate act.",
                    style = MaterialTheme.typography.bodySmall,
                )
                ActionRow {
                    OutlinedButton(onClick = { onSimulate(card) }) { Text("Simulate a result") }
                }
            } else {
                Text(card.reason.orEmpty(), style = MaterialTheme.typography.bodyMedium)
                Text(
                    "A refusal is a result, not a crash. Expect it from a model that has not been " +
                        "fine-tuned on this action set — train on the Training screen, then try again.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            TextButton(onClick = { showRaw = !showRaw }) {
                Text(if (showRaw) "Hide raw output" else "What the model emitted")
            }
            if (showRaw) Text(card.raw.take(1000), style = MaterialTheme.typography.bodySmall)
        }
    }
}

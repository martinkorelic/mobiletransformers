package com.martinkorelic.mobiletransformers.app.views

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Bolt
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
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
import com.martinkorelic.mobiletransformers.app.viewmodels.ToolExecution

/**
 * Generate and stream, ground in ingested documents, and call tools, all in one
 * conversation.
 *
 * The retrieved sources and the assembled prompt hang off **the message they produced**, not off the
 * screen. Previously they lived in a screen-level "Sources" section detached from any answer and
 * cleared by the next question, so a conversation with several grounded answers showed one set of
 * sources belonging to none of them in particular.
 */
@Composable
fun ChatScreen(vm: ChatViewModel, onOpenSettings: () -> Unit = {}) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    // The system permission dialog, launched from the Activity — a ViewModel holds an application
    // context and cannot show one. The ViewModel asks by setting `pendingPermissions`; this answers.
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
) { granted -> vm.onPermissionResult(granted.values.all { it }) }

    LaunchedEffect(ui.pendingPermissions) {
        ui.pendingPermissions?.let { permissionLauncher.launch(it.permissions.toTypedArray()) }
        }

    ModelGate(state, needs = "Chat needs an inference-capable package.") { model ->
        val listState = rememberLazyListState()
        // Follow the conversation: without this a streaming answer scrolls out from under the reader.
        LaunchedEffect(ui.messages.size, ui.streaming) {
            val last = ui.messages.lastIndex
            if (last >= 0) listState.animateScrollToItem(last)
        }

        Column(Modifier.fillMaxSize()) {
            ChatToolbar(
                useRag = ui.useRag,
                supportsRag = model.capabilities.supportsRag,
                supportsTools = model.capabilities.supportsToolCalling,
                onRag = vm::onRagToggled,
                onOpenSettings = onOpenSettings,
            )

            LazyColumn(
                Modifier.weight(1f),
                state = listState,
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                if (ui.messages.isEmpty()) {
                    item {
                        EmptyState(
                            title = "Say something",
                            detail = if (model.capabilities.supportsToolCalling) {
                                "This model can call tools. Ask it anything — a reply that turns out " +
                                    "to be a call to one of this app's actions is shown as a call, " +
                                    "with the intent it would fire; anything else is shown as an answer."
                            } else {
                                "Tokens appear as they are generated."
                            },
                        )
                    }
                }
                items(ui.messages) { m ->
                    MessageCard(m, onSimulate = vm::simulateToolResult, onRun = vm::runToolCall)
                }

                if (ui.streaming.isNotEmpty()) {
                    item { Bubble(fromUser = false, text = ui.streaming, streaming = true) }
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

/**
 * One toggle, one badge, one link.
 *
 * The "Tool calls" chip that used to sit here asked the user to declare, before sending, whether
 * their next message was a tool call — which is a property of the *reply*, not of the question.
 * Getting it wrong in either direction produced a wrong-looking result from a correctly working
 * system. Tool calling is now detected from the answer, so what remains here is a statement of
 * capability rather than a control.
 *
 * Grounding stays a toggle, because "answer from the documents I ingested" genuinely is a decision
 * the user makes in advance.
 *
 * ### Why "Setup" became a link to Configuration
 *
 * It used to expand a panel here holding three things, none of which belonged in a conversation: the
 * engine (a read-only fact about the loaded model, now in the model bar), the ingest controls (a
 * property of retrieval, now on the Retrieval screen) and the tool-call allowlist (now in
 * Configuration). Meanwhile the settings a user actually wants mid-chat — temperature, length,
 * sampling — were never there at all, so "Setup" opened a panel that could not do the thing its name
 * promised. It now opens the Generation settings, which is what was wanted.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ChatToolbar(
    useRag: Boolean,
    supportsRag: Boolean,
    supportsTools: Boolean,
    onRag: (Boolean) -> Unit,
    onOpenSettings: () -> Unit,
) {
    FlowRow(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterVertically),
    ) {
        ModeChip("Ground with RAG", useRag, enabled = supportsRag) { onRag(!useRag) }
        if (supportsTools) {
            AssistChip(
                onClick = { },
                enabled = false,
                leadingIcon = { Icon(Icons.Outlined.Bolt, contentDescription = null) },
                label = { Text("can call tools") },
            )
        }
        TextButton(onClick = onOpenSettings) { Text("Settings") }
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

@Composable
private fun MessageCard(
    m: ChatMessage,
    onSimulate: (ToolCallCard) -> Unit,
    onRun: (ToolCallCard) -> Unit,
) {
    when {
        m.toolCall != null -> ToolCallBubble(m.toolCall, m.turnStats, onSimulate, onRun)
        m.retrieval != null -> RetrievalBubble(m.retrieval)
        else -> Bubble(
            fromUser = m.fromUser,
            text = m.text,
            assembledPrompt = m.assembledPrompt,
            stats = m.stats,
            turnStats = m.turnStats,
        )
    }
}

/**
 * What retrieval found, as its own turn above the answer it produced.
 *
 * ### Why it does not look like either speaker
 *
 * It is neither the user's turn nor the model's — it is the app reporting on a step it took, so it
 * takes the full width, a `tertiaryContainer` tint and no "you"/"model" caption. A reader should be
 * able to tell at a glance that this line is machinery rather than conversation.
 *
 * ### Why it is collapsed
 *
 * Retrieved chunks are long — `chunkSize` is 512 characters — and several of them between the
 * question and the answer would push the answer off the screen, which is the opposite of what
 * showing the sources is for. The headline is the claim; the passages are there when you want them.
 */
@Composable
private fun RetrievalBubble(card: com.martinkorelic.mobiletransformers.app.viewmodels.RetrievalCard) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 3.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
            contentColor = MaterialTheme.colorScheme.onTertiaryContainer,
        ),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("retrieval", style = MaterialTheme.typography.labelSmall)
            Text(card.headline, style = MaterialTheme.typography.bodyMedium)

            // The file names, which are what a user recognises — the passage text is the detail
            // behind them, not the headline.
            if (card.documents.isNotEmpty()) {
                Text(
                    card.documents.joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (card.queryTimeMs > 0) {
                Text("search %d ms".format(card.queryTimeMs), style = MaterialTheme.typography.labelSmall)
            }

            if (card.passages.isNotEmpty()) {
                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "Hide passages" else "Show passages")
                }
                if (expanded) {
                    card.passages.forEach { p ->
                        Column(Modifier.padding(bottom = 8.dp)) {
                            Text(
                                // Source first: "which file, how close" is the pair that makes a
                                // passage judgeable. A bare score says nothing about provenance.
                                if (p.title.isBlank()) {
                                    "score %.3f".format(p.score)
                                } else {
                                    "%s · score %.3f".format(p.title, p.score)
                                },
                                style = MaterialTheme.typography.labelSmall,
                            )
                            Text(p.text, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
}

/**
 * One turn.
 *
 * ### Telling the two speakers apart
 *
 * Both sides used to be an identical full-width `Card` distinguished only by the words "you" and
 * "model" in 11sp grey above the text — so scanning back through a conversation meant reading the
 * label on every bubble. Three cues carry it now, none of them load-bearing alone: the user's turn
 * is tinted with `primaryContainer` and inset from the left, the model's uses `surfaceVariant` and
 * is inset from the right, and both keep the caption. Colour is never the only signal, which matters
 * for the same reason the status dot has words next to it.
 */
@Composable
private fun Bubble(
    fromUser: Boolean,
    text: String,
    streaming: Boolean = false,
    assembledPrompt: String? = null,
    stats: String? = null,
    turnStats: com.martinkorelic.mobiletransformers.app.viewmodels.TurnStats? = null,
) {
    var showPrompt by remember { mutableStateOf(false) }

    Row(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 3.dp),
        // The asymmetric inset is the cue that survives a greyscale screenshot.
        horizontalArrangement = if (fromUser) Arrangement.End else Arrangement.Start,
    ) {
        Card(
            Modifier.fillMaxWidth(0.92f),
            colors = CardDefaults.cardColors(
                containerColor = if (fromUser) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceVariant
                },
                contentColor = if (fromUser) {
                    MaterialTheme.colorScheme.onPrimaryContainer
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            ),
        ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                when {
                    fromUser -> "you"
                    streaming -> "model · generating…"
                    else -> "model"
                },
                style = MaterialTheme.typography.labelSmall,
            )
            Text(text, style = MaterialTheme.typography.bodyMedium)

            // The measured cost of the turn: speed, and how full the window now is.
            turnStats?.let {
                Text(
                    it.render(),
                    style = MaterialTheme.typography.labelSmall,
                )
            }

            stats?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.labelSmall,
                )
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
}

/**
 * A tool call as a turn.
 *
 * Accepted and refused render as peers, because a refusal is the expected answer for untrusted output
 * — it is the safety property working, not a failure to display.
 */
@Composable
private fun ToolCallBubble(
    card: ToolCallCard,
    turnStats: com.martinkorelic.mobiletransformers.app.viewmodels.TurnStats?,
    onSimulate: (ToolCallCard) -> Unit,
    onRun: (ToolCallCard) -> Unit,
) {
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
                Text(card.intentAction.orEmpty(), style = MaterialTheme.typography.labelSmall)
                ActionRow {
                    // The card used to explain the dry-run contract in two sentences on every call.
                    // That belongs in the docs, not in the conversation — here it is just a button.
                    if (!card.executed) {
                        Button(onClick = { onRun(card) }) { Text("Run") }
                    } else {
                        Text("ran", style = MaterialTheme.typography.labelSmall)
                    }
                    OutlinedButton(onClick = { onSimulate(card) }) { Text("Simulate a result") }
                }
            } else {
                Text(card.reason.orEmpty(), style = MaterialTheme.typography.bodyMedium)
            }

            turnStats?.let {
                Text(it.render(), style = MaterialTheme.typography.labelSmall)
            }

            TextButton(onClick = { showRaw = !showRaw }) {
                Text(if (showRaw) "Hide raw output" else "What the model emitted")
            }
            if (showRaw) Text(card.raw.take(1000), style = MaterialTheme.typography.bodySmall)
        }
    }
}

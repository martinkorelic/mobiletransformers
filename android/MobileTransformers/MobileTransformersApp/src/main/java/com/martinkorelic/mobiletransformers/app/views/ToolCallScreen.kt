package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.Outcome
import com.martinkorelic.mobiletransformers.app.viewmodels.ToolCallViewModel

/**
 * #37 — instruction → validated call → dry-run intent.
 *
 * Accepted and Rejected are rendered as peers. That is the honest shape of the feature: a refusal is
 * the expected answer for untrusted output, and on a model that has not been fine-tuned on the app's
 * action set it is the *only* answer.
 */
@Composable
fun ToolCallScreen(vm: ToolCallViewModel) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    ModelGate(state, needs = "Tool calls need an inference-capable package.") {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Section("What this app allows") {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    ui.allowedActions.forEach { Text("• $it", style = MaterialTheme.typography.bodyMedium) }
                    Text(
                        "A model selects an action; it cannot name an intent. Intent strings come from " +
                            "the app's own ActionSpec, so the reachable set is fixed by this list — not " +
                            "by anything the model emits.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Section("Instruction") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextField("Say what you want", ui.instruction, vm::onInstructionChanged)
                    Button(onClick = vm::submit, enabled = !ui.running) {
                        Text(if (ui.running) "Generating…" else "Generate tool call")
                    }
                }
            }

            when (val o = ui.outcome) {
                null -> Unit
                is Outcome.Accepted -> Section("Accepted") {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("action: ${o.actionName}", style = MaterialTheme.typography.bodyLarge)
                        o.parameters.forEach { (k, v) -> Text("  $k = $v") }
                        Text("intent: ${o.intentAction}")
                        Text(
                            "willExecute = ${o.willExecute}",
                            style = MaterialTheme.typography.bodyLarge,
                        )
                        Text(
                            "Dry run. Nothing is fired: IntentBinder holds no Context and has no " +
                                "startActivity call site, so executing is the caller's deliberate act.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text("raw: ${o.raw}", style = MaterialTheme.typography.bodySmall)
                    }
                }
                is Outcome.Rejected -> Section("Rejected") {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(o.reason, style = MaterialTheme.typography.bodyLarge)
                        Text("raw: ${o.raw.take(400)}", style = MaterialTheme.typography.bodySmall)
                        Text(
                            "A refusal is a result, not a crash. Expect it from a model that has not " +
                                "been fine-tuned on this action set — and note that as of 2026-08-14 " +
                                "the on-device gate still fails even after fine-tuning, for reasons " +
                                "recorded against #37 (merge numerics, not convergence).",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            ui.error?.let { Section("Error") { Text(it) } }
        }
    }
}

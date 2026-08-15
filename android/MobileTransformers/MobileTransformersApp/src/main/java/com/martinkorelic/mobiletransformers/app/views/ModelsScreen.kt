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
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.viewmodels.ModelsViewModel

/**
 * #21/#13 — pull a package by repo id, see what is installed, load one.
 *
 * The first screen for a reason: the previous sample app assumed an `adb push`ed package, so a real
 * user had no way to reach any other feature.
 */
@Composable
fun ModelsScreen(vm: ModelsViewModel) {
    val ui by vm.ui.collectAsState()
    val model by vm.modelState.collectAsState()

    LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        item {
            Guide(
                "Start here — the tabs are in dependency order",
                listOf(
                    "1. MODELS (this tab). Enter a repo id and tap Pull & load. Nothing on any other " +
                        "tab works until a package is installed. Tick 'training' if you want the " +
                        "Train and Tool calls tabs; tick 'RAG' if you want grounding in Chat — each " +
                        "is a separate download group, and you cannot add one later without " +
                        "re-pulling.",
                    "2. Expect a big download. A real package is 1–4 GB and needs roughly its own " +
                        "size again free while it installs. Progress is per-file, and it resumes if " +
                        "interrupted.",
                    "3. CHAT. Type and Send. Tokens stream as they arrive. To try RAG, tap 'Ingest " +
                        "sample document' first — retrieval searches only what you have ingested, so " +
                        "grounding before that returns no sources.",
                    "4. TRAIN. Tap 'Install sample dataset' first — packages ship no training data, " +
                        "by design. Then Start. On a phone a short run is minutes, not seconds; watch " +
                        "the Events list. Merge writes the learned weights into the inference graph.",
                    "5. TOOL CALLS. On a freshly pulled model this will usually say Rejected, and " +
                        "that is the correct answer, not a bug — the validator refuses anything that " +
                        "is not a call it recognises. Train first (step 4), then come back and the " +
                        "same instruction should be Accepted.",
                    "6. FEDERATED is off unless the build enables it, and CONFIG edits the knobs every " +
                        "other tab uses. Both show their state honestly rather than hiding it.",
                ),
            )
        }
        item {
            Section("Pull from the Hub") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextField("Repo id", ui.repoId, vm::onRepoIdChanged)
                    LabeledSwitch(
                        "Also request the training feature",
                        ui.requestTraining,
                        vm::onTrainingRequestedChanged,
                    )
                    LabeledSwitch(
                        "Also request the RAG feature (~91 MB encoder)",
                        ui.requestRag,
                        vm::onRagRequestedChanged,
                    )
                    Text(
                        "Requesting Training fails closed when the package has no train/ stage — that " +
                            "is deliberate, not a bug. RAG is a separate download group: without it " +
                            "no embedding encoder is fetched and the Chat tab's grounding cannot work, " +
                            "so it is asked here where the cost is visible rather than discovered later.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        if (vm.hasHfToken) {
                            "Hub credentials: HF_TOKEN present — private and gated repos are reachable."
                        } else {
                            "Hub credentials: none. Public repos work as-is; a private one will fail " +
                                "with 401. Rebuild with HF_TOKEN=… to reach it."
                        },
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = { vm.loadSelected() },
                            enabled = model !is ModelState.Loading,
                        ) { Text("Pull & load") }
                        OutlinedButton(onClick = vm::unload) { Text("Unload") }
                        OutlinedButton(onClick = vm::refresh) { Text("Refresh") }
                    }
                }
            }
        }

        ui.download?.let { d ->
            item {
                Section("Downloading") {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("${d.filesDone} / ${d.filesTotal} files", style = MaterialTheme.typography.bodyMedium)
                        Text(d.path, style = MaterialTheme.typography.bodySmall)
                        // fraction is null until the download plan is resolved — an indeterminate bar
                        // is the honest rendering of "total not known yet".
                        if (d.fraction != null) {
                            LinearProgressIndicator(progress = { d.fraction!! }, modifier = Modifier.fillMaxWidth())
                        } else {
                            LinearProgressIndicator(Modifier.fillMaxWidth())
                        }
                    }
                }
            }
        }

        ui.message?.let { item { Section("Note") { Text(it) } } }

        item {
            Section("Current model") {
                when (val m = model) {
                    is ModelState.None -> Text("none loaded")
                    is ModelState.Loading -> Text("loading ${m.repoId}…")
                    is ModelState.Failed -> Text("${m.repoId}: ${m.reason}")
                    is ModelState.Loaded -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        val c = m.model.capabilities
                        Text(m.model.repoId, style = MaterialTheme.typography.bodyLarge)
                        Text("engine: ${c.engine} · available: ${c.availableEngines.joinToString()}")
                        Text("features: ${c.availableFeatures.joinToString().ifEmpty { "none" }}")
                        Text(
                            "training=${c.supportsTraining} · rag=${c.supportsRag} · " +
                                "scheduled=${c.supportsScheduledTraining}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }

        item {
            Text(
                "Installed packages",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.padding(start = 16.dp, top = 12.dp),
            )
        }

        if (ui.isEmpty) {
            item {
                EmptyState(
                    title = "Nothing installed yet",
                    detail = "Pull a package above. On device the cache lives in the app's files dir; " +
                        "a package pushed with `make device-package` also shows up here.",
                )
            }
        } else {
            items(ui.installed) { row ->
                Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(row.sanitizedRepoId, style = MaterialTheme.typography.bodyLarge)
                        Text(row.subtitle, style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(onClick = {
                            // The cache key is the sanitized id; loading by it round-trips through
                            // PackageFormat.sanitizeRepoId to the same directory.
                            vm.loadSelected(row.baseModelId ?: row.sanitizedRepoId)
                        }) { Text("Load") }
                    }
                }
            }
        }
    }
}

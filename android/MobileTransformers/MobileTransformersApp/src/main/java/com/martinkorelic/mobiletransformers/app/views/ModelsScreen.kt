package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.ModelCatalog
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.viewmodels.ModelsViewModel
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine

/**
 * #21/#13 — where a model comes from: pick one from the catalog, or name any exported package.
 *
 * The first destination for a reason: the previous sample app assumed an `adb push`ed package, so a
 * real user had no way to reach any other feature.
 *
 * Three tabs, because "where does a model come from" genuinely has three answers and they were
 * previously stacked into one long scroll with the installed list below the fold.
 */
@Composable
fun ModelsScreen(vm: ModelsViewModel) {
    var tab by remember { mutableIntStateOf(0) }
    val download by vm.download.collectAsState()

    Column(Modifier.fillMaxSize()) {
        SubTabs(listOf("Catalog", "Installed", "Pull by id"), tab) { tab = it }

        // Shown above every tab: a pull started from the Catalog tab must stay visible when the user
        // switches to Installed to watch it appear.
        download?.let { DownloadCard(it, onCancel = vm::cancelDownload) }

        when (tab) {
            0 -> CatalogTab(vm)
            1 -> InstalledTab(vm)
            else -> PullByIdTab(vm)
        }
    }
}

@Composable
private fun DownloadCard(d: com.martinkorelic.mobiletransformers.app.DownloadUi, onCancel: () -> Unit) {
    Section(downloadPhaseLabel(d.phase)) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            // Bytes first: with a two-file plan whose second file is 99% of the package, the file
            // counter sits at "1 / 2" for essentially the whole download.
            Text(d.summary, style = MaterialTheme.typography.bodyMedium)
            Text(
                "${d.filesDone} / ${d.filesTotal} files · ${d.path}",
                style = MaterialTheme.typography.bodySmall,
            )
            val fraction = d.fraction
            if (fraction != null) {
                LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
            } else {
                // Null until the plan is resolved — an indeterminate bar is the honest rendering of
                // "the total is not known yet".
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            ActionRow { OutlinedButton(onClick = onCancel) { Text("Cancel") } }
        }
    }
}

@Composable
private fun CatalogTab(vm: ModelsViewModel) {
    val context = LocalContext.current
    val entries = remember { ModelCatalog.load(context) }
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        item {
            ScreenIntro(
                "Packages exported for on-device use. A first pull is hundreds of megabytes to a few " +
                    "gigabytes and needs roughly its own size again free while it installs — it " +
                    "resumes if interrupted, so leaving the app mid-download is safe.",
            )
        }

        if (entries.isEmpty()) {
            item {
                EmptyState(
                    title = "The catalog is empty",
                    detail = "assets/model_catalog.json is missing or malformed. Use the 'Pull by id' " +
                        "tab to name a package directly.",
                )
            }
        }

        items(entries) { entry ->
            Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(entry.displayName, style = MaterialTheme.typography.titleSmall)
                    Text(entry.repoId, style = MaterialTheme.typography.labelSmall)
                    Text(entry.description, style = MaterialTheme.typography.bodySmall)

                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        AssistChip(onClick = {}, label = { Text(entry.sizeLabel) })
                        AssistChip(onClick = {}, label = { Text(entry.task) })
                        if (entry.supportsTraining) AssistChip(onClick = {}, label = { Text("train") })
                        if (entry.supportsRag) AssistChip(onClick = {}, label = { Text("rag") })
                    }

                    if (entry.recommendedFor.isNotBlank()) {
                        Text(
                            "Good for: ${entry.recommendedFor}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }

                    // Each blocked case names its own cause. "Install failed" would collapse three
                    // different problems — not published, no credentials, network — into one.
                    val blocked: String? = when {
                        !entry.published ->
                            "Not published to the Hub yet — export it with `mobiletransformers " +
                                "export` and push, or pick another entry."
                        entry.requiresToken && !vm.hasHfToken ->
                            "This is a private repo and this build carries no HF_TOKEN, so the pull " +
                                "would fail with 401. Rebuild with HF_TOKEN=… to reach it."
                        else -> null
                    }
                    blocked?.let { Text(it, style = MaterialTheme.typography.bodySmall) }

                    ActionRow {
                        Button(
                            onClick = { vm.installFromCatalog(entry) },
                            enabled = blocked == null && state !is ModelState.Loading,
                        ) { Text("Install & load") }
                    }
                }
            }
        }

        ui.message?.let { item { Section("Note") { Text(it) } } }
    }
}

@Composable
private fun InstalledTab(vm: ModelsViewModel) {
    val ui by vm.ui.collectAsState()
    val model by vm.modelState.collectAsState()

    LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        item {
            ScreenIntro(
                "Packages already on this device. Loading one opens a native session; only one is " +
                    "loaded at a time, so loading a second closes the first.",
            )
        }

        if (ui.isEmpty) {
            item {
                EmptyState(
                    title = "Nothing installed yet",
                    detail = "Install one from the Catalog tab. On device the cache lives in the " +
                        "app's files dir; a package pushed with `make device-package` also shows up " +
                        "here.",
                )
            }
        }

        items(ui.installed) { row ->
            Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(row.repoId, style = MaterialTheme.typography.titleSmall)
                    Text(row.subtitle, style = MaterialTheme.typography.bodySmall)
                    ActionRow {
                        OutlinedButton(
                            onClick = {
                                // The repo id this package was INSTALLED from, recorded at install
                                // time. Loading by `baseModelId` — what this used to do — asks for
                                // the upstream model instead, which sanitizes to a different and
                                // absent cache directory, and reports the package as not installed.
                                vm.loadSelected(row.repoId)
                            },
                            enabled = model !is ModelState.Loading,
                        ) { Text("Load") }
                    }
                }
            }
        }

        item {
            ActionRow {
                OutlinedButton(onClick = vm::refresh) { Text("Refresh") }
                OutlinedButton(onClick = vm::unload, enabled = model is ModelState.Loaded) {
                    Text("Unload current")
                }
            }
        }
    }
}

@Composable
private fun PullByIdTab(vm: ModelsViewModel) {
    val ui by vm.ui.collectAsState()
    val model by vm.modelState.collectAsState()

    LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        item {
            ScreenIntro(
                "Any repo holding an exported MobileTransformers package — one with a " +
                    "mobiletransformers_manifest.json at its root. A plain Hugging Face model id will " +
                    "fail on the first request: the manifest is what plans the download.",
            )
        }

        item {
            Section("Repository") {
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
                        "Requesting Training fails closed when the package has no train/ stage — " +
                            "that is deliberate, not a bug. RAG is a separate download group: " +
                            "without it no embedding encoder is fetched and Chat's grounding cannot " +
                            "work, so it is asked here where the cost is visible rather than " +
                            "discovered later.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        item {
            Section("Engine") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        InferenceEngine.entries.forEach { e ->
                            FilterChip(
                                selected = ui.engine == e,
                                onClick = { vm.onEngineChanged(e) },
                                label = { Text(e.name) },
                            )
                        }
                    }
                    Text(
                        "Fixed at load, so switching means reloading. GenAI additionally needs the " +
                            "package to ship genai_config.json and the native probe to succeed on " +
                            "this device; asking for it otherwise fails closed rather than quietly " +
                            "handing back Native.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        item {
            Section("Hub credentials") {
                Text(
                    if (vm.hasHfToken) {
                        "HF_TOKEN present — private and gated repos are reachable."
                    } else {
                        "None. Public repos work as-is; a private one will fail with 401. Rebuild " +
                            "with HF_TOKEN=… to reach it."
                    },
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        item {
            Column(Modifier.padding(16.dp)) {
                ActionRow {
                    Button(
                        onClick = { vm.loadSelected() },
                        enabled = model !is ModelState.Loading,
                    ) { Text("Pull & load") }
                    OutlinedButton(onClick = vm::unload, enabled = model is ModelState.Loaded) {
                        Text("Unload")
                    }
                }
            }
        }

        ui.message?.let { item { Section("Note") { Text(it) } } }
    }
}

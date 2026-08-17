package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.ModelCatalog
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.viewmodels.ModelsViewModel
import com.martinkorelic.mobiletransformers.app.viewmodels.peftDisplayName
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine

/**
 * Where a model comes from: pick one from the catalog, or name any exported package.
 *
 * The first destination for a reason: the previous sample app assumed an `adb push`ed package, so a
 * real user had no way to reach any other feature.
 *
 * **Two tabs, not three.** "Catalog" and "Installed" are the two questions a user actually has — what
 * can I get, and what do I have. Pulling an arbitrary repo id is a third *answer* to the first
 * question, not a peer of it: it is what you do when the shelf does not carry what you want, which is
 * an integrator's need rather than a first-run one. It now sits behind a disclosure at the bottom of
 * Catalog, where it is reachable without being one of the three things the screen appears to be about.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelsScreen(vm: ModelsViewModel) {
    var tab by remember { mutableIntStateOf(0) }
    val download by vm.download.collectAsState()

    Column(Modifier.fillMaxSize()) {
        SubTabs(listOf("Catalog", "Installed"), tab) { tab = it }

        // Shown above every tab: a pull started from the Catalog tab must stay visible when the user
        // switches to Installed to watch it appear.
        download?.let {
            DownloadCard(
                it,
                onCancel = vm::cancelDownload,
                onUseMobileData = vm::retryWithoutWifiRequirement,
            )
        }

        when (tab) {
            0 -> CatalogTab(vm)
            else -> InstalledTab(vm)
        }
    }
}

@Composable
private fun DownloadCard(
    d: com.martinkorelic.mobiletransformers.app.DownloadUi,
    onCancel: () -> Unit,
    onUseMobileData: () -> Unit,
) {
    Section(downloadPhaseLabel(d.phase, d.waitingForConstraints)) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (d.waitingForConstraints) {
                // An indefinite wait needs a way out of itself. Downloads default to Wi-Fi only, and
                // the switch that governs that lives inside the Advanced disclosure further down —
                // which a user who arrived by tapping Install on a catalog card has never opened. So
                // the escape hatch is offered here, where the wait is actually visible.
                Text(
                    "Nothing is downloading. This pull is set to Wi-Fi only and the phone is not on " +
                        "Wi-Fi, so it is queued rather than failed — it will start on its own once " +
                        "Wi-Fi is back.",
                    style = MaterialTheme.typography.bodySmall,
                )
                ActionRow {
                    Button(onClick = onUseMobileData) { Text("Download on mobile data") }
                    OutlinedButton(onClick = onCancel) { Text("Cancel") }
                }
                return@Section
            }

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

@OptIn(ExperimentalLayoutApi::class)
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
                    detail = "assets/model_catalog.json is missing or malformed. Use 'Advanced: pull " +
                        "any package' at the bottom of this screen to name one directly.",
                )
            }
        }

        items(entries) { entry ->
            Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(entry.displayName, style = MaterialTheme.typography.titleSmall)
                    Text(entry.repoId, style = MaterialTheme.typography.labelSmall)
                    Text(entry.description, style = MaterialTheme.typography.bodySmall)

                    // Four assist chips do not fit one phone-width row once `sizeLabel` is a real
                    // figure ("3.9 GB") and `task` is "text-generation".
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        AssistChip(onClick = {}, label = { Text(entry.sizeLabel) })
                        AssistChip(onClick = {}, label = { Text(entry.task) })
                        if (entry.supportsTraining) AssistChip(onClick = {}, label = { Text("train") })
                        if (entry.supportsRag) AssistChip(onClick = {}, label = { Text("rag") })
                        if (entry.peft.isNotBlank()) {
                            AssistChip(onClick = {}, label = { Text(peftDisplayName(entry.peft)) })
        }
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

        // The former third tab. Below the shelf rather than beside it: naming a repo id is what you do
        // when the catalog does not carry what you want.
        item { PullByIdPanel(vm) }
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
            // Padded: every other control on this screen sits inside a Section (16dp), so a bare
            // ActionRow put these two buttons hard against the left edge of the display.
            ActionRow(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                OutlinedButton(onClick = vm::refresh) { Text("Refresh") }
                OutlinedButton(onClick = vm::unload, enabled = model is ModelState.Loaded) {
                    Text("Unload current")
                }
            }
        }
    }
}

/**
 * Pull any exported package by repo id.
 *
 * Was the third tab. Its four-sentence manifest/download-group paragraph is now one sentence plus a
 * [Details] disclosure — the content is all load-bearing (each sentence exists because the behaviour
 * it describes otherwise reads as a bug), but stacked above the controls it buried them.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PullByIdPanel(vm: ModelsViewModel) {
    val ui by vm.ui.collectAsState()
    val model by vm.modelState.collectAsState()
    var expanded by remember { mutableStateOf(false) }

    Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
        TextButton(onClick = { expanded = !expanded }, contentPadding = PaddingValues(0.dp)) {
            Text(
                if (expanded) "Advanced: pull any package  ▾" else "Advanced: pull any package  ▸",
                style = MaterialTheme.typography.labelLarge,
            )
        }
        if (!expanded) return@Column

        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                "Any repo holding an exported MobileTransformers package.",
                style = MaterialTheme.typography.bodySmall,
            )
            Details(
                "The repo needs a mobiletransformers_manifest.json at its root — a plain Hugging Face " +
                    "model id fails on the first request, because the manifest is what plans the " +
                    "download.",
            )

            TextField("Repo id", ui.repoId, vm::onRepoIdChanged)

            LabeledSwitch(
                "Also request training",
                ui.requestTraining,
                vm::onTrainingRequestedChanged,
            )
            LabeledSwitch(
                "Also request retrieval (~91 MB encoder)",
                ui.requestRag,
                vm::onRagRequestedChanged,
            )
            LabeledSwitch("Download over Wi-Fi only", ui.wifiOnly, vm::onWifiOnlyChanged)

            Details {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        "Requesting training fails closed when the package has no train/ stage. That " +
                            "is deliberate, not a bug — a silent downgrade to inference-only would be " +
                            "discovered much later, on the Train screen.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "Retrieval is a separate download group. Without it no embedding encoder is " +
                            "fetched, and Chat's grounding cannot work — so it is asked here, where " +
                            "the cost is visible, rather than discovered later.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "Pulls run in the background and survive leaving the app. With Wi-Fi only on, " +
                            "one started on mobile data waits rather than failing.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Text("Engine", style = MaterialTheme.typography.titleSmall)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                InferenceEngine.entries.forEach { e ->
                    FilterChip(
                        selected = ui.engine == e,
                        onClick = { vm.onEngineChanged(e) },
                        label = { Text(e.name) },
                    )
                }
            }
            Details(
                "Fixed at load, so switching means reloading. Most packages are Native-only: GenAI " +
                    "additionally needs the variant's manifest to declare it, and Gemma-3 packages " +
                    "(FunctionGemma included) are exported through optimum rather than the GenAI " +
                    "builder, so they declare native alone. Choosing GenAI for one of those fails " +
                    "closed at load, naming the declaration, rather than quietly handing back Native.",
            )

            Text(
                if (vm.hasHfToken) {
                    "HF_TOKEN present — private and gated repos are reachable."
                } else {
                    "No HF_TOKEN in this build. Public repos work as-is; a private one fails with 401."
                },
                style = MaterialTheme.typography.bodySmall,
            )

            ActionRow {
                Button(
                    onClick = { vm.loadSelected() },
                    enabled = model !is ModelState.Loading,
                ) { Text("Pull & load") }
                OutlinedButton(onClick = vm::unload, enabled = model is ModelState.Loaded) {
                    Text("Unload")
                }
            }

            ui.message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

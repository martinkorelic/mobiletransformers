package com.martinkorelic.mobiletransformers.app.views

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.RetrievalViewModel
import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch

/**
 * Retrieval with nothing generated on top of it: a query, and the passages closest to it.
 *
 * Every result carries its similarity score, because the score is what makes the screen falsifiable.
 * "The top result looks relevant" is an impression; a top result at 0.71 against a second at 0.38 is
 * a ranking you can argue with — and a set of four results all within 0.02 of each other says the
 * store has nothing useful in it, which is a different problem from a wrong answer.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun RetrievalScreen(vm: RetrievalViewModel) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    val pickDocument = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(vm::ingest) }

    Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState())) {
        ScreenIntro(
            "Search the documents you have ingested. This is retrieval on its own — nothing is " +
                "generated, so what you see is exactly what a grounded answer would be built from.",
        )

        ModelGate(state, needs = "Retrieval needs a package with an embedding stage.") { model ->
            if (!model.capabilities.supportsRag) {
                EmptyState(
                    title = "This package has no embedding stage",
                    detail = "Retrieval needs one, and it is a separate download group. Re-pull this " +
                        "model from the Models screen with RAG requested.",
                )
                return@ModelGate
            }

            Section("Documents") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        if (ui.ingestedDocuments.isEmpty()) {
                            "The store starts empty, so a search now returns nothing. Add the sample " +
                                "set — four short documents on separate subjects, so the ranking has " +
                                "something to tell apart."
                        } else {
                            "In the store: ${ui.ingestedDocuments.joinToString(", ")}"
                        },
                        style = MaterialTheme.typography.bodySmall,
                    )
                    ActionRow {
                        Button(onClick = vm::ingestSamples, enabled = !ui.ingesting) {
                            Text(if (ui.ingesting) "Ingesting…" else "Add sample documents")
                        }
                        OutlinedButton(
                            onClick = {
                                pickDocument.launch(
                                    arrayOf("text/*", "text/markdown", "application/json"),
                                )
                            },
                            enabled = !ui.ingesting,
                        ) { Text("Pick a file…") }
                    }
                    if (ui.ingesting) LinearProgressIndicator(Modifier.fillMaxWidth())
                    ui.note?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
            }

            Section("Query") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextField("Search for", ui.query, vm::onQueryChanged)
                    ActionRow {
                        Button(
                            onClick = vm::search,
                            enabled = !ui.searching && ui.query.isNotBlank(),
                        ) { Text(if (ui.searching) "Searching…" else "Search") }
                        OutlinedButton(
                            onClick = vm::clear,
                            enabled = ui.matches.isNotEmpty(),
                        ) { Text("Clear") }
                    }
                    // Each of these has its best match in a different bundled document, so tapping
                    // through them shows the ranking discriminating rather than always winning.
                    Text("Try:", style = MaterialTheme.typography.labelSmall)
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        vm.exampleQueries.forEach { example ->
                            AssistChip(
                                onClick = { vm.useExample(example) },
                                label = { Text(example, style = MaterialTheme.typography.labelSmall) },
                            )
                        }
                    }
                }
            }

            if (ui.searching) LinearProgressIndicator(Modifier.fillMaxWidth().padding(16.dp))

            if (ui.matches.isNotEmpty()) {
                Section("${ui.matches.size} closest passages · ${ui.queryTimeMs} ms") {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            "for \"${ui.searchedQuery}\"",
                            style = MaterialTheme.typography.labelSmall,
                        )
                        val best = ui.matches.first().score
                        ui.matches.forEachIndexed { index, match ->
                            MatchCard(index + 1, match, best)
                        }
                    }
                }
            }

            if (ui.foundNothing) {
                EmptyState(
                    title = "Nothing matched",
                    // These two look identical in an empty result list and need opposite fixes.
                    detail = if (ui.searchedWithEmptyStore) {
                        "The store was empty when this ran — retrieval searches only what you have " +
                            "ingested. Add the sample documents above and search again."
                    } else {
                        "There are documents in the store, but nothing in them was close enough to " +
                            "the query. Try one of the suggested searches to see the ranking work."
                    },
                )
            }

            ui.error?.let { Section("Error") { Text(it) } }
        }
    }
}

/**
 * One result: its rank, its score, and a bar showing the score **relative to the best match**.
 *
 * Relative rather than absolute because cosine similarities over a small store cluster in a narrow
 * band — a set of raw 0.3–0.4 bars all look equally short and equally uninformative. Scaling against
 * the top hit makes the gap between first and second visible, which is the thing worth seeing.
 */
@Composable
private fun MatchCard(rank: Int, match: RetrievalMatch, bestScore: Double) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
            contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        ),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("#$rank", style = MaterialTheme.typography.labelSmall)
                Text("score %.3f".format(match.score), style = MaterialTheme.typography.labelSmall)
            }
            val fraction = if (bestScore > 0.0) (match.score / bestScore).coerceIn(0.0, 1.0) else 0.0
            LinearProgressIndicator(
                progress = { fraction.toFloat() },
                modifier = Modifier.fillMaxWidth().height(4.dp),
            )
            Text(match.text, style = MaterialTheme.typography.bodySmall)
        }
    }
}

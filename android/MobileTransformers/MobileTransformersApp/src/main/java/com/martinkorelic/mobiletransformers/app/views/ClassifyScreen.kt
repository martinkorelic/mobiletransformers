package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.ClassifyViewModel
import com.martinkorelic.mobiletransformers.runtime.LabelScore

/**
 * Ask a classifier something and see the probability it assigns each label.
 *
 * The screen is deliberately a distribution rather than a single answer. A classifier that is 34%/33%/
 * 33% across three labels and one that is 99%/0.5%/0.5% both "predict" the same class, and only the
 * bars distinguish them — which is exactly the difference a fine-tuning run is supposed to make, and
 * the reason this screen is the encoder story's payoff rather than a debug view.
 */
@Composable
fun ClassifyScreen(viewModel: ClassifyViewModel) {
    val ui by viewModel.ui.collectAsState()
    val modelState by viewModel.modelState.collectAsState()

    Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState())) {
        ScreenIntro(
            "Runs the package's classification head over your text and shows every label's " +
                "probability. Train the head first and run the same text again — the bars moving is " +
                "the fine-tune working.",
        )

        ModelGate(modelState, needs = "This screen needs a text-classification package.") {
            Section("Text") {
                TextField(label = "Input", value = ui.text, onChange = viewModel::onTextChanged)
                ActionRow {
                    Button(
                        onClick = viewModel::submit,
                        enabled = !ui.running && ui.text.isNotBlank(),
                    ) { Text(if (ui.running) "Classifying…" else "Classify") }
                    OutlinedButton(
                        onClick = viewModel::clear,
                        enabled = !ui.running && ui.scores.isNotEmpty(),
                    ) { Text("Clear") }
                    if (ui.running) {
                        CircularProgressIndicator(Modifier.width(20.dp))
                    }
                }
            }

            ui.error?.let { message ->
                Section("Could not classify") {
                    // Verbatim. The likeliest cause is a package whose head ships no `id2label`, and
                    // `classify()` says so precisely; a friendlier paraphrase loses the fix.
                    Text(message, style = MaterialTheme.typography.bodySmall)
                }
            }

            if (ui.scores.isNotEmpty()) {
                Section("Prediction") {
                    ui.best?.let { best ->
                        Text(best.label, style = MaterialTheme.typography.headlineSmall)
                        Text(
                            "%.1f%% confident · class index ${best.index}".format(best.score * 100),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    if (ui.classifiedText.isNotBlank()) {
                        Text(
                            "for: \"${ui.classifiedText}\"",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }

                Section("All labels") {
                    ui.scores.forEach { LabelBar(it) }
                }
            }

            ui.previous?.let { previous ->
                Section("Previous run") {
                    if (ui.previousText.isNotBlank()) {
                        Text(
                            "for: \"${ui.previousText}\"",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    previous.forEach { LabelBar(it) }
                }
            }
        }
    }
}

/** One label, its probability bar, and the number — the bar alone cannot be read precisely. */
@Composable
private fun LabelBar(score: LabelScore) {
    Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(
                score.label,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
            )
            Text(
                "%.1f%%".format(score.score * 100),
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.End,
            )
        }
        LinearProgressIndicator(
            // coerceIn because a softmax that has drifted (or a head read through a stale
            // embeddingDim) can hand back something outside 0..1, and the indicator would otherwise
            // draw past its own bounds rather than showing that anything is wrong.
            progress = { score.score.toFloat().coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth().padding(top = 2.dp),
        )
    }
}

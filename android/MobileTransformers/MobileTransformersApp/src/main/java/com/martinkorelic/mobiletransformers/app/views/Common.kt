package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.ModelState

/**
 * The empty state every screen needs.
 *
 * On a real device most screens do nothing without an installed package — arm64-v8a only, no emulator
 * — so "no model installed" is the first thing a new user sees and has to be legible rather than a
 * blank screen or a crash. [action] tells them where to go.
 */
@Composable
fun EmptyState(title: String, detail: String, modifier: Modifier = Modifier) {
    Card(modifier = modifier.fillMaxWidth().padding(16.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(detail, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

/** Renders the four [ModelState] cases uniformly, so no screen invents its own vocabulary for them. */
@Composable
fun ModelGate(
    state: ModelState,
    needs: String,
    content: @Composable (com.martinkorelic.mobiletransformers.MobileTransformerModel) -> Unit,
) {
    when (state) {
        is ModelState.None -> EmptyState(
            title = "No model loaded",
            detail = "Open the Models tab and pull a package by repo id. $needs",
        )
        is ModelState.Loading -> Column(Modifier.fillMaxWidth().padding(16.dp)) {
            Text("Loading ${state.repoId}…")
            LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
        }
        is ModelState.Failed -> EmptyState(
            title = "Could not load ${state.repoId}",
            // Verbatim: the SDK's exceptions name the missing feature or artifact, and paraphrasing
            // them is how the diagnosis gets lost.
            detail = state.reason,
        )
        is ModelState.Loaded -> content(state.model)
    }
}

/** A labelled section, used by every screen so the app reads as one thing. */
@Composable
fun Section(title: String, content: @Composable () -> Unit) {
    Card(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        colors = CardDefaults.cardColors(),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            content()
        }
    }
}

@Composable
fun LabeledSwitch(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

/** A numeric field that refuses to write a malformed value rather than silently coercing it to 0. */
@Composable
fun IntField(label: String, value: Int, onChange: (Int) -> Unit) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { text -> text.toIntOrNull()?.let(onChange) },
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
fun FloatField(label: String, value: Float, onChange: (Float) -> Unit) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { text -> text.toFloatOrNull()?.let(onChange) },
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
fun TextField(label: String, value: String, onChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

/**
 * A collapsible "what do I do here" card.
 *
 * The showcase app is the reference example for the SDK, and someone opening it for the first time on
 * a real phone has no way to know that the order matters (nothing works before a package is pulled),
 * that a pull is gigabytes, or that the Tool calls screen is *supposed* to refuse until the model has
 * been fine-tuned. None of that is discoverable from the controls themselves, and a wrong expectation
 * reads as a broken app.
 *
 * Collapsible because it is scaffolding: useful once, noise afterwards.
 */
@Composable
fun Guide(title: String, steps: List<String>, initiallyExpanded: Boolean = true) {
    var expanded by remember { mutableStateOf(initiallyExpanded) }
    Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(title, style = MaterialTheme.typography.titleSmall)
                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "Hide" else "Show")
                }
            }
            if (expanded) {
                steps.forEach { step ->
                    Text(step, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

/**
 * One line saying what a screen does and what to expect from it.
 *
 * Deliberately separate from [Guide]: this always shows, because "what am I looking at" stays useful
 * after "how do I start" stops being.
 */
@Composable
fun ScreenIntro(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.bodySmall,
        modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
    )
}

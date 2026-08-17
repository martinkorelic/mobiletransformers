package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.ui.text.input.KeyboardType
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

/**
 * A label and a switch, with the **label** giving way when space runs out.
 *
 * `SpaceBetween` alone lets the text claim its full intrinsic width and pushes the switch past the
 * right edge — which is how "Also request the RAG feature (~91 MB encoder)" ended up with an
 * unreachable control. `weight(1f)` makes the label the flexible half, so the switch keeps its fixed
 * size and stays on screen and the label wraps instead.
 */
@Composable
fun LabeledSwitch(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

/**
 * A numeric field that refuses to write a malformed value rather than silently coercing it to 0.
 *
 * It now also **shows what you typed**. The previous version rendered `value.toString()` and dropped
 * any keystroke that did not parse, which makes the field feel broken in the most ordinary editing
 * there is: clearing "128" to type "256" produces an empty string, which does not parse, so the field
 * snapped back to "128" and the keyboard appeared dead. The text is local state; the config is
 * written only when the text parses and satisfies [range].
 */
@Composable
fun IntField(label: String, value: Int, range: IntRange? = null, hint: String? = null, onChange: (Int) -> Unit) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    val parsed = text.toIntOrNull()
    val error = parsed == null || (range != null && parsed !in range)

    OutlinedTextField(
        value = text,
        onValueChange = { new ->
            text = new
            new.toIntOrNull()?.let { if (range == null || it in range) onChange(it) }
        },
        label = { Text(label) },
        isError = error,
        supportingText = {
            when {
                parsed == null && text.isNotBlank() -> Text("not a whole number")
                range != null && parsed != null && parsed !in range ->
                    Text("must be between ${range.first} and ${range.last}")
                hint != null -> Text(hint)
            }
        },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
fun FloatField(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>? = null,
    hint: String? = null,
    onChange: (Float) -> Unit,
) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    val parsed = text.toFloatOrNull()
    val error = parsed == null || (range != null && parsed !in range)

    OutlinedTextField(
        value = text,
        onValueChange = { new ->
            text = new
            new.toFloatOrNull()?.let { if (range == null || it in range) onChange(it) }
        },
        label = { Text(label) },
        isError = error,
        supportingText = {
            when {
                parsed == null && text.isNotBlank() -> Text("not a number")
                range != null && parsed != null && parsed !in range ->
                    Text("must be between ${range.start} and ${range.endInclusive}")
                hint != null -> Text(hint)
            }
        },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

/**
 * A picker over a closed set.
 *
 * Several settings the SDK models as enums or as a fixed registry were rendered as free text, which
 * turns a choice into a spelling test whose failure surfaces far from where it was made: mistyping
 * `DatasetConfig.task` is accepted here and reported as `Unsupported task: …` minutes into a training
 * run, on a different screen. Anything with a knowable set of values belongs here instead.
 *
 * @param describe optional second line per option, for sets whose names do not explain themselves.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun <T> Dropdown(
    label: String,
    options: List<T>,
    selected: T?,
    optionLabel: (T) -> String,
    describe: (T) -> String? = { null },
    onSelect: (T) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = Modifier.fillMaxWidth(),
    ) {
        OutlinedTextField(
            value = selected?.let(optionLabel) ?: "",
            onValueChange = { },
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            supportingText = selected?.let(describe)?.let { { Text(it) } },
            modifier = Modifier.menuAnchor().fillMaxWidth(),
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = {
                        Column {
                            Text(optionLabel(option))
                            describe(option)?.let {
                                Text(it, style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    },
                    onClick = {
                        onSelect(option)
                        expanded = false
                    },
                )
            }
        }
    }
}

/** A row of chips over a closed set — the compact form of [Dropdown] for three or four options. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun <T> ChipPicker(label: String, options: List<T>, selected: T?, optionLabel: (T) -> String, onSelect: (T) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium)
        // Wraps for the same reason ActionRow does: four chips with real labels do not fit a phone.
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            options.forEach { option ->
                FilterChip(
                    selected = option == selected,
                    onClick = { onSelect(option) },
                    label = { Text(optionLabel(option)) },
                )
            }
        }
    }
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
 * A collapsed disclosure for the *justification* behind a control.
 *
 * The screens accumulated multi-sentence paragraphs explaining why a setting exists, why a default is
 * what it is, and what happens if you choose wrong. Every one of those was written because someone
 * would otherwise read the behaviour as a bug — so none of it is deletable — but stacked above the
 * controls it buried them, and a user looking for a switch had to read an essay to find it.
 *
 * The rule this encodes: **lead with the control, put the argument behind [Details].** Distinct from
 * [Guide], which is a whole-screen walkthrough shown once, and from [ScreenIntro], which is the one
 * always-visible line saying what you are looking at.
 *
 * Collapsed by default, unlike [Guide] — this is reference material for the moment something looks
 * wrong, not an introduction.
 */
@Composable
fun Details(label: String = "Why?", content: @Composable () -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Column {
        TextButton(onClick = { expanded = !expanded }, contentPadding = PaddingValues(0.dp)) {
            Text(
                if (expanded) "$label  ▾" else "$label  ▸",
                style = MaterialTheme.typography.labelMedium,
            )
        }
        if (expanded) content()
    }
}

/** [Details] over a single paragraph — the common case, so callers do not repeat the `Text` styling. */
@Composable
fun Details(text: String, label: String = "Why?") {
    Details(label) { Text(text, style = MaterialTheme.typography.bodySmall) }
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

/**
 * Tabs *within* one drawer destination.
 *
 * Now that the drawer carries navigation, a tab row means one thing only: alternative views of the
 * same subject. Configuration splits into the config objects it edits; Models splits into where a
 * package comes from. Previously the tab row was the navigation, so it had to carry six unrelated
 * destinations and could express neither grouping nor dependency.
 */
@Composable
fun SubTabs(titles: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    TabRow(selectedTabIndex = selected) {
        titles.forEachIndexed { index, title ->
            Tab(
                selected = index == selected,
                onClick = { onSelect(index) },
                text = { Text(title, style = MaterialTheme.typography.labelLarge) },
            )
        }
    }
}

/**
 * A row of actions with one primary.
 *
 * Buttons were previously laid out ad hoc per screen, so identical action rows had different spacing
 * and no shared baseline — the "unaligned buttons" that read as sloppiness rather than as a bug.
 *
 * **Wraps.** It was a plain `Row`, which lays children out past the right edge rather than onto a
 * second line, so any row of three buttons whose labels were long enough lost the last one entirely —
 * off screen, unreachable, with nothing to indicate it existed. Three-button rows are the norm here
 * (Start / Cancel / Merge, Install / Load / Unload), so this was a matter of label length, not of
 * layout intent.
 *
 * @param modifier for call sites that live outside a [Section] and must supply their own padding —
 *   a bare `fillMaxWidth()` row sits flush against the screen edge.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ActionRow(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    FlowRow(
        modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
    ) { content() }
}

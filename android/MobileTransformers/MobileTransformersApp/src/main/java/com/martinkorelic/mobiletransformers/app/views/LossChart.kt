package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.StepPoint
import kotlin.math.abs

/**
 * The training run as a picture: loss per step, with learning rate beneath it.
 *
 * ### Why the app needed one
 *
 * `TrainingEvent.Step` has always carried `stepLoss`, `epochLoss`, `learningRate` and
 * `stepDurationMs`, and the Train screen rendered the whole event with `toString()` into a scrolling
 * list of data-class dumps. Everything needed to see whether a run was working was arriving and being
 * thrown away — and "is the loss going down" is the only question a fine-tuning demo has to answer.
 *
 * ### Why two charts rather than one with two axes
 *
 * Loss and learning rate differ by four orders of magnitude, so plotting both against one y-axis makes
 * one of them a flat line; plotting them against *two* y-axes is worse, because the alignment between
 * the two scales is arbitrary and the reader sees a relationship the data does not contain. Small
 * multiples over a shared x-axis show both honestly: same steps, separate scales, no implied
 * correlation.
 *
 * One series per chart, so neither needs a legend — the title names it. Colors come from the theme,
 * and no text is drawn in a series colour: the numbers wear text tokens and the line carries identity.
 */
@Composable
fun TrainingCharts(points: List<StepPoint>, modifier: Modifier = Modifier) {
    if (points.size < 2) {
        Text(
            if (points.isEmpty()) {
                "No steps yet — the curve appears once training reports its first step."
            } else {
                "One step so far; a curve needs two."
            },
            style = MaterialTheme.typography.bodySmall,
            modifier = modifier.padding(horizontal = 16.dp, vertical = 8.dp),
        )
        return
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        StatTiles(points)

        LineChart(
            title = "Loss",
            values = points.map { it.loss },
            steps = points.map { it.step },
            color = MaterialTheme.colorScheme.primary,
            height = 160.dp,
            format = { "%.4f".format(it) },
        )

        LineChart(
            title = "Learning rate",
            values = points.map { it.learningRate },
            steps = points.map { it.step },
            color = MaterialTheme.colorScheme.tertiary,
            height = 80.dp,
            format = { "%.2e".format(it) },
        )
    }
}

/**
 * The three numbers worth reading without decoding a curve.
 *
 * "Δ from first" is the one that answers the actual question — a loss that has not moved is the
 * failure mode a short run at the default `gradientAccumulationSteps` produces silently.
 */
@Composable
private fun StatTiles(points: List<StepPoint>) {
    val first = points.first().loss
    val last = points.last().loss
    val delta = last - first

    Row(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Tile("loss now", "%.4f".format(last), Modifier.weight(1f))
        Tile("best", "%.4f".format(points.minOf { it.loss }), Modifier.weight(1f))
        Tile(
            "Δ from first",
            (if (delta <= 0) "−" else "+") + "%.4f".format(abs(delta)),
            Modifier.weight(1f),
        )
    }
}

@Composable
private fun Tile(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(value, style = MaterialTheme.typography.titleMedium)
    }
}

/**
 * One series over global step.
 *
 * Hairline solid axes (never dashed — dashing reads as a threshold), a 2px line, and a single direct
 * label at the last point. A number on every point is unreadable, and the values are all present in
 * the Events list below, which is this chart's table-view twin.
 */
@Composable
private fun LineChart(
    title: String,
    values: List<Float>,
    steps: List<Int>,
    color: Color,
    height: androidx.compose.ui.unit.Dp,
    format: (Float) -> String,
) {
    val axis = MaterialTheme.colorScheme.outlineVariant
    val min = values.min()
    val max = values.max()
    // A flat series has zero range; without a floor every point maps to the same y and the line
    // collapses onto an edge.
    val span = (max - min).takeIf { it > 1e-9f } ?: 1f

    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(title, style = MaterialTheme.typography.labelMedium)
            // The direct label: current value, in text ink rather than the series colour.
            Text(
                format(values.last()),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Canvas(Modifier.fillMaxWidth().height(height).padding(top = 6.dp, bottom = 6.dp)) {
            val w = size.width
            val h = size.height

            // Recessive chrome: two hairlines, one shade off the surface, and no grid at all — at
            // phone width a gridded 160dp plot is mostly grid.
            drawLine(axis, Offset(0f, h), Offset(w, h), strokeWidth = 1f)
            drawLine(axis, Offset(0f, 0f), Offset(0f, h), strokeWidth = 1f)

            val path = Path()
            values.forEachIndexed { i, v ->
                val x = if (values.size == 1) 0f else w * i / (values.size - 1).toFloat()
                // Inverted: canvas y grows downward, and a falling loss must read as falling.
                val y = h - ((v - min) / span) * h
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round))

            // The endpoint, with a surface ring so it stays legible where the line doubles back.
            val lastX = w
            val lastY = h - ((values.last() - min) / span) * h
            drawCircle(color, radius = 4.dp.toPx(), center = Offset(lastX, lastY))
        }

        Text(
            "step ${steps.first()} → ${steps.last()}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

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
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
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
 * One series over global step, with a labelled x-axis.
 *
 * ### Three things this got wrong
 *
 * - **`strokeWidth = 1f` is one *pixel*, not one dp.** At this device's 3x density that is a third of
 *   a dp, which lands between physical pixels and renders as a barely-visible grey smear — the axes
 *   looked absent.
 * - **The x-axis was drawn at exactly `y = h`**, the last row of the canvas, so half the stroke fell
 *   outside the drawing bounds and was clipped. What survived was half of an already-invisible line:
 *   "cut off at the bottom" was literally true.
 * - **There were no ticks.** The only x information was a `step 0 → 108` caption under the plot, so a
 *   reader could see the range but could not place any point within it.
 *
 * Now the canvas reserves gutters and the series is drawn into a plot rect inset from them, which is
 * what leaves room for tick labels without clipping either axis. Ticks are drawn with the measured
 * text so they sit exactly under their gridline rather than being approximated by a `Row`.
 *
 * Still deliberately sparse: no grid, hairline axes one shade off the surface, one direct label at
 * the last point. At phone width a gridded 160dp plot is mostly grid, and every value is also in the
 * Events list below, which is this chart's table-view twin.
 */
@OptIn(ExperimentalTextApi::class)
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
    val tickInk = MaterialTheme.colorScheme.onSurfaceVariant
    val tickStyle = MaterialTheme.typography.labelSmall
    val measurer = rememberTextMeasurer()

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

        // The gutter is part of the canvas, not padding around it: the axis has to be drawn INSIDE
        // the drawing bounds or it gets clipped, and the tick labels need somewhere to live.
        Canvas(Modifier.fillMaxWidth().height(height + X_AXIS_GUTTER).padding(top = 6.dp)) {
            val gutter = X_AXIS_GUTTER.toPx()
            val stroke = 1.dp.toPx()
            // Inset by half a stroke so neither axis is half-outside the canvas.
            val left = stroke / 2f
            val right = size.width - stroke / 2f
            val top = stroke / 2f
            val bottom = size.height - gutter

            drawLine(axis, Offset(left, bottom), Offset(right, bottom), strokeWidth = stroke)
            drawLine(axis, Offset(left, top), Offset(left, bottom), strokeWidth = stroke)

            val plotWidth = right - left
            val plotHeight = bottom - top
            fun xAt(index: Int): Float =
                if (values.size == 1) left else left + plotWidth * index / (values.size - 1).toFloat()
            fun yAt(value: Float): Float =
                // Inverted: canvas y grows downward, and a falling loss must read as falling.
                bottom - ((value - min) / span) * plotHeight

            // Ticks at the ends and at even fractions between, capped so labels cannot collide on a
            // narrow screen. Indices, not step numbers, so a run with irregular step reporting still
            // puts every tick on a real data point.
            val tickCount = minOf(MAX_X_TICKS, values.size)
            val tickIndices = if (tickCount <= 1) {
                listOf(0)
            } else {
                (0 until tickCount).map { it * (values.size - 1) / (tickCount - 1) }
            }
            for (index in tickIndices.distinct()) {
                val x = xAt(index)
                drawLine(
                    axis,
                    Offset(x, bottom),
                    Offset(x, bottom + TICK_LENGTH.toPx()),
                    strokeWidth = stroke,
                )
                val label = measurer.measure(steps[index].toString(), tickStyle)
                // Centred on the tick, then nudged inward at the edges so the first and last labels
                // stay inside the canvas instead of being clipped by it.
                val half = label.size.width / 2f
                val labelX = (x - half).coerceIn(0f, size.width - label.size.width)
                drawText(
                    textLayoutResult = label,
                    color = tickInk,
                    topLeft = Offset(labelX, bottom + TICK_LENGTH.toPx() + 2.dp.toPx()),
                )
            }

            val path = Path()
            values.forEachIndexed { i, v ->
                val x = xAt(i)
                val y = yAt(v)
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round))

            // The endpoint, so the current value is locatable on the curve.
            drawCircle(color, radius = 4.dp.toPx(), center = Offset(xAt(values.size - 1), yAt(values.last())))
        }

        Text(
            "step",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center,
        )
    }
}

/** Room under the plot for the tick marks and their labels. */
private val X_AXIS_GUTTER = 18.dp

private val TICK_LENGTH = 3.dp

/** Enough to place a point, few enough that the labels never collide at phone width. */
private const val MAX_X_TICKS = 5

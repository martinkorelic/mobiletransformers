package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.views.loadedStatusLine
import com.martinkorelic.mobiletransformers.app.views.statusLine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What the model bar's indicator says.
 *
 * ### The defect
 *
 * The dot was painted from [ModelState] alone: `Loaded -> colorScheme.primary`. In this theme
 * `primary` is the project red, so a healthy idle model showed the colour every user reads as
 * "stop" — and a model that was genuinely mid-generation showed exactly the same thing, because
 * `ModelState` cannot distinguish the two. The one question a status light exists to answer, "can I
 * ask it something right now", had no representation anywhere in the app.
 *
 * The colour itself needs a composition to resolve, so what is pinned here is the state machine
 * behind it — [ModelActivity.isBusy] and the words that must accompany the colour, since a red/green
 * dot is not a status for anyone who cannot tell red from green.
 */
class ModelStatusTest {

    @Test
    fun idleIsTheOnlyNonBusyActivity() {
        assertFalse(ModelActivity.Idle.isBusy)
        for (activity in ModelActivity.entries - ModelActivity.Idle) {
            assertTrue("$activity must count as busy", activity.isBusy)
        }
    }

    @Test
    fun everyKindOfWorkIsRepresented() {
        // Generating, training and merging all occupy the same native session, and all three used to
        // be invisible. If a fourth kind of work is added it belongs here, not in a new flag.
        val names = ModelActivity.entries.map { it.name }.toSet()
        assertTrue(names.containsAll(setOf("Loading", "Generating", "Training", "Merging", "Ingesting")))
    }

    @Test
    fun theWordsMatchTheState() {
        assertEquals("nothing loaded", statusLine(ModelState.None, ModelActivity.Idle))
        assertEquals("loading", statusLine(ModelState.Loading("org/model"), ModelActivity.Loading))
        assertEquals("failed to load", statusLine(ModelState.Failed("org/model", "no such file"), ModelActivity.Idle))
    }

    @Test
    fun aLoadedIdleModelReportsReady() {
        // The case the old dot got wrong: free, and painted with the theme's red.
        assertEquals("ready", loadedStatusLine(ModelActivity.Idle))
    }

    @Test
    fun aBusyLoadedModelNamesWhatItIsDoing() {
        // "busy" alone would be a smaller lie than the old dot but still a lie: the user's next
        // question is always "busy with what", and training and generating have very different waits.
        for (activity in ModelActivity.entries - ModelActivity.Idle) {
            val line = loadedStatusLine(activity)
            assertTrue("'$line' should start with busy", line.startsWith("busy · "))
            assertTrue("'$line' should name the activity", line.endsWith(activity.label))
        }
    }
}

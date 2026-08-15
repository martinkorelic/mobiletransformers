package com.martinkorelic.mobiletransformers.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What the drawer offers, and what it says about what it will not do yet.
 *
 * The states that matter here are the ones hardest to reach by hand on a device — no package
 * installed, a package with no train stage, a failed load — and therefore the ones most likely to rot
 * unnoticed. They are also the first thing a new user meets, so an unhelpful answer here is the whole
 * first impression.
 */
class NavigationStateTest {

    @Test
    fun withNoModelOnlyTheEntryPointsAreEnabled() {
        val state = ModelState.None

        // Models is where a model comes from and About explains the app; both have to work before
        // anything is loaded, or a fresh install is a dead end.
        assertEquals(Availability.Enabled, Destination.Models.availability(state))
        assertEquals(Availability.Enabled, Destination.About.availability(state))

        for (d in listOf(Destination.Chat, Destination.ToolCalls, Destination.Train, Destination.Federated)) {
            val availability = d.availability(state)
            assertTrue("$d should be blocked with no model", availability is Availability.Blocked)
            // The reason is the instruction — it has to say what to do, not merely that something is wrong.
            assertTrue(
                "$d's reason does not say where to go",
                (availability as Availability.Blocked).reason.contains("Models"),
            )
        }
    }

    @Test
    fun aFailedLoadNamesTheModelThatFailed() {
        val availability = Destination.Chat.availability(ModelState.Failed("org/pkg", "no inference stage"))
        assertTrue(availability is Availability.Blocked)
        assertTrue((availability as Availability.Blocked).reason.contains("org/pkg"))
    }

    @Test
    fun loadingSaysSoRatherThanLookingBroken() {
        val availability = Destination.Chat.availability(ModelState.Loading("org/pkg"))
        assertTrue(availability is Availability.Blocked)
        assertTrue((availability as Availability.Blocked).reason.contains("loading"))
    }

    /**
     * Nothing is hidden while no model is loaded.
     *
     * Hiding is reserved for "this package cannot do that" — an empty drawer on first launch would
     * remove the only thing the app can tell a new user, which is what the app is for.
     */
    @Test
    fun theDrawerIsCompleteBeforeAnythingIsLoaded() {
        assertEquals(Destination.entries.toList(), visibleDestinations(ModelState.None))
    }

    @Test
    fun aDestinationThatIsStillVisibleIsNotRedirectedAwayFrom() {
        assertEquals(Destination.Chat, redirectFor(Destination.Chat, ModelState.None))
        assertEquals(Destination.Train, redirectFor(Destination.Train, ModelState.Loading("x")))
    }
}

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

        for (d in listOf(Destination.Chat, Destination.Retrieval, Destination.Train, Destination.Federated)) {
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

    // ------------------------------------------------------------------ per-package capabilities

    /**
 * Classify exists only for a package that classifies **and** names its labels.
 *
 * The second half is the part worth pinning. A classification graph with no `id2label` runs fine
 * and answers `LABEL_3`, so keying the destination on `isClassifier` alone would offer a screen of
 * probability bars against labels that mean nothing — see `RuntimeCapabilities.supportsClassification`.
 */
    @Test
    fun classifyAppearsOnlyForAClassifierThatNamesItsLabels() {
        assertEquals(Availability.Enabled, Destination.Classify.availabilityFor(classifier()))
        assertEquals(Availability.Hidden, Destination.Classify.availabilityFor(classifierWithoutLabels()))
        assertEquals(Availability.Hidden, Destination.Classify.availabilityFor(decoder()))
}

    /** The inverse: an encoder has no generative head, so the generative screens go away. */
    @Test
    fun theGenerativeScreensAreHiddenOnAClassifier() {
        assertEquals("Chat must hide on a classifier", Availability.Hidden, Destination.Chat.availabilityFor(classifier()))
        assertEquals("Chat must show on a decoder", Availability.Enabled, Destination.Chat.availabilityFor(decoder()))
        }

    /**
 * A plain embedding model is not a classifier and cannot generate either.
 *
 * The check used to be `isClassifier`, which covered a `text-classification` encoder and missed
 * `feature-extraction` — so `all-MiniLM-L6-v2` pulled on its own was offered a chat box for a
 * head it does not have. It is exactly the package the Retrieval screen exists to serve.
 */
    @Test
    fun chatIsHiddenForAPlainEmbeddingModelToo() {
        assertEquals(Availability.Hidden, Destination.Chat.availabilityFor(embeddingEncoder()))
        }

    /** Retrieval needs the embedding stage and nothing else — not a generative head. */
    @Test
    fun retrievalFollowsTheEmbeddingStageRatherThanTheTask() {
        assertEquals(
            "an encoder with an embedding stage must offer Retrieval",
            Availability.Enabled,
            Destination.Retrieval.availabilityFor(embeddingEncoder(rag = true)),
            )
        assertEquals(
            "a decoder with RAG installed must offer Retrieval",
            Availability.Enabled,
            Destination.Retrieval.availabilityFor(caps(decoderTask(), rag = true)),
            )
        // Blocked, not Hidden: RAG is a download group away, so the reason IS the instruction.
            assertTrue(
            "a package with no embedding stage must explain how to get one",
            Destination.Retrieval.availabilityFor(decoder()) is Availability.Blocked,
            )
        }

    /**
 * Loading a classifier while sitting on Chat must move the user somewhere that exists — the
 * destination they are on has just left the drawer, and the drawer is behind the screen.
 */
    @Test
    fun aClassifierAndADecoderNeverOfferTheSameScreens() {
        val forClassifier = Destination.entries.filter { it.availabilityFor(classifier()) !is Availability.Hidden }
        val forDecoder = Destination.entries.filter { it.availabilityFor(decoder()) !is Availability.Hidden }

        assertTrue(Destination.Classify in forClassifier)
        assertTrue(Destination.Classify !in forDecoder)
        assertTrue(Destination.Chat in forDecoder)
        assertTrue(Destination.Chat !in forClassifier)
        }

    private fun caps(
        task: com.martinkorelic.mobiletransformers.packages.PackageTask,
        training: Boolean = true,
        rag: Boolean = false,
) = com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities(
        engine = com.martinkorelic.mobiletransformers.runtime.InferenceEngine.NATIVE,
        supportsTraining = training,
        supportsMerge = false,
        supportsRag = rag,
        supportsEmbedding = rag,
        task = task,
            )

    private fun decoderTask() = com.martinkorelic.mobiletransformers.packages.PackageTask(
        declaredTask = "text-generation-with-past",
        modelType = "llama",
            )

    /** `feature-extraction`: an embedding model with no head of any kind. */
    private fun embeddingEncoder(rag: Boolean = true) = caps(
        com.martinkorelic.mobiletransformers.packages.PackageTask(
            declaredTask = "feature-extraction",
            modelType = "bert",
),
        rag = rag,
            )

    private fun classifier() = caps(
        com.martinkorelic.mobiletransformers.packages.PackageTask(
            declaredTask = "text-classification",
            modelType = "bert",
            id2label = mapOf(0 to "negative", 1 to "positive"),
),
            )

    /** Runs, but every prediction would read `LABEL_n`. */
    private fun classifierWithoutLabels() = caps(
        com.martinkorelic.mobiletransformers.packages.PackageTask(
            declaredTask = "text-classification",
            modelType = "bert",
            id2label = emptyMap(),
),
            )

    private fun decoder() = caps(
        com.martinkorelic.mobiletransformers.packages.PackageTask(
            declaredTask = "text-generation-with-past",
            modelType = "llama",
),
            )
}

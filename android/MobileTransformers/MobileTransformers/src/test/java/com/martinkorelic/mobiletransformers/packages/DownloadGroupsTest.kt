package com.martinkorelic.mobiletransformers.packages

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Which download groups a feature set implies — the one mapping both download paths must agree on.
 *
 * `MobileTransformers.fromPretrained` and `PackageDownloadWorker` each reach `HubDownloader` with a
 * set of group names. They used to compute it separately (the worker took whatever strings its caller
 * passed), so a background pull could install a different package than a foreground one for the same
 * request — and the difference would only surface later as a missing `train/` stage. One function
 * now, called by both, and this pins it.
 */
class DownloadGroupsTest {

    /**
     * Every package has an inference stage, and a pull that omitted it would install a training stage
     * with nothing to train against.
     */
    @Test
    fun inferenceIsAlwaysIncludedEvenWhenNotAskedFor() {
        assertEquals(setOf("inference"), DeviceCapabilities.downloadGroups(emptySet()))
        assertTrue("inference" in DeviceCapabilities.downloadGroups(setOf(ModelFeature.Training)))
    }

    @Test
    fun trainingAddsTheTrainGroup() {
        assertEquals(
            setOf("inference", "train"),
            DeviceCapabilities.downloadGroups(setOf(ModelFeature.Inference, ModelFeature.Training)),
        )
    }

    /**
     * Rag and Embedding are the same download. Embedding alone must still fetch it — the encoder is
     * what `ingest` embeds with, and without the group there is nothing to embed and every grounded
     * query returns zero sources.
     */
    @Test
    fun ragAndEmbeddingBothSelectTheRagGroup() {
        val viaRag = DeviceCapabilities.downloadGroups(setOf(ModelFeature.Rag))
        val viaEmbedding = DeviceCapabilities.downloadGroups(setOf(ModelFeature.Embedding))

        assertEquals(setOf("inference", "rag"), viaRag)
        assertEquals(viaRag, viaEmbedding)
    }

    /**
     * Engine selectors are not downloads. GenAI runs over the SAME package — requesting it must not
     * add a group, or the pull would look for files no package publishes.
     */
    @Test
    fun engineSelectorsAddNoGroup() {
        for (selector in listOf(ModelFeature.GenAI, ModelFeature.ManualInference)) {
            assertTrue("$selector must be an engine selector", selector.isEngineSelector)
            assertEquals(
                "$selector must not add a download group",
                setOf("inference"),
                DeviceCapabilities.downloadGroups(setOf(selector)),
            )
        }
    }

    @Test
    fun everythingAtOnce() {
        assertEquals(
            setOf("inference", "train", "rag"),
            DeviceCapabilities.downloadGroups(
                setOf(ModelFeature.Inference, ModelFeature.Training, ModelFeature.Rag, ModelFeature.GenAI),
            ),
        )
    }

    /** Adapter is not a download group either; only train and rag are. */
    @Test
    fun anUnmappedFeatureDoesNotInventAGroup() {
        val groups = DeviceCapabilities.downloadGroups(setOf(ModelFeature.Adapter))
        assertEquals(setOf("inference"), groups)
        assertFalse("adapter" in groups)
    }
}

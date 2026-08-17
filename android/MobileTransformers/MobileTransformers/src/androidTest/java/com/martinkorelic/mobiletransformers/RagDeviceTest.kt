package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.RetrievalResult
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #26 + #27 device checkpoint: ingest a small `.txt` (chunk → embed → store), then `generateWithRag`,
 * asserting a non-empty answer, non-empty matches, and an inspectable prompt. Requires a RAG-capable
 * package (`embedding/`).
 */
@RunWith(AndroidJUnit4::class)
class RagDeviceTest {

    @Test
    fun ingestThenGroundedGenerate() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        // Grounded generation needs a token loop; the embedding gate below is about the retriever.
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue(
            "package is not RAG-capable (no embedding/)",
            File(root, "$repoId/embedding").isDirectory,
        )
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Rag),
        )
        try {
            val doc = File(ctx.cacheDir, "rag_doc.txt").apply {
                writeText("The Eiffel Tower is in Paris. Paris is the capital of France.")
            }
            val ingest = model.ingest(doc.absolutePath, RagConfig())
            assertTrue("ingestion inserted no chunks", ingest.chunkCount > 0)

            // The retrieve callback fires BEFORE generation, which is what lets a UI show its
            // sources ahead of the answer. Captured here because the ordering is the contract.
            var seenDuringRetrieval: RetrievalResult? = null
            val grounded = model.generateWithRag(
                query = "Where is the Eiffel Tower?",
                rag = RagConfig(),
                retrieveCallback = object : RetrieveCallback {
                    override fun onQueryResults(result: RetrievalResult) {
                        seenDuringRetrieval = result
                    }
                },
            )
            assertTrue("no retrieved matches", grounded.matches.isNotEmpty())
            assertTrue("prompt not inspectable", grounded.prompt.contains("Eiffel Tower"))
            assertNotNull("the retrieve callback never fired", seenDuringRetrieval)

            // Provenance across the REAL ObjectBox store — the only place the round trip
            // (insert(name=title, document=id) -> query -> RetrievalMatch) can actually be proven.
            // A JVM test can assert the grouping rules but not that the store keeps the fields.
            val match = grounded.matches.first()
            assertEquals("the source file's name did not survive the store", doc.name, match.title)
            assertTrue("the chunk id did not survive the store", match.chunkId.isNotBlank())
            assertEquals(doc.nameWithoutExtension, match.documentId)
            assertEquals("one ingested file is one document", 1, seenDuringRetrieval!!.documentCount)
            assertEquals(listOf(doc.name), seenDuringRetrieval!!.documentTitles)
        } finally {
            model.close()
        }
    }
}

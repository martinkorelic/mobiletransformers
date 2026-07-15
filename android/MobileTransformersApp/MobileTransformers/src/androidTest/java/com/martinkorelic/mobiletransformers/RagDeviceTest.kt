package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import java.io.File
import kotlinx.coroutines.runBlocking
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

            val grounded = model.generateWithRag("Where is the Eiffel Tower?", RagConfig())
            assertTrue("no retrieved matches", grounded.matches.isNotEmpty())
            assertTrue("prompt not inspectable", grounded.prompt.contains("Eiffel Tower"))
        } finally {
            model.close()
        }
    }
}

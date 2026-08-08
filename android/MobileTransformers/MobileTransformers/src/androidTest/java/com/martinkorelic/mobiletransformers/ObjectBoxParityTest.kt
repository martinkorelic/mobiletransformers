package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.rag.ObjectBoxVectorStore
import com.martinkorelic.mobiletransformers.rag.RagDocument
import kotlin.math.sqrt
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #25 device parity smoke: the real ObjectBox HNSW store must agree with a plain cosine reference.
 *
 * The JVM suite only ever exercised `InMemoryVectorStore`, so the score contract that actually ships —
 * ObjectBox returns COSINE *distance* and `ORTVectorDatabase` converts it to `1 - distance` — had never
 * been checked against real ObjectBox on a device. This closes that: same vectors into both, compare
 * ranking and similarity.
 *
 * Needs no model package (it inserts its own vectors), so it runs on any connected device.
 */
@RunWith(AndroidJUnit4::class)
class ObjectBoxParityTest {

    private var store: ObjectBoxVectorStore? = null

    @After
    fun tearDown() {
        store?.close()
    }

    /** Reference cosine similarity — what `RagMatch.score` is contractually supposed to carry. */
    private fun cosine(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return if (na == 0.0 || nb == 0.0) 0.0 else dot / (sqrt(na) * sqrt(nb))
    }

    private fun unit(dim: Int, seed: Int): FloatArray {
        val rnd = java.util.Random(seed.toLong())
        val v = FloatArray(dim) { rnd.nextGaussian().toFloat() }
        val norm = sqrt(v.fold(0.0) { acc, x -> acc + x * x }).toFloat()
        return FloatArray(dim) { v[it] / norm }
    }

    @Test
    fun objectBoxRankingAndScoresMatchCosineReference() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dim = 384
        val cacheDir = ctx.cacheDir.absolutePath
        val modelName = "objectbox_parity_${System.currentTimeMillis()}"
        val config = ORTRagConfig(repoName = modelName, embeddingDimension = dim)
        val db = ORTVectorDatabase.getInstance(modelName, ctx, cacheDir, config)
        val obx = ObjectBoxVectorStore(db).also { store = it }

        val embeddings = (0 until 8).associateWith { unit(dim, it + 1) }
        embeddings.forEach { (i, emb) ->
            val id = obx.insert(RagDocument(id = "doc$i", title = "t$i", text = "body $i"), emb)
            assertTrue("insert failed for doc$i (id=$id)", id >= 0)
        }
        assertEquals(8L, obx.count())

        // Query near doc3 but not identical, so the ordering is a real ranking rather than an exact hit.
        val base = embeddings.getValue(3)
        val query = FloatArray(dim) { base[it] + 0.01f * unit(dim, 99)[it] }

        val expected = embeddings.entries
            .map { (i, emb) -> "doc$i" to cosine(query, emb) }
            .sortedByDescending { it.second }

        val topK = 4
        val actual = obx.search(query, topK = topK, minScore = 0.0)

        assertEquals("topK not honoured", topK, actual.size)
        assertEquals(
            "ObjectBox ranking differs from the cosine reference",
            expected.take(topK).map { it.first },
            actual.map { it.document.id },
        )
        actual.forEachIndexed { rank, match ->
            val ref = expected[rank].second
            assertEquals(
                "similarity mismatch at rank $rank for ${match.document.id} " +
                    "(is the 1 - distance conversion still applied?)",
                ref,
                match.score,
                1e-3,
            )
        }

        // The store must hand back the document it was given, embeddings stripped (#25's no-leak rule).
        assertEquals("body 3", actual.first().document.text)

        // minScore filters on the similarity, not the raw distance.
        val floor = expected[1].second
        val filtered = obx.search(query, topK = topK, minScore = floor)
        assertTrue(
            "minScore admitted a hit below the floor",
            filtered.all { it.score >= floor - 1e-6 },
        )
    }
}

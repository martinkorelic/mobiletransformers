package com.martinkorelic.mobiletransformers.rag

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for the #25 VectorStore boundary (no Android / no ObjectBox). Exercises the pure
 * [InMemoryVectorStore] against the same semantics the on-device [ObjectBoxVectorStore] preserves:
 * cosine ordering, `minScore`, embeddings stripped, the non-ranked text path, and the fail-closed
 * dimension registry.
 */
class InMemoryVectorStoreTest {

    private val dim = 64

    /** A 64-d vector with the given leading components; the rest are zero. */
    private fun vec(vararg leading: Float): FloatArray = FloatArray(dim).also { arr ->
        for (i in leading.indices) arr[i] = leading[i]
    }

    private fun doc(id: String, text: String = id) = RagDocument(id = id, title = id, text = text)

    private fun store(): VectorStore = InMemoryVectorStore(dim)

    // --- insert / count ---------------------------------------------------------

    @Test
    fun insertAssignsIdsAndCounts() {
        val s = store()
        assertEquals(0L, s.count())
        val id1 = s.insert(doc("a"), vec(1f, 0f))
        val id2 = s.insert(doc("b"), vec(0f, 1f))
        assertTrue(id1 > 0 && id2 > id1)
        assertEquals(2L, s.count())
    }

    @Test
    fun insertRejectsWrongDimension() {
        val s = store()
        assertThrows(IllegalArgumentException::class.java) {
            s.insert(doc("a"), FloatArray(dim + 1))
        }
    }

    // --- cosine ordering + similarities (hand-computed) -------------------------

    @Test
    fun searchOrdersByCosineSimilarity() {
        val s = store()
        s.insert(doc("x"), vec(1f, 0f)) // unit-x
        s.insert(doc("y"), vec(0f, 1f)) // unit-y
        // query [0.6, 0.8] (unit): cos(x)=0.6, cos(y)=0.8 -> y first, then x.
        val matches = s.search(vec(0.6f, 0.8f), topK = 2)
        assertEquals(listOf("y", "x"), matches.map { it.document.id })
        assertEquals(0.8, matches[0].score, 1e-4)
        assertEquals(0.6, matches[1].score, 1e-4)
    }

    @Test
    fun searchHonorsTopK() {
        val s = store()
        s.insert(doc("x"), vec(1f, 0f))
        s.insert(doc("y"), vec(0f, 1f))
        s.insert(doc("z"), vec(1f, 1f))
        assertEquals(2, s.search(vec(1f, 0f), topK = 2).size)
    }

    /**
     * Score-semantics: an identical vector yields similarity 1.0 and an orthogonal one 0.0 — matching
     * ObjectBox's `1 - distance` conversion (ORTVectorDatabase.kt:262). `minScore` filters on that
     * similarity: at 0.5 the orthogonal hit is dropped.
     */
    @Test
    fun minScoreFiltersOnSimilarity() {
        val s = store()
        s.insert(doc("same"), vec(1f, 0f))
        s.insert(doc("orthogonal"), vec(0f, 1f))
        val all = s.search(vec(1f, 0f), topK = 10)
        assertEquals(1.0, all.first { it.document.id == "same" }.score, 1e-6)
        assertEquals(0.0, all.first { it.document.id == "orthogonal" }.score, 1e-6)

        val filtered = s.search(vec(1f, 0f), topK = 10, minScore = 0.5)
        assertEquals(listOf("same"), filtered.map { it.document.id })
    }

    // --- no embeddings leak out -------------------------------------------------

    @Test
    fun resultsCarryNoEmbeddingVector() {
        // The boundary returns only a RagDocument (id/title/text/metadata) + score — the FloatArray
        // embedding never crosses it (mirrors ObjectBox's :226 strip). RagDocument has no embedding
        // field, so a leak is impossible by construction; verify the returned document is exactly the
        // inserted one and carries no vector data.
        val s = store()
        val d = doc("d1", text = "hello")
        s.insert(d, vec(1f, 0f))
        val match = s.search(vec(1f, 0f), topK = 1).single()
        assertEquals(d, match.document)
    }

    // --- text search ------------------------------------------------------------

    @Test
    fun textSearchReturnsSubstringHitsWithFixedScore() {
        val s = store()
        s.insert(doc("d1", text = "The quick brown fox"), vec(1f, 0f))
        s.insert(doc("d2", text = "lazy dog sleeps"), vec(0f, 1f))
        val hits = s.textSearch("BROWN", topK = 10) // case-insensitive substring
        assertEquals(listOf("d1"), hits.map { it.document.id })
        assertEquals(TEXT_SEARCH_SCORE, hits[0].score, 0.0)
    }

    // --- dimension registry (fail closed) ---------------------------------------

    @Test
    fun unsupportedDimensionFailsClosed() {
        assertFalse(DimensionRegistry.isSupported(300))
        assertThrows(IllegalArgumentException::class.java) { InMemoryVectorStore(300) }
    }

    @Test
    fun registeredDimensionIsAccepted() {
        assertFalse(DimensionRegistry.isSupported(301))
        DimensionRegistry.register(301)
        assertTrue(DimensionRegistry.isSupported(301))
        // Construction now succeeds (no throw) for the newly declared dimension.
        val s = InMemoryVectorStore(301)
        assertEquals(0L, s.count())
    }

    @Test
    fun defaultDimensionsAreDeclared() {
        assertTrue(DimensionRegistry.SUPPORTED_DIMENSIONS.containsAll(setOf(64, 128, 256, 384, 512, 768, 1024, 1536)))
    }

    // --- pluggable backend registry (F4) ----------------------------------------

    @Test
    fun registryCreatesRegisteredBackend() {
        VectorStoreRegistry.register("inmemory") { ctx -> InMemoryVectorStore(ctx.embeddingDimension) }
        val s = VectorStoreRegistry.create("inmemory", VectorStoreContext(embeddingDimension = 64))
        assertEquals(0L, s.count())
        assertTrue(VectorStoreRegistry.keys().contains(VectorStoreRegistry.DEFAULT_KEY))
    }

    @Test
    fun registryRejectsUnknownBackend() {
        assertThrows(IllegalArgumentException::class.java) {
            VectorStoreRegistry.create("does-not-exist", VectorStoreContext(embeddingDimension = 64))
        }
    }
}

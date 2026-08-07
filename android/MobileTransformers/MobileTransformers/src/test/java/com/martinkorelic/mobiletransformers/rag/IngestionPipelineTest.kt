package com.martinkorelic.mobiletransformers.rag

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * #26: the pure chunk → embed → insert pipeline over `InMemoryVectorStore` with a fake embedder — proves
 * ingestion + progress without JNI/ObjectBox.
 */
class IngestionPipelineTest {

    private val dim = 64
    private fun fakeEmbedder(): (String) -> FloatArray = { s ->
        // deterministic non-zero embedding derived from the chunk (content-independent shape is fine here)
        FloatArray(dim) { i -> ((s.length + i) % 7 + 1).toFloat() }
    }

    @Test
    fun insertsOneRowPerChunk() = runBlocking {
        val store = InMemoryVectorStore(dim)
        val text = (0 until 100).joinToString("") { (it % 10).toString() }
        val inserted = IngestionPipeline.ingest(
            documents = listOf(RagDocument("doc1", "Doc 1", text)),
            chunkSize = 40,
            chunkOverlap = 10,
            embed = fakeEmbedder(),
            store = store,
        )
        assertEquals(3, inserted) // stride 30 over 100 chars -> 3 windows
        assertEquals(3L, store.count())
    }

    @Test
    fun progressSequencePerDocument() = runBlocking {
        val store = InMemoryVectorStore(dim)
        val events = mutableListOf<String>()
        val progress = object : IngestionProgress {
            override fun onDocumentStart(id: String, totalDocs: Int) { events += "start:$id:$totalDocs" }
            override fun onChunkEmbedded(docId: String, chunkIndex: Int, totalChunks: Int) {
                events += "chunk:$docId:$chunkIndex/$totalChunks"
            }
            override fun onDocumentComplete(id: String) { events += "done:$id" }
            override fun onError(id: String?, error: Throwable) { events += "err:$id" }
        }
        IngestionPipeline.ingest(
            documents = listOf(RagDocument("d", "D", "abcdefghij")), // 10 chars
            chunkSize = 5,
            chunkOverlap = 0,
            embed = fakeEmbedder(),
            store = store,
            progress = progress,
        )
        assertEquals(
            listOf("start:d:1", "chunk:d:0/2", "chunk:d:1/2", "done:d"),
            events,
        )
    }

    @Test
    fun embedderReturningNullReportsErrorAndSkipsDocument() = runBlocking {
        val store = InMemoryVectorStore(dim)
        var errored = false
        val progress = object : IngestionProgress {
            override fun onError(id: String?, error: Throwable) { errored = true }
        }
        val inserted = IngestionPipeline.ingest(
            documents = listOf(RagDocument("d", "D", "some text here")),
            chunkSize = 5,
            chunkOverlap = 0,
            embed = { null }, // embedding fails
            store = store,
            progress = progress,
        )
        assertEquals(0, inserted)
        assertEquals(0L, store.count())
        assertEquals(true, errored)
    }

    @Test
    fun multipleDocumentsGetPrefixedChunkIds() = runBlocking {
        val store = InMemoryVectorStore(dim)
        IngestionPipeline.ingest(
            documents = listOf(RagDocument("a", "A", "abcde"), RagDocument("b", "B", "fghij")),
            chunkSize = 5,
            chunkOverlap = 0,
            embed = fakeEmbedder(),
            store = store,
        )
        assertEquals(2L, store.count())
    }
}

package com.martinkorelic.mobiletransformers.rag

import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** #26: F3 document-loader registry + fail-closed on unsupported extensions. */
class DocumentSourceTest {

    private val dir: File = Files.createTempDirectory("docsource").toFile()

    @After
    fun cleanup() {
        dir.deleteRecursively()
    }

    private fun write(name: String, content: String): File =
        File(dir, name).apply { writeText(content) }

    @Test
    fun txtMapsToOneRecord() {
        val docs = loadDocuments(write("notes.txt", "hello world").path)
        assertEquals(1, docs.size)
        assertEquals("notes", docs[0].id)
        assertEquals("notes.txt", docs[0].title)
        assertEquals("hello world", docs[0].text)
    }

    @Test
    fun mdMapsToOneRecord() {
        val docs = loadDocuments(write("readme.md", "# Title\nbody").path)
        assertEquals(1, docs.size)
        assertEquals("# Title\nbody", docs[0].text)
    }

    @Test
    fun jsonlParsesRecordPerLine() {
        val jsonl = """
            {"id":"a","title":"A","text":"alpha","metadata":{"k":"v"}}
            {"id":"b","title":"B","text":"beta"}
        """.trimIndent()
        val docs = loadDocuments(write("data.jsonl", jsonl).path)
        assertEquals(2, docs.size)
        assertEquals("a", docs[0].id)
        assertEquals("alpha", docs[0].text)
        assertEquals("v", docs[0].metadata["k"])
        assertEquals("beta", docs[1].text)
    }

    @Test
    fun unsupportedExtensionRejected() {
        val ex = assertThrows(IllegalArgumentException::class.java) {
            loadDocuments(write("doc.pdf", "%PDF-1.4").path)
        }
        assertTrue(ex.message!!.contains("text/Markdown/JSONL only"))
        assertThrows(IllegalArgumentException::class.java) {
            loadDocuments(write("doc.docx", "PK").path)
        }
    }

    @Test
    fun extensionMatchIsCaseInsensitive() {
        val docs = loadDocuments(write("UP.TXT", "x").path)
        assertEquals(1, docs.size)
    }

    @Test
    fun registryHasOnlyV1Formats() {
        assertEquals(setOf("txt", "md", "jsonl"), DOCUMENT_LOADER_REGISTRY.keys)
    }
}

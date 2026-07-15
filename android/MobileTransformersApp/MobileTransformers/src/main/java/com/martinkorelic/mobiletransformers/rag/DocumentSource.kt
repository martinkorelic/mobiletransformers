package com.martinkorelic.mobiletransformers.rag

import com.google.gson.Gson
import java.io.File

/**
 * #26: document loaders behind a data-driven registry (F3). v1 supports plain text, Markdown, and JSONL;
 * a new format is one [DOCUMENT_LOADER_REGISTRY] row — no pipeline edit. PDF/Word/HTML are explicitly out
 * of v1 scope and rejected fail-closed.
 */
fun interface DocumentLoader {
    fun load(file: File): List<RagDocument>
}

private val gson = Gson()

private fun textRecord(file: File): List<RagDocument> =
    listOf(RagDocument(id = file.nameWithoutExtension, title = file.name, text = file.readText()))

private data class JsonlRecord(
    val id: String? = null,
    val title: String? = null,
    val text: String? = null,
    val metadata: Map<String, String>? = null,
)

private fun jsonlRecords(file: File): List<RagDocument> =
    file.readLines().filter { it.isNotBlank() }.mapIndexed { i, line ->
        val o = gson.fromJson(line, JsonlRecord::class.java) ?: JsonlRecord()
        RagDocument(
            id = o.id ?: "${file.nameWithoutExtension}#$i",
            title = o.title ?: file.name,
            text = o.text ?: "",
            metadata = o.metadata ?: emptyMap(),
        )
    }

/** Extension (lowercase) -> loader. New formats slot in here (F3). */
val DOCUMENT_LOADER_REGISTRY: Map<String, DocumentLoader> =
    mapOf(
        "txt" to DocumentLoader { textRecord(it) },
        "md" to DocumentLoader { textRecord(it) },
        "jsonl" to DocumentLoader { jsonlRecords(it) },
    )

/** Load [path] into records via the registry; fail closed on an unsupported extension. */
fun loadDocuments(path: String): List<RagDocument> {
    val file = File(path)
    require(file.isFile) { "not a file: $path" }
    val ext = file.extension.lowercase()
    val loader =
        DOCUMENT_LOADER_REGISTRY[ext]
            ?: throw IllegalArgumentException("v1 supports text/Markdown/JSONL only, got '.$ext' ($path)")
    return loader.load(file)
}

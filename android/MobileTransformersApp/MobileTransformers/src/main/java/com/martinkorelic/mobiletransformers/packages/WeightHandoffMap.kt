package com.martinkorelic.mobiletransformers.packages

import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.martinkorelic.mobiletransformers.MissingArtifactException
import java.io.File

/**
 * Kotlin read model of `weight_handoff_map.json` (#8 schema / #23 load-side consumer).
 *
 * The schema is owned by the Python side (`artifacts/handoff_map.py`); Gson ignores unknown fields so
 * additive minor bumps stay non-breaking (F1). Only the fields the native LOAD path needs are modeled:
 * per-role external `.bin` filenames ([Entry.externalDataLocation]), the canonical initializer names
 * ([Entry.inferenceInitializerNames]), dtype/shape, and the optional per-role `sha256` the exporter
 * stamps. The C++ merger owns the WRITE side (`weight_merger.cpp`), which also writes a sibling
 * `<name>.bin.sha256` next to each `.bin`.
 */
data class WeightHandoffMap(
    val schemaVersion: String = "1.0",
    val minReaderVersion: String = "1.0",
    val handoffMode: String = "external_initializer",
    val externalDataLayout: String = "one_file_per_tensor",
    val entries: List<Entry> = emptyList(),
) {
    data class Entry(
        val trainingBaseLayerName: String = "",
        val dtype: String = "",
        val shape: List<Long> = emptyList(),
        val inferenceInitializerNames: Map<String, String> = emptyMap(),
        val externalDataLocation: Map<String, String> = emptyMap(),
        val sha256: Map<String, String> = emptyMap(),
    )

    companion object {
        const val READER_VERSION = "1.0"
        const val FILENAME = "weight_handoff_map.json"
        private val gson = Gson()

        fun parse(json: String): WeightHandoffMap =
            try {
                gson.fromJson(json, WeightHandoffMap::class.java)
                    ?: throw MissingArtifactException("weight_handoff_map.json parsed to null")
            } catch (e: JsonSyntaxException) {
                throw MissingArtifactException("weight_handoff_map.json is invalid JSON: ${e.message}")
            }

        fun load(file: File): WeightHandoffMap {
            if (!file.isFile) throw MissingArtifactException("weight_handoff_map.json not found: ${file.path}")
            return parse(file.readText(Charsets.UTF_8))
        }
    }
}

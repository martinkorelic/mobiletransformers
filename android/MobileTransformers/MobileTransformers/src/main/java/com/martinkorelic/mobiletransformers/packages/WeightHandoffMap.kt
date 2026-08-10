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
        /** Entry-level dtype/shape: the weight-like role's, and the fallback for [dtypeFor]/[shapeFor]. */
        val dtype: String = "",
        val shape: List<Long> = emptyList(),
        /**
         * Per-role on-disk dtype/shape. Each `<name>.bin` holds RAW external-data bytes with no header,
         * so this is the native loader's only description of a packed `weight_quantized`/`scale`/
         * `zero_point`, whose layout differs from the entry-level weight's. Empty on maps written before
         * the field existed.
         */
        val tensorDtypes: Map<String, String> = emptyMap(),
        val tensorShapes: Map<String, List<Long>> = emptyMap(),
        val inferenceInitializerNames: Map<String, String> = emptyMap(),
        val externalDataLocation: Map<String, String> = emptyMap(),
        /** Per-role digest of the SHIPPED bytes. The live digest is `<name>.bin.sha256` and wins. */
        val sha256: Map<String, String> = emptyMap(),
        /**
         * ORT **checkpoint parameter** names per role — `backbone.model…lora_A.lora.weight`, not the
         * PEFT module path. This is the identity a federated client looks a tensor up by (#36).
         */
        val checkpointNames: Map<String, String> = emptyMap(),
        /**
         * Schema **1.1**: dtype/shape of the rank-r ADAPTER FACTORS, per adapter role.
         *
         * `tensorDtypes`/`tensorShapes` above describe the MERGED inference initializers — a different
         * set of objects (60 merged weights vs 120 rank-r factors on SmolLM2). Without these the map
         * could not describe the tensors federation actually exchanges, and a client would have to
         * infer shapes from the rank — the re-derivation that causes layer-identity defects.
         *
         * Additive: `minReaderVersion` stays 1.0 and a 1.0 map simply lacks them, which
         * [adapterTensorSpecs] reports as a fail-closed error naming the re-export, never a silent
         * fallback to merged weights.
         */
        val adapterDtypes: Map<String, String> = emptyMap(),
        val adapterShapes: Map<String, List<Long>> = emptyMap(),
    ) {
        fun dtypeFor(role: String): String = tensorDtypes[role] ?: dtype

        fun shapeFor(role: String): List<Long> = tensorShapes[role] ?: shape

        /**
         * This entry's adapter factors in codec order, or an empty list when it declares none.
         *
         * Mirrors `HandoffEntry.adapter_tensor_specs`, including its rule that a role missing from
         * EITHER map is skipped here — the fail-closed decision is made one level up, in
         * [WeightHandoffMap.adapterTensorSpecs], so the message can name the whole package.
         */
        fun adapterTensorSpecs(): List<AdapterTensorSpec> = ADAPTER_ROLE_ORDER.mapNotNull { role ->
            val dtype = adapterDtypes[role] ?: return@mapNotNull null
            val shape = adapterShapes[role] ?: return@mapNotNull null
            val checkpoint = checkpointNames[role] ?: return@mapNotNull null
            AdapterTensorSpec(
                name = "${toCheckpointName(checkpoint)}.weight",
                dtype = dtype,
                shape = shape,
                role = role,
            )
        }
    }

    /** One tensor a federated record carries: the checkpoint identity plus its on-wire description. */
    data class AdapterTensorSpec(
        val name: String,
        val dtype: String,
        val shape: List<Long>,
        val role: String,
        val aggregationRole: String = "adapter_only",
    ) {
        val elementCount: Long get() = shape.fold(1L) { acc, d -> acc * d }
    }

    /**
     * Every adapter factor in the package, in **codec order**: entries sorted by canonical weight name,
     * each expanded by [ADAPTER_ROLE_ORDER].
     *
     * Mirrors `federated/adapter_record.py::codec_tensor_specs`, including the fail-closed behaviour: a
     * package exported before schema 1.1 cannot describe its factors, and that is an error naming the
     * re-export rather than a silent fallback to merged weights.
     */
    fun adapterTensorSpecs(): List<AdapterTensorSpec> {
        val specs = sortedEntries().flatMap { it.adapterTensorSpecs() }
        if (specs.isEmpty()) {
            throw MissingArtifactException(
                "weight_handoff_map.json (schemaVersion $schemaVersion) describes no adapter factors: " +
                    "it carries no adapterDtypes/adapterShapes, so this package predates schema 1.1. " +
                    "Re-export it with a current exporter — a federated round cannot infer factor " +
                    "shapes from the rank."
            )
        }
        return specs
    }

    /** Entry order: by the merged weight's inference initializer name, falling back to the base layer. */
    fun sortedEntries(): List<Entry> = entries.sortedBy {
        it.inferenceInitializerNames["weight"] ?: it.trainingBaseLayerName
    }

    companion object {
        const val READER_VERSION = "1.1"
        const val FILENAME = "weight_handoff_map.json"

        /**
         * Adapter roles in codec order. A HARD CONSTANT, mirroring `HandoffEntry.ADAPTER_ROLE_ORDER`.
         * The wire format is (entries by canonical weight name) x this order; changing it changes the
         * bytes, which the cross-language golden exists to prevent.
         */
        val ADAPTER_ROLE_ORDER = listOf("shared_A", "intermediate", "adapter_A", "adapter_B")

        /**
         * Kotlin twin of `artifacts/checkpoint_names.py::to_checkpoint_name` (and `layer_name.h`'s
         * `to_checkpoint`): `base_model.model.model.` -> `backbone.model.`.
         */
        fun toCheckpointName(name: String): String =
            if (name.startsWith("base_model.model.model.")) {
                "backbone.model." + name.removePrefix("base_model.model.model.")
            } else {
                name
            }

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

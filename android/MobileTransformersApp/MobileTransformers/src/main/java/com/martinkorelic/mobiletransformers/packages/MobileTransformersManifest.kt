package com.martinkorelic.mobiletransformers.packages

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.File

/**
 * Kotlin mirror of `mobiletransformers_manifest.json` (#14 schema, #13 consumer).
 *
 * Gson ignores unknown fields, so additive minor schema bumps are non-breaking (F1). The schema/field
 * list is owned by the Python side (`hub/package_format.py`); this is the read model the cache bridge
 * (validator / selector / installer) consumes.
 */
data class MobileTransformersManifest(
    @SerializedName("schemaVersion") val schemaVersion: String = "",
    @SerializedName("minReaderVersion") val minReaderVersion: String = "",
    @SerializedName("baseModelId") val baseModelId: String = "",
    @SerializedName("defaultVariant") val defaultVariant: String = "",
    @SerializedName("variants") val variants: List<Variant> = emptyList(),
    @SerializedName("requiredFiles") val requiredFiles: List<String> = emptyList(),
    @SerializedName("sha256") val sha256: Map<String, String> = emptyMap(),
    @SerializedName("fileSizes") val fileSizes: Map<String, Long> = emptyMap(),
    @SerializedName("downloadPlan") val downloadPlan: Map<String, Map<String, List<String>>> = emptyMap(),
    @SerializedName("weightHandoff") val weightHandoff: String = "",
) {
    data class Variant(
        @SerializedName("id") val id: String = "",
        @SerializedName("executionProvider") val executionProvider: String = "",
        @SerializedName("quantization") val quantization: String = "",
        @SerializedName("supportedEngines") val supportedEngines: List<String> = emptyList(),
        @SerializedName("abi") val abi: List<String>? = null,
        @SerializedName("features") val features: List<String> = emptyList(),
        @SerializedName("minimumAndroidApi") val minimumAndroidApi: Int? = null,
        @SerializedName("recommendedDeviceMemoryMb") val recommendedDeviceMemoryMb: Int? = null,
        @SerializedName("weightHandoff") val weightHandoff: String = "",
        @SerializedName("paths") val paths: Map<String, String> = emptyMap(),
    )

    fun variant(id: String): Variant? = variants.firstOrNull { it.id == id }

    companion object {
        private val gson = Gson()

        fun parse(json: String): MobileTransformersManifest =
            gson.fromJson(json, MobileTransformersManifest::class.java)
                ?: throw ManifestException("manifest JSON parsed to null")

        fun load(file: File): MobileTransformersManifest {
            if (!file.isFile) throw ManifestException("manifest not found: ${file.path}")
            return parse(file.readText(Charsets.UTF_8))
        }
    }
}

/** Parallel to the Python `ManifestError`. */
class ManifestException(message: String) : Exception(message)

/** Parallel to the Python `NoCompatibleVariant`. */
class NoCompatibleVariantException(message: String) : Exception(message)

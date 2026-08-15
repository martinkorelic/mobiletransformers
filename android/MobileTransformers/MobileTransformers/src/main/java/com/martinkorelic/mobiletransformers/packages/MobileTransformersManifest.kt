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
    /**
     * Every parameter the training graph materialises — not the trainable subset.
     *
     * The exporter has written both since the training stage existed and the device read neither.
     * It is what [com.martinkorelic.mobiletransformers.runtime.MemoryHeadroom] estimates from: for a
     * LoRA export the two differ by three orders of magnitude (268,098,176 against 368,640), and
     * sizing memory from the trainable count would under-estimate by the whole model.
     */
    @SerializedName("trainingParameterCount") val trainingParameterCount: Long = 0L,
    @SerializedName("trainableParameterCount") val trainableParameterCount: Long = 0L,
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

    /**
     * The engines the installed variant declares, for [ModelRuntimeFactory.create]'s selection.
     *
     * `ModelRuntimeFactory.create` used to be called with a hard-coded `setOf("native","genai")` and a
     * comment saying the set "would come from the manifest variant (#13)" — so a native-only variant
     * was still offered to GenAI, and the manifest field this class has always parsed was never read.
     *
     * A package that declares no engines at all (an older export) yields `null`, and the caller keeps
     * the permissive default: an unknown declaration must not become a *narrower* one, or upgrading the
     * SDK would break packages that work today.
     */
    fun supportedEnginesFor(variantId: String? = null): Set<String>? {
        val chosen = variantId?.let { variant(it) }
            ?: variants.singleOrNull()
            ?: variant(defaultVariant)
            ?: return null
        return chosen.supportedEngines.takeIf { it.isNotEmpty() }?.toSet()
    }

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

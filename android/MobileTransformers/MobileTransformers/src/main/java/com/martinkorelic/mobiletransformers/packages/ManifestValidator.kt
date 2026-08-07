package com.martinkorelic.mobiletransformers.packages

import java.io.File

/**
 * Validates a [MobileTransformersManifest] against a materialized package directory (#13). Fails closed
 * with [ManifestException]. Mirrors the Python `artifacts.manifest.MobileTransformersManifest.validate`.
 */
object ManifestValidator {
    fun validate(manifest: MobileTransformersManifest, packageDir: File) {
        if (!PackageFormat.checkCompat(
                manifest.schemaVersion,
                manifest.minReaderVersion,
                PackageFormat.MANIFEST_READER_VERSION,
            )
        ) {
            throw ManifestException(
                "manifest schema ${manifest.schemaVersion} (minReader ${manifest.minReaderVersion}) " +
                    "incompatible with reader ${PackageFormat.MANIFEST_READER_VERSION}",
            )
        }
        if (manifest.variants.isEmpty()) throw ManifestException("manifest declares no variants")
        if (manifest.variant(manifest.defaultVariant) == null) {
            throw ManifestException("defaultVariant '${manifest.defaultVariant}' not among variants")
        }
        for (v in manifest.variants) {
            val features = v.features.toSet()
            for ((feature, path) in listOf("train" to "train", "inference" to "inference", "rag" to "embedding")) {
                if (feature in features && !v.paths.containsKey(path)) {
                    throw ManifestException("variant '${v.id}' claims feature '$feature' but has no '$path' path")
                }
            }
            if (v.weightHandoff.isEmpty()) {
                throw ManifestException("variant '${v.id}' has no weightHandoff pointer")
            }
            if (!File(packageDir, v.weightHandoff).isFile) {
                throw ManifestException("variant '${v.id}' weightHandoff does not resolve: ${v.weightHandoff}")
            }
        }
        for (rel in manifest.requiredFiles) {
            if (!File(packageDir, rel).exists()) {
                throw ManifestException("requiredFile missing on disk: $rel")
            }
        }
    }
}

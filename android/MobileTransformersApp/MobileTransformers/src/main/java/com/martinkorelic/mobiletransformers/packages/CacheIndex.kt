package com.martinkorelic.mobiletransformers.packages

import java.io.File

/** Enumerates installed model packages in the cache dir, tolerating legacy (manifest-less) dirs (#13). */
object CacheIndex {
    data class InstalledPackage(
        val sanitizedRepoId: String,
        val dir: File,
        val baseModelId: String?,
        val variantIds: List<String>,
        val sizeBytes: Long,
        val hasManifest: Boolean,
    )

    fun list(cacheDir: File): List<InstalledPackage> {
        val out = mutableListOf<InstalledPackage>()
        val children = cacheDir.listFiles() ?: return out
        for (dir in children.sortedBy { it.name }) {
            if (!dir.isDirectory || dir.name.startsWith(".")) continue
            val manifestFile = File(dir, PackageFormat.MANIFEST_FILENAME)
            if (manifestFile.isFile) {
                val m = runCatching { MobileTransformersManifest.load(manifestFile) }.getOrNull()
                out.add(
                    InstalledPackage(
                        sanitizedRepoId = dir.name,
                        dir = dir,
                        baseModelId = m?.baseModelId,
                        variantIds = m?.variants?.map { it.id } ?: emptyList(),
                        sizeBytes = dirSize(dir),
                        hasManifest = true,
                    ),
                )
            } else {
                // Legacy layout: no manifest, but still a discoverable model dir.
                out.add(
                    InstalledPackage(
                        sanitizedRepoId = dir.name,
                        dir = dir,
                        baseModelId = null,
                        variantIds = emptyList(),
                        sizeBytes = dirSize(dir),
                        hasManifest = false,
                    ),
                )
            }
        }
        return out
    }

    private fun dirSize(dir: File): Long =
        dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
}

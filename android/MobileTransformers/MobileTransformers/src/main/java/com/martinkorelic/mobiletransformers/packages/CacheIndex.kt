package com.martinkorelic.mobiletransformers.packages

import java.io.File

/** Enumerates installed model packages in the cache dir, tolerating legacy (manifest-less) dirs (#13). */
object CacheIndex {
    data class InstalledPackage(
        val sanitizedRepoId: String,
        val dir: File,
        /**
         * The repo id this package was installed from — **the value to pass to `fromPretrained`**.
         *
         * Distinct from [baseModelId], which names the upstream model the package was exported from.
         * The two are routinely different (`mobiletransformers/functiongemma-270m-it` is built from
         * `google/functiongemma-270m-it`), and loading by the wrong one resolves to a cache directory
         * that does not exist. Comes from [InstallRecord]; for a legacy or hand-pushed package with no
         * record, from un-sanitizing the directory name.
         */
        val repoId: String,
        /** The upstream model this package was exported from, per the manifest. Never a load key. */
        val baseModelId: String?,
        val variantIds: List<String>,
        val sizeBytes: Long,
        val hasManifest: Boolean,
        /** The variant materialized here, when an [InstallRecord] says so. */
        val installedVariantId: String? = null,
        /** Feature groups requested when this package was pulled, when recorded. */
        val requestedFeatures: List<String> = emptyList(),
        /** When this package was installed, or `0` when unrecorded. */
        val installedAtEpochMs: Long = 0L,
    )

    fun list(cacheDir: File): List<InstalledPackage> {
        val out = mutableListOf<InstalledPackage>()
        val children = cacheDir.listFiles() ?: return out
        for (dir in children.sortedBy { it.name }) {
            if (!dir.isDirectory || dir.name.startsWith(".")) continue
            val record = InstallRecord.read(dir)
            val manifestFile = File(dir, PackageFormat.MANIFEST_FILENAME)
            val manifest =
                if (manifestFile.isFile) {
                    runCatching { MobileTransformersManifest.load(manifestFile) }.getOrNull()
                } else {
                    null
                }
            out.add(
                InstalledPackage(
                    sanitizedRepoId = dir.name,
                    dir = dir,
                    repoId = record?.repoId ?: InstallRecord.unsanitize(dir.name),
                    baseModelId = manifest?.baseModelId,
                    variantIds = manifest?.variants?.map { it.id } ?: emptyList(),
                    sizeBytes = dirSize(dir),
                    // Legacy layout: no manifest, but still a discoverable model dir.
                    hasManifest = manifestFile.isFile,
                    installedVariantId = record?.variantId?.takeIf { it.isNotBlank() },
                    requestedFeatures = record?.features ?: emptyList(),
                    installedAtEpochMs = record?.installedAtEpochMs ?: 0L,
                ),
            )
        }
        return out
    }

    private fun dirSize(dir: File): Long =
        dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
}

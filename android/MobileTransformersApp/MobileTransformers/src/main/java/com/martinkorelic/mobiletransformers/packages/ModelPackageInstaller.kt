package com.martinkorelic.mobiletransformers.packages

import java.io.File

/**
 * Materializes a downloaded/staged package into the conventional cache layout `LLMRepository` already
 * probes — `<cacheDir>/<sanitizedRepoId>/{train,inference,embedding,tokenizer}` + manifest + checksums —
 * then publishes atomically via [File.renameTo] (#13, "cache bridge"). Does not change `LLMRepository`.
 */
object ModelPackageInstaller {
    data class Installed(val repoDir: File, val sanitizedRepoId: String)

    /**
     * @param stagedPackageDir a verified package tree in #14 Hub layout (variants/, shared/, manifest).
     * @param cacheDir the app cache root.
     * @param repoId the HF repo id (sanitized internally).
     * @param variantId the selected variant to materialize.
     */
    fun install(
        stagedPackageDir: File,
        cacheDir: File,
        repoId: String,
        variantId: String,
    ): Installed {
        val sanitized = PackageFormat.sanitizeRepoId(repoId)
        val stagingRoot = File(cacheDir, ".staging/$sanitized").apply { deleteRecursively(); mkdirs() }
        val variantRoot = File(stagedPackageDir, "variants/$variantId")

        for (stage in PackageFormat.VARIANT_SUBDIRS) {
            val src = File(variantRoot, stage)
            if (src.isDirectory) src.copyRecursively(File(stagingRoot, stage), overwrite = true)
        }
        val tokenizer = File(stagedPackageDir, "shared/tokenizer")
        if (tokenizer.isDirectory) tokenizer.copyRecursively(File(stagingRoot, "tokenizer"), overwrite = true)

        for (name in listOf(PackageFormat.MANIFEST_FILENAME, "variants/$variantId/checksums.json")) {
            val src = File(stagedPackageDir, name)
            if (src.isFile) src.copyTo(File(stagingRoot, File(name).name), overwrite = true)
        }

        val target = File(cacheDir, sanitized)
        if (target.exists()) target.deleteRecursively()
        if (!stagingRoot.renameTo(target)) {
            // Fallback for cross-mount rename failure: copy + clean.
            stagingRoot.copyRecursively(target, overwrite = true)
            stagingRoot.deleteRecursively()
        }
        return Installed(target, sanitized)
    }
}

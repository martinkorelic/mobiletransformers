package com.martinkorelic.mobiletransformers.packages

import com.martinkorelic.mobiletransformers.MissingArtifactException
import java.io.File

/**
 * Materializes a downloaded/staged package into the conventional cache layout `LLMRepository` already
 * probes — `<cacheDir>/<sanitizedRepoId>/{train,inference,embedding,tokenizer}` + manifest + checksums —
 * then publishes via a rename-aside → rename-in → delete-old sequence so a failed or interrupted
 * install never destroys the package already on disk (#13 "cache bridge" / #21). Does not change
 * `LLMRepository`.
 */
object ModelPackageInstaller {
    data class Installed(val repoDir: File, val sanitizedRepoId: String)

    /**
     * @param stagedPackageDir a verified package tree in #14 Hub layout (variants/, shared/, manifest).
     * @param cacheDir the app cache root.
     * @param repoId the HF repo id (sanitized internally).
     * @param variantId the selected variant to materialize.
     * @param consumeSource move the staged files instead of copying them, destroying
     *   [stagedPackageDir] in the process. Only pass `true` when the caller OWNS a throwaway staging
     *   tree — [com.martinkorelic.mobiletransformers.hub.HubDownloader] does, and for a real package
     *   the copy it avoids is over a gigabyte. Defaults to `false` because the staged tree is not
     *   generally the installer's to destroy: the JVM suite installs the same checked-in
     *   `tiny_package` fixture twice, and a hand-provisioned directory is a legitimate source.
     */
    @JvmOverloads
    fun install(
        stagedPackageDir: File,
        cacheDir: File,
        repoId: String,
        variantId: String,
        consumeSource: Boolean = false,
    ): Installed {
        val sanitized = PackageFormat.sanitizeRepoId(repoId)
        val stagingRoot = File(cacheDir, ".staging/$sanitized").apply { deleteRecursively(); mkdirs() }
        val variantRoot = File(stagedPackageDir, "variants/$variantId")

        for (stage in PackageFormat.VARIANT_SUBDIRS) {
            val src = File(variantRoot, stage)
            if (src.isDirectory) materialize(src, File(stagingRoot, stage), consumeSource)
        }
        val tokenizer = File(stagedPackageDir, "shared/tokenizer")
        if (tokenizer.isDirectory) materialize(tokenizer, File(stagingRoot, "tokenizer"), consumeSource)

        for (name in listOf(PackageFormat.MANIFEST_FILENAME, "variants/$variantId/checksums.json")) {
            val src = File(stagedPackageDir, name)
            if (src.isFile) src.copyTo(File(stagingRoot, File(name).name), overwrite = true)
        }

        // #21 crash safety: rename the OLD install aside first, put the new one in place, and only
        // then delete the old. The previous order deleted the live package before the new tree
        // existed, so a kill / disk-full / failed rename between the two left the user with no model
        // at all — including any locally trained train/checkpoint + training_state.json.
        val target = File(cacheDir, sanitized)
        val retired = File(cacheDir, ".retired-$sanitized-${System.nanoTime()}")
        val hadPrevious = target.exists() && target.renameTo(retired)
        if (target.exists() && !hadPrevious) {
            // Could not move the old tree aside; do not destroy it — fail with it still intact.
            stagingRoot.deleteRecursively()
            throw MissingArtifactException(
                "cannot install $repoId: the existing package at ${target.path} could not be moved aside",
            )
        }

        val published =
            stagingRoot.renameTo(target) ||
                runCatching {
                    // Fallback for cross-mount rename failure: copy + clean.
                    stagingRoot.copyRecursively(target, overwrite = true)
                    stagingRoot.deleteRecursively()
                    true
                }.getOrDefault(false)

        if (!published) {
            // Roll the previous install back so a failed update is a no-op, not data loss.
            target.deleteRecursively()
            if (hadPrevious) retired.renameTo(target)
            throw MissingArtifactException("failed to publish the package for $repoId at ${target.path}")
        }

        if (hadPrevious) retired.deleteRecursively()
        return Installed(target, sanitized)
    }

    /**
     * Put [src]'s contents at [dest], moving when [consume] allows it and copying otherwise.
     *
     * When the caller owns the staged tree this is a directory-entry update rather than bytes copied.
     * The unconditional `copyRecursively` it replaces wrote a real package a second time, which put
     * peak usage at roughly **three** simultaneous copies of a 1.3 GB model on a phone (download
     * staging + install staging + the published tree) and made install time scale with model size for
     * no reason.
     *
     * The copy stays as the fallback even when [consume] is set: a rename genuinely can fail when the
     * staged tree is on a different mount, and the caller — not this function — chooses where staging
     * lives.
     */
    private fun materialize(src: File, dest: File, consume: Boolean) {
        dest.parentFile?.mkdirs()
        if (consume && src.renameTo(dest)) return
        src.copyRecursively(dest, overwrite = true)
    }
}

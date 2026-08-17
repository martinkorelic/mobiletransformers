package com.martinkorelic.mobiletransformers.packages

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.File

/**
 * What the cache knows about *how* an installed package got there.
 *
 * ### Why this exists
 *
 * The cache directory name is `sanitizeRepoId(repoId)`, and nothing recorded the `repoId` that
 * produced it. [CacheIndex] therefore had only two candidates to offer a caller that wanted to load an
 * installed package: the directory name, or the manifest's [MobileTransformersManifest.baseModelId] —
 * and **`baseModelId` is a different thing**. It names the upstream model the package was exported
 * *from* (`google/functiongemma-270m-it`), not the repo it was pulled *from*
 * (`mobiletransformers/functiongemma-270m-it`).
 *
 * The showcase app's Models screen picked `baseModelId`, so tapping Load on a package that was
 * visibly installed sanitized to a *different* directory, found nothing there, tried to pull the base
 * model from the Hub, and reported `ModelNotInstalledException` — "not installed at
 * /data/user/0/…/google__functiongemma-270m-it" for a package sitting one directory over. There was no
 * way to get the right answer from what the cache stored, which is what this file changes.
 *
 * Written **inside the staging tree before publish**, so it lands atomically with the package it
 * describes: there is no window in which an installed tree has no record, and a rolled-back install
 * takes its record with it.
 */
data class InstallRecord(
    /** The repo id the package was installed from — the argument to `fromPretrained`. */
    @SerializedName("repoId") val repoId: String = "",
    /** The variant materialized into the flat cache layout. */
    @SerializedName("variantId") val variantId: String = "",
    /** Feature groups requested at download time (`inference`, `train`, `rag`, `genai`). */
    @SerializedName("features") val features: List<String> = emptyList(),
    @SerializedName("installedAtEpochMs") val installedAtEpochMs: Long = 0L,
) {
    companion object {
        const val FILENAME = "install_record.json"

        private val gson = Gson()

        fun write(dir: File, record: InstallRecord) {
            File(dir, FILENAME).writeText(gson.toJson(record), Charsets.UTF_8)
        }

        /** The record in [dir], or `null` for a legacy or hand-pushed package that has none. */
        fun read(dir: File): InstallRecord? {
            val file = File(dir, FILENAME)
            if (!file.isFile) return null
            return runCatching { gson.fromJson(file.readText(Charsets.UTF_8), InstallRecord::class.java) }
                .getOrNull()
                ?.takeIf { it.repoId.isNotBlank() }
        }

        /**
         * Best-effort inverse of [PackageFormat.sanitizeRepoId], for a package installed before this
         * record existed or pushed by hand with `scripts/device_package.sh`.
         *
         * Only `'/' -> "__"` is reversible: every other unsafe character collapses to a single `'_'`,
         * which is lossy by design. That is enough for a Hub id, whose owner and name are both drawn
         * from `[A-Za-z0-9._-]`. It is a fallback, not the source of truth — [read] is, and a package
         * installed by this SDK always has one.
         */
        fun unsanitize(sanitizedRepoId: String): String = sanitizedRepoId.replace("__", "/")
    }
}

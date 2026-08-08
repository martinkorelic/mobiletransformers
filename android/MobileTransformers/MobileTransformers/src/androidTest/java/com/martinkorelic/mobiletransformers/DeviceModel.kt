package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assume.assumeTrue

/**
 * Shared locator for the device (instrumented) suites. Probes the cache dirs `make device-package` pushes
 * a real #9 package into, and `assumeTrue`-skips (never fails) when none is present — so a device-less or
 * un-provisioned run stays green (Android device behaviour is manual per the test taxonomy). Mirrors the
 * `GenAISpikeTest` candidate-dir pattern.
 */
object DeviceModel {
    private const val LOG_TAG = "DeviceModel"
    const val CACHE_SUBDIR = "mt_pkg"

    /**
     * The cache root containing `<sanitizedRepoId>/{inference,train,embedding,tokenizer}`, or null.
     *
     * The external files dir is probed first because that is where `scripts/device_package.sh` pushes:
     * `/data/local/tmp` is SELinux-labelled `shell_data_file`, so the app domain can neither list it on
     * a modern Android nor write into it (which the merge/checkpoint legs need). It stays in the list
     * for a manually-provisioned run on an older device.
     */
    fun cacheRoot(): File? = candidates().firstOrNull { root -> installedPackages(root).isNotEmpty() }

    private fun candidates(): List<File> {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val external = ctx.getExternalFilesDir(null)
        return listOfNotNull(
            external?.let { File(it, CACHE_SUBDIR) },
            File(ctx.filesDir, CACHE_SUBDIR),
            File("/data/local/tmp/$CACHE_SUBDIR"),
        )
    }

    /**
     * Skip the test unless a package is present; returns the cache root when it is.
     *
     * The skip message names every candidate and what was wrong with it. A bare "no package" told you
     * nothing about *why* — and since every device suite skips rather than fails, an un-diagnosable skip
     * reads exactly like a pass in CI output.
     */
    fun requireCacheRoot(): File {
        val root = cacheRoot()
        if (root == null) {
            val report = candidates().joinToString("; ") { "$it -> ${describe(it)}" }
            Log.w(LOG_TAG, "no device package found. Candidates: $report")
            assumeTrue(
                "no device package under '$CACHE_SUBDIR' (run `make device-package`). Candidates: $report",
                false,
            )
        }
        return root!!
    }

    /** Why a candidate root was rejected, for the skip message. */
    private fun describe(root: File): String = when {
        !root.exists() -> "absent"
        !root.isDirectory -> "not a directory"
        root.listFiles() == null -> "unreadable (permission/SELinux)"
        installedPackages(root).isEmpty() -> {
            val children = root.list()?.joinToString(",") { name ->
                val child = File(root, name)
                val inf = File(child, "inference")
                "$name[dir=${child.isDirectory},read=${child.canRead()}," +
                    "inference=${inf.exists()}/${inf.isDirectory}]"
            }
            "no package dir with inference/ (children: ${children ?: "?"})"
        }
        else -> "ok"
    }

    /**
     * The (already-sanitized) repo ids installed under [root], newest first.
     *
     * Only directories carrying an `inference/` subtree count — a bare directory (a half-finished push,
     * or the vector store the RAG leg writes) is not a model package, and picking one would make the
     * suites fail with an unrelated error instead of skipping.
     */
    private fun installedPackages(root: File): List<File> =
        root.listFiles { f: File -> f.isDirectory && File(f, "inference").isDirectory }
            ?.sortedByDescending { it.lastModified() }
            .orEmpty()

    /** The (already-sanitized) repo id of the installed package under the cache root. */
    fun repoId(root: File): String = installedPackages(root).first().name

    /** True iff the installed package carries a train-capable subtree (for the train→merge→generate legs). */
    fun hasTraining(root: File, repoId: String): Boolean =
        File(root, "$repoId/train/training_config.json").isFile
}

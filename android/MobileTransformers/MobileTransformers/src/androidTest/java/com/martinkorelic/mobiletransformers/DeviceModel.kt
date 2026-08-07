package com.martinkorelic.mobiletransformers

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
    const val CACHE_SUBDIR = "mt_pkg"

    /** The cache root containing `<sanitizedRepoId>/{inference,train,embedding,tokenizer}`, or null. */
    fun cacheRoot(): File? {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        return listOf(
            File(ctx.filesDir, CACHE_SUBDIR),
            File("/data/local/tmp/$CACHE_SUBDIR"),
            File(ctx.getExternalFilesDir(null), CACHE_SUBDIR),
        ).firstOrNull { root -> root.isDirectory && root.listFiles()?.any { it.isDirectory } == true }
    }

    /** Skip the test unless a package is present; returns the cache root when it is. */
    fun requireCacheRoot(): File {
        val root = cacheRoot()
        assumeTrue("no device package under '$CACHE_SUBDIR' (run `make device-package`)", root != null)
        return root!!
    }

    /** The (already-sanitized) repo id = the first installed package dir under the cache root. */
    fun repoId(root: File): String = root.listFiles { f -> f.isDirectory }!!.first().name

    /** True iff the installed package carries a train-capable subtree (for the train→merge→generate legs). */
    fun hasTraining(root: File, repoId: String): Boolean =
        File(root, "$repoId/train/training_config.json").isFile
}

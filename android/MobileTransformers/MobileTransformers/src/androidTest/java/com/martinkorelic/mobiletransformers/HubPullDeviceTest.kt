package com.martinkorelic.mobiletransformers

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeFalse
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #21's missing device leg: pull a package **from the Hugging Face Hub onto the phone** and load it.
 *
 * ### Why this did not exist
 *
 * Every other device suite runs against a package `adb push`ed by `scripts/device_package.sh`, so the
 * entire download half — [com.martinkorelic.mobiletransformers.hub.HubResolver],
 * `DownloadPlanner`, `PackageDownloader`, `PackageDownloadWorker`, `ModelPackageInstaller` — had only
 * ever been exercised on the JVM against MockWebServer. That is localhost, in a process with no Android
 * permission model, which is why nothing caught that the library manifest did not declare
 * `android.permission.INTERNET`: the code could not have worked on any device, and no test could see it.
 *
 * This test is the one that would have. It calls the same entry point an integrator calls, with **no
 * package pre-installed**, so the pull is what is under test rather than an afterthought.
 *
 * ### Running it
 *
 * ```
 * make device-hub-test REPO=<org>/<name>
 * ```
 *
 * It `assumeTrue`-skips without `mtHubRepoId`, matching [DeviceModel]'s skip-don't-fail convention so
 * `make device-test` stays green without one. A repo id that is present but wrong must **fail**, not
 * skip — a test that skips on a bad id is indistinguishable from no test at all.
 */
@RunWith(AndroidJUnit4::class)
class HubPullDeviceTest {

    private companion object {
        const val LOG_TAG = "HubPullDeviceTest"

        /** Instrumentation args: `-e mtHubRepoId <id>` (+ optional `-e mtHubToken <token>`). */
        const val ARG_REPO_ID = "mtHubRepoId"
        const val ARG_TOKEN = "mtHubToken"
    }

    /**
     * A cache root of this test's OWN, never the `mt_pkg` one the other suites read.
     *
     * Sharing it would let a Hub pull overwrite the pushed package that `PristinePackageRule` exists to
     * keep pristine, and a 1.3 GB download landing on top of the suite's fixture would be a
     * spectacularly confusing way to fail an unrelated test.
     */
    private val cacheRoot: File
        get() = File(
            InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null),
            "mt_hub_pull",
        )

    @After
    fun removeTheDownloadedPackage() {
        // A real package is over a gigabyte. Leaving it behind would starve every later suite of the
        // free space `scripts/device_package.sh` checks for before it will push anything.
        cacheRoot.deleteRecursively()
    }

    @Test
    fun pullsAPackageFromTheHubInstallsItAndGenerates() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val repoId = args.getString(ARG_REPO_ID)
        assumeFalse(
            "no '$ARG_REPO_ID' instrumentation argument — run `make device-hub-test REPO=<org>/<name>` " +
                "to exercise the Hub pull. Skipping is correct here: the package under test lives on " +
                "the network, not in this checkout.",
            repoId.isNullOrBlank(),
        )
        val token = args.getString(ARG_TOKEN)?.takeIf { it.isNotBlank() }

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        cacheRoot.deleteRecursively()
        cacheRoot.mkdirs()

        // Nothing is installed: this is the cold-start path a new user is on, and the only one that
        // exercises the download at all.
        val modelDir = File(cacheRoot, PackageFormat.sanitizeRepoId(repoId!!))
        assertTrue("precondition: the cache must start empty", !modelDir.exists())

        var lastLoggedDecile = -1
        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = cacheRoot.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Training),
            hubConfig = token?.let { HubConfig(token = it) },
            onDownloadProgress = { p ->
                // One line per 10%: a 200-file package would otherwise bury the logcat this test is
                // read from, and the per-file path is what makes a stall diagnosable.
                val decile = if (p.filesTotal > 0) p.filesDone * 10 / p.filesTotal else 0
                if (decile > lastLoggedDecile) {
                    lastLoggedDecile = decile
                    Log.i(LOG_TAG, "downloaded ${p.filesDone}/${p.filesTotal} — ${p.path}")
                }
            },
        )

        try {
            // 1. The installer produced the layout `LLMRepository` probes, not just "some files".
            for (expected in listOf("inference", "train", "tokenizer")) {
                assertTrue(
                    "installed package has no $expected/ under $modelDir " +
                        "(present: ${modelDir.list()?.joinToString()})",
                    File(modelDir, expected).isDirectory,
                )
            }
            assertTrue(
                "installed package has no manifest — the installer copies it, and its absence means " +
                    "variant selection and capability reporting have nothing to read",
                File(modelDir, PackageFormat.MANIFEST_FILENAME).isFile,
            )

            // 2. The staging trees are GONE. Both are full copies of the package, so a leak here is
            //    over a gigabyte of dead weight per pull — it used to survive until the next pull of
            //    the same repo.
            val sanitized = PackageFormat.sanitizeRepoId(repoId)
            assertTrue(
                "download staging .download/$sanitized survived the install",
                !File(cacheRoot, ".download/$sanitized").exists(),
            )
            assertTrue(
                "install staging .staging/$sanitized survived the install",
                !File(cacheRoot, ".staging/$sanitized").exists(),
            )

            // 3. The features that were REQUESTED are the ones reported — a train group that silently
            //    did not download would otherwise only surface when training failed much later.
            assertTrue(
                "pulled package does not report Inference (features: " +
                    "${model.capabilities.availableFeatures})",
                ModelFeature.Inference in model.capabilities.availableFeatures,
            )
            assertTrue(
                "pulled package does not report training support although the train feature was " +
                    "requested (features: ${model.capabilities.availableFeatures})",
                model.capabilities.supportsTraining,
            )

            // 4. It actually runs. Everything above is structural; this is the one that proves the
            //    downloaded bytes are a working model and not a well-shaped directory.
            val result = model.generate("Hello", GenerationConfig(maxNewTokens = 4))
            Log.i(LOG_TAG, "generated ${result.tokenCount} tokens: ${result.text.take(120)}")
            assertTrue("generation produced no tokens from the pulled package", result.tokenCount > 0)
        } finally {
            model.close()
        }
    }
}

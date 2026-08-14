package com.martinkorelic.mobiletransformers

import android.util.Log
import java.io.File
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement

/**
 * Restores the on-device package to its pre-test bytes after every test in a class that trains or
 * merges. **Without this the device suite cannot be run in one pass.**
 *
 * The failure it exists to prevent, measured on the S21 FE 2026-08-14 (23 tests, 3 failures):
 *
 * * `TrainMergeGenerateTest` fingerprints the per-tensor weight blobs before and after a merge and
 *   asserts the bytes moved. The merge rewrites those files **in place**, so once any earlier test has
 *   merged,
 *   a second merge of the same adapter re-computes identical bytes and the assertion fails —
 *   correctly reporting "merge wrote no new weights" about a merge that worked fine.
 * * `TrainConvergenceTest` asserts the model still starts from *pretrained* weights, and reported
 *   `NaN` for both initial losses. **The cause is not the weights** — that was the first guess and it
 *   was wrong. It is `training_state.json`: the file does not exist in a freshly exported package,
 *   training CREATES it, and its presence makes the next run RESUME instead of starting fresh. Every
 *   one of these passes against a freshly pushed package.
 *
 * JUnit orders classes by name, which puts the mutating classes ahead of the classes that need a
 * clean fixture, so the contamination is deterministic rather than flaky — it will reproduce on every
 * suite run until the fixture is restored.
 *
 * **What is captured:** the training checkpoint, `training_state.json`, and every per-tensor weight
 * blob plus its `.sha256` sidecar under the inference stage. Those are the only mutable artifacts;
 * the ONNX graphs, tokenizer and manifest are read-only at runtime.
 *
 * **Restoring means both halves.** A file that was ABSENT before the test and exists after it is
 * restored by *deleting* it — see [restore]. The first version of this rule only copied saved files
 * back, which fixed `TrainMergeGenerateTest` (whose blobs pre-exist) and left `TrainConvergenceTest`
 * failing identically, because the file that broke it was one the test created. State is what is
 * there **and** what is not.
 *
 * ## What it actually guarantees — the name overstates it
 *
 * The rule restores each test's **pre-test** state, which equals *pristine* only if the suite began
 * pristine. So the precondition stands: push a fresh package before a suite run. What the rule adds is
 * that the state no longer degrades *during* the run, which is what was broken.
 *
 * Two consequences worth knowing:
 *
 * * because every test restores, a suite that starts clean also **ends** clean — so a second run in a
 *   row is valid without re-pushing, which was not true before;
 * * if a test dies without reaching its `finally` (native crash, OOM kill, `adb` disconnect), the
 *   package is left dirty and its clean copy is orphaned in the stash. The next run detects the
 *   orphan and recovers from it before capturing. Without that, the rule would adopt the corruption
 *   as its baseline and preserve it faithfully for every following test — guaranteeing contamination
 *   instead of preventing it.
 *
 * **This is a fixture rule, not a correctness guarantee.** It cannot make a test independent of one
 * that mutates state it does not know about, and a test whose assertion depends on a clean package
 * should still say so in its failure message — the messages above are what made this diagnosable at
 * all.
 *
 * *(Editing note, learned twice in one day: Kotlin block comments NEST. Writing a glob for the weight
 * blobs as `inference` + slash + star inside this KDoc opens a nested comment that swallows the
 * closing delimiter, and the compiler reports "Unclosed comment" at the END of the file, pointing
 * nowhere near the cause. Name the files in prose instead.)*
 */
class PristinePackageRule : TestRule {

    override fun apply(base: Statement, description: Description): Statement = object : Statement() {
        override fun evaluate() {
            val root = DeviceModel.cacheRoot()
            if (root == null) {
                // No package: the test itself will assumeTrue-skip. Nothing to protect.
                base.evaluate()
                return
            }
            val repoId = DeviceModel.repoId(root)
            val stash = File(root, "$repoId/.pristine_stash")
            val mutable = mutableArtifacts(root, repoId)

            // A stash that already exists means a previous test never reached its `finally` — the
            // process died mid-test (native crash, OOM kill, `adb` disconnect). The package is dirty
            // AND its clean copy is sitting right there. Recover from it before capturing, otherwise
            // this run adopts the corruption as its baseline and preserves it faithfully for every
            // test that follows: the rule would then guarantee contamination rather than prevent it.
            if (stash.isDirectory) {
                val recovered = restore(stash, root, repoId, mutable)
                Log.w(
                    LOG_TAG,
                    "found an orphaned stash from a previous run (a test died before restoring); " +
                        "recovered $recovered artifacts before capturing",
                )
            }

            stash.deleteRecursively()
            stash.mkdirs()
            for (f in mutable) {
                if (f.exists()) f.copyRecursively(File(stash, f.name), overwrite = true)
            }
            Log.i(LOG_TAG, "captured ${mutable.count { it.exists() }} mutable artifacts for ${description.methodName}")

            try {
                base.evaluate()
            } finally {
                // Restore even when the test failed — a failing test that leaves the package dirty
                // turns one real failure into a cascade of misleading ones in the tests after it.
                val restored = restore(stash, root, repoId, mutable)
                stash.deleteRecursively()
                Log.i(LOG_TAG, "restored $restored artifacts after ${description.methodName}")
            }
        }
    }

    private companion object {
        const val LOG_TAG = "PristinePackageRule"

        /**
         * Put the artifacts back exactly as they were — including the ones that were **absent**.
         *
         * The absent half is not an edge case, it is the common one: `training_state.json` does not
         * exist in a freshly exported package, training CREATES it, and a restore that only copies
         * saved files back leaves it behind forever. `TrainConvergenceTest` then resumes from a
         * trained state instead of starting from pretrained weights and reports `NaN` initial losses
         * — which is exactly how this rule failed its first suite run, having fixed
         * `TrainMergeGenerateTest` but not this.
         *
         * The list is recomputed here rather than reusing the capture-time one, so a weight blob the
         * merge newly created is also seen. Returns restored + deleted.
         */
        fun restore(stash: File, root: File, repoId: String, capturedList: List<File>): Int {
            val now = (mutableArtifacts(root, repoId) + capturedList).distinctBy { it.absolutePath }
            var touched = 0
            for (f in now) {
                val saved = File(stash, f.name)
                if (saved.exists()) {
                    if (f.exists()) f.deleteRecursively()
                    saved.copyRecursively(f, overwrite = true)
                    touched++
                } else if (f.exists()) {
                    // Absent at capture, present now => the test created it. Removing it IS the restore.
                    f.deleteRecursively()
                    touched++
                }
            }
            return touched
        }

        /** Everything a training or merge run can rewrite. Read-only artifacts are deliberately absent. */
        fun mutableArtifacts(root: File, repoId: String): List<File> {
            val trainDir = File(root, "$repoId/train")
            val inferenceDir = File(root, "$repoId/inference")
            val weights = inferenceDir
                .listFiles { f -> f.name.endsWith(".bin") || f.name.endsWith(".sha256") }
                .orEmpty()
                .toList()
            return listOf(
                File(trainDir, "checkpoint"),
                File(trainDir, "training_state.json"),
            ) + weights
        }
    }
}

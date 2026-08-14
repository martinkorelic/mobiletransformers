package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #18 + #19 device checkpoint: baseline generate → train(maxSteps=1) → merge() → generate; assert the
 * merged output diverges from the pre-train baseline. Requires a train-capable package (`train/`).
 */
@RunWith(AndroidJUnit4::class)
class TrainMergeGenerateTest {

    /**
     * Restore the package after every test in this class. It trains and/or merges, which rewrites the
     * checkpoint and the `inference/` weight blobs in place — see [PristinePackageRule] for the three
     * suite failures this prevents.
     */
    @get:Rule
    val pristinePackage = PristinePackageRule()

    @Test
    // Explicit `: Unit`. With expression-body syntax the return type is inferred from the block's
    // last expression, so ending it with something non-Unit (a `Log.i`, which returns Int) silently
    // makes the method non-void and JUnit rejects the whole class with
    // "Method ... should be void" — surfacing as a runner-instantiation failure, not a test failure.
    fun trainMergeGenerateDivergesFromBaseline(): Unit = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext

        // The package ships model artifacts, not training data, so the caller supplies both the dataset
        // and the preprocessor that parses it (`DatasetConfig.task`). ORTDataCurator reads
        // `<cacheDir>/<repo>/train/<trainFile>.jsonl`; `cola` is the simplest supported schema.
        val trainFile = "mt_device_test_cola"
        File(root, "$repoId/train/$trainFile.jsonl").writeText(
            (1..8).joinToString("\n") { i ->
                """{"sentence": "The cat sat on the mat number $i.", "label": ${i % 2}}"""
            } + "\n",
        )

        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Training),
        )
        try {
            val gen = GenerationConfig(maxNewTokens = 8, loadMerged = true)
            val baseline = model.generate("The capital of France is", gen).text

            // Fingerprint the trainable tensors BEFORE training. The merge writes new weights into
            // these per-tensor `.bin` files in place (#9), so their bytes changing is the direct
            // evidence that train -> merge -> handoff actually moved weights.
            //
            // Asserting only that the generated text changes does NOT test that: one LoRA step on q/k
            // at lr 1e-4 legitimately leaves 8 greedy tokens identical, so a text-only assertion fails
            // on a good merge and would equally pass if the merge silently wrote nothing.
            val binNames = ArrayList<String>()
            val beforeHashes = HashMap<String, Int>()
            val listed = File(root, repoId + "/inference").listFiles()
            if (listed != null) {
                for (f in listed) {
                    if (f.name.endsWith(".MatMul.weight.bin")) {
                        binNames.add(f.name)
                        beforeHashes[f.name] = f.readBytes().contentHashCode()
                    }
                }
            }
            assertTrue("package has no per-tensor trainable .bin files", binNames.size > 0)

            model.train(
                DatasetConfig(
                    trainFile = trainFile,
                    task = "cola",
                    maxSequenceLength = 64,
                    maxDatasetLength = 8,
                    datasetBatchSize = 4,
                ),
                TrainConfig(maxSteps = 1, batchSize = 2, mergeAtEnd = true),
            )
            model.merge()

            var changed = 0
            for (name in binNames) {
                val h = File(root, repoId + "/inference/" + name).readBytes().contentHashCode()
                if (h != beforeHashes[name]) changed++
            }
            assertTrue(
                "merge wrote no new weights: all " + binNames.size + " trainable .bin files are " +
                    "unchanged. The merge rewrites these files IN PLACE, so a package that has " +
                    "already been merged re-merges to identical bytes — this test therefore needs a " +
                    "package no earlier test has merged. PristinePackageRule restores it around every " +
                    "mutating class, so if you are seeing this, either the rule is not applied to a " +
                    "class that merges, or the merge genuinely wrote nothing (check logcat for " +
                    "'Starting weight merging process').",
                changed > 0,
            )

            // Generation must still work off the merged weights. The text is reported rather than
            // asserted *different*: after a single step the argmax may legitimately be unchanged.
            val after = model.generate("The capital of France is", gen).text

            // `after.isNotEmpty()` was the whole assertion here until 2026-08-14, and it is far too
            // weak: a model whose weights the merge had destroyed emitted 48 consecutive newlines,
            // which is non-empty. That is not a hypothetical — the merge was writing every weight
            // TRANSPOSED for months and this test passed throughout.
            //
            // Still deliberately behavioural rather than exact: this suite trains ONE step, so the
            // output legitimately varies. What a corrupted model does is degenerate — a single
            // character or token repeated, or pure whitespace — so that is what is excluded.
            // `PostMergeNumericsTest` owns the numerical assertion; this one owns "did the text
            // survive at all".
            assertTrue("generation returned nothing after merge", after.isNotEmpty())
            assertTrue(
                "generation after merge produced only whitespace (${after.length} chars) — the classic " +
                    "signature of a model whose merged weights are corrupt. Raw: <$after>",
                after.isNotBlank(),
            )
            val distinctNonSpace = after.filterNot { it.isWhitespace() }.toSet().size
            assertTrue(
                "generation after merge produced $distinctNonSpace distinct non-whitespace character(s) " +
                    "in ${after.length} chars — a degenerate repeated token, not text. This is what a " +
                    "corrupted merge looks like, and what `isNotEmpty()` used to accept. Raw: <$after>",
                distinctNonSpace >= 2,
            )
            android.util.Log.i(
                "TrainMergeGenerateTest",
                "merged " + changed + "/" + binNames.size + " tensors; baseline=<" + baseline + "> after=<" + after + ">",
            )
        } finally {
            model.close()
        }
    }
}

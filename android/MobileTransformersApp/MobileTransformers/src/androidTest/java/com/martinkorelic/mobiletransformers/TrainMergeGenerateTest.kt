package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.DatasetConfig
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #18 + #19 device checkpoint: baseline generate → train(maxSteps=1) → merge() → generate; assert the
 * merged output diverges from the pre-train baseline. Requires a train-capable package (`train/`).
 */
@RunWith(AndroidJUnit4::class)
class TrainMergeGenerateTest {

    @Test
    fun trainMergeGenerateDivergesFromBaseline() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        assumeTrue("package is not train-capable (no train/ stage)", DeviceModel.hasTraining(root, repoId))

        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = MobileTransformers.fromPretrained(
            context = ctx,
            repoId = repoId,
            cacheDir = root.absolutePath,
            features = setOf(ModelFeature.Inference, ModelFeature.Training),
        )
        try {
            val gen = GenerationConfig(maxNewTokens = 8, loadMerged = true)
            val baseline = model.generate("The capital of France is", gen).text

            model.train(DatasetConfig(maxSequenceLength = 64), TrainConfig(maxSteps = 1, mergeAtEnd = true))
            model.merge()

            val after = model.generate("The capital of France is", gen).text
            assertTrue("merged output did not diverge from baseline", after != baseline)
        } finally {
            model.close()
        }
    }
}

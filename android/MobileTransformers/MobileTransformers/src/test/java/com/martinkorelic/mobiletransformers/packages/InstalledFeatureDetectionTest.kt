package com.martinkorelic.mobiletransformers.packages

import com.martinkorelic.mobiletransformers.MobileTransformers
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * Which feature groups an installed package reports — asserted from a real directory layout.
 *
 * ### The defect this exists for
 *
 * `Inference` was detected from `LLMRepository.isGenerationAvailable`, which is set by the presence
 * of `inference/generation_config.json`. That file is the model's own **generation** config: only a
 * model with a generative head has one. So an encoder — DistilBERT SST-2, the MiniLM embedder —
 * shipped a complete `inference/` stage and reported the Inference group as *not installed*, and
 * `fromPretrained` refused the package with
 *
 *     Feature 'Inference' is not installed for this package. Installed features: Training.
 *
 * Installing DistilBERT from the catalog therefore failed at the last step of a 270 MB download,
 * naming the one group the package definitely had. `classify()` runs off exactly that stage.
 *
 * ### Why the layouts are built on disk rather than mocked
 *
 * The bug was never in the rule — `if (available) features += Inference` was always right. It was in
 * what "available" was read from, i.e. in the step from *files on disk* to a boolean. A test that
 * passes the booleans in asserts the half that was never broken, so both halves are covered here:
 * the layouts below are what the installer actually writes.
 */
@RunWith(RobolectricTestRunner::class)
class InstalledFeatureDetectionTest {

    private val cacheDir: File = Files.createTempDirectory("installed-features").toFile()

    @After
    fun cleanup() {
        cacheDir.deleteRecursively()
    }

    private fun repositoryFor(repoId: String): LLMRepository =
        LLMRepository(RuntimeEnvironment.getApplication(), cacheDir.absolutePath, initialModel = repoId)

    private fun stage(repoId: String, stage: String): File =
        File(File(cacheDir, repoId), stage).apply { mkdirs() }

    /** The inference stage every package has: a graph, and the exporter's task side-car beside it. */
    private fun writeInferenceStage(repoId: String, generative: Boolean) {
        val inference = stage(repoId, "inference")
        File(inference, "model.onnx").writeText("not a real graph")
        File(inference, PackageTask.FILENAME).writeText(
            if (generative) {
                """{"task":"text-generation-with-past","modelType":"llama"}"""
            } else {
                """{"task":"text-classification","modelType":"distilbert","id2label":{"0":"NEGATIVE","1":"POSITIVE"}}"""
            },
        )
        // The distinguishing file, and the whole bug: an encoder has no HF generation config.
        if (generative) {
            File(inference, "generation_config.json").writeText("""{"eos_token_id":2}""")
        }
    }

    private fun writeTrainStage(repoId: String) {
        File(stage(repoId, "train"), "training_config.json").writeText("""{"taskName":"text-classification"}""")
    }

    private fun writeEmbeddingStage(repoId: String) {
        File(stage(repoId, "embedding"), "rag_config.json").writeText("""{"embeddingDimension":384}""")
    }

    // --- the seam that broke: files on disk -> availability flags --------------------------------

    @Test
    fun aClassifierPackageReportsAnInstalledInferenceStage() {
        writeInferenceStage("mobiletransformers_distilbert-sst2-english", generative = false)
        writeTrainStage("mobiletransformers_distilbert-sst2-english")

        val repo = repositoryFor("mobiletransformers_distilbert-sst2-english")

        assertTrue("the graph is on disk, so the inference group is installed", repo.isInferenceAvailable)
        assertFalse("an encoder has no HF generation config, and never will", repo.isGenerationAvailable)
        assertTrue(repo.isTrainingAvailable)
    }

    @Test
    fun aDecoderPackageReportsBothInferenceAndGeneration() {
        writeInferenceStage("mobiletransformers_SmolLM2-135M-Instruct", generative = true)

        val repo = repositoryFor("mobiletransformers_SmolLM2-135M-Instruct")

        assertTrue(repo.isInferenceAvailable)
        assertTrue(repo.isGenerationAvailable)
    }

    @Test
    fun aPackageWithNoGraphReportsNoInferenceStage() {
        stage("mobiletransformers_empty", "inference")
        val repo = repositoryFor("mobiletransformers_empty")
        assertFalse(repo.isInferenceAvailable)
    }

    // --- and the rule those flags feed --------------------------------------------------------

    @Test
    fun everyPackageShapeMapsToTheGroupsItCarries() {
        // A classifier/encoder: inference + train, no generation config anywhere.
        assertEquals(
            setOf(ModelFeature.Inference, ModelFeature.Training),
            MobileTransformers.detectFeatures(inference = true, training = true, rag = false),
        )
        // A decoder pulled with every group.
        assertEquals(
            setOf(ModelFeature.Inference, ModelFeature.Training, ModelFeature.Rag, ModelFeature.Embedding),
            MobileTransformers.detectFeatures(inference = true, training = true, rag = true),
        )
        // Inference-only: the default request, and the smallest useful install.
        assertEquals(
            setOf(ModelFeature.Inference),
            MobileTransformers.detectFeatures(inference = true, training = false, rag = false),
        )
        // Nothing on disk is what `fromPretrained` turns into MissingArtifactException.
        assertTrue(MobileTransformers.detectFeatures(inference = false, training = false, rag = false).isEmpty())
    }

    @Test
    fun ragAlwaysBringsEmbeddingWithIt() {
        writeInferenceStage("mobiletransformers_all-MiniLM-L6-v2", generative = false)
        writeEmbeddingStage("mobiletransformers_all-MiniLM-L6-v2")

        val repo = repositoryFor("mobiletransformers_all-MiniLM-L6-v2")

        assertTrue(repo.isRagAvailable)
        val groups = MobileTransformers.detectFeatures(
            inference = repo.isInferenceAvailable || repo.isGenerationAvailable,
            training = repo.isTrainingAvailable,
            rag = repo.isRagAvailable,
        )
        // The encoder that is useful on its own: retrieval AND a runnable inference stage.
        assertEquals(
            setOf(ModelFeature.Inference, ModelFeature.Rag, ModelFeature.Embedding),
            groups,
        )
    }
}

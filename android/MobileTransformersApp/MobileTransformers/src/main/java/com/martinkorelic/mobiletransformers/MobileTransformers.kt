package com.martinkorelic.mobiletransformers

import android.content.Context
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.internal.runtime.RepositoryBackedModelSession
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import java.io.File

/**
 * Stable, HF-style entry point to the MobileTransformers SDK (#17). [fromPretrained] returns a
 * [MobileTransformerModel] whose work is delegated to a [RepositoryBackedModelSession] wrapping the existing
 * repositories — no engine rewrite. Public types stay neutral; the `ORT*` names remain internal.
 */
object MobileTransformers {

    /**
     * Load a locally-installed package into a working [MobileTransformerModel].
     *
     * The remote pull/install path is #21; this foundation resolves an already-installed package in
     * [cacheDir] and models engine selection (Native default; GenAI is a *selectable* engine over the SAME
     * package, not a second download — see [ModelFeature]). Fails closed with [ModelNotInstalledException]
     * when the package is absent.
     */
    @JvmStatic
    @Suppress("UNUSED_PARAMETER") // revision/variant/hubConfig are the stable signature; used by #21's pull
    suspend fun fromPretrained(
        context: Context,
        repoId: String,
        cacheDir: String = context.filesDir.absolutePath,
        revision: String = "main",
        variant: String? = null,
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
        engine: InferenceEngine = InferenceEngine.NATIVE,
        hubConfig: HubConfig? = null,
    ): MobileTransformerModel {
        val sanitized = PackageFormat.sanitizeRepoId(repoId)
        val modelDir = File(cacheDir, sanitized)
        if (!modelDir.isDirectory) {
            throw ModelNotInstalledException(
                "package '$repoId' is not installed at $modelDir (pull it first — #21)",
            )
        }

        val repo = LLMRepository(context.applicationContext, cacheDir, initialModel = sanitized)
        if (!repo.isGenerationAvailable && !repo.isTrainingAvailable && !repo.isRagAvailable) {
            throw MissingArtifactException(
                "package '$repoId' has no usable train/inference/embedding config under $modelDir",
            )
        }

        // GenAI is a selectable engine over the one shared package; #11 owns real GenAI wiring + fallback.
        // Requesting the GenAI/ManualInference feature only sets the engine — it never downloads a 2nd group.
        val resolvedEngine =
            if (features.any { it == ModelFeature.GenAI }) InferenceEngine.GENAI else engine

        val capabilities =
            RuntimeCapabilities(
                engine = resolvedEngine,
                supportsTraining = repo.isTrainingAvailable,
                supportsMerge = repo.isTrainingAvailable,
                supportsRag = repo.isRagAvailable,
                supportsEmbedding = repo.isRagAvailable,
                availableFeatures = detectFeatures(repo),
            )

        val session =
            RepositoryBackedModelSession(
                repo = repo,
                capabilities = capabilities,
                inferencePackagePath = File(modelDir, "inference").absolutePath,
            )
        return MobileTransformerModel(session, capabilities)
    }

    private fun detectFeatures(repo: LLMRepository): Set<ModelFeature> {
        val features = mutableSetOf<ModelFeature>()
        if (repo.isGenerationAvailable) features += ModelFeature.Inference
        if (repo.isTrainingAvailable) features += ModelFeature.Training
        if (repo.isRagAvailable) {
            features += ModelFeature.Rag
            features += ModelFeature.Embedding
        }
        return features
    }
}

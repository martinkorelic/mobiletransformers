package com.martinkorelic.mobiletransformers

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import com.martinkorelic.mobiletransformers.config.HubConfig
import com.martinkorelic.mobiletransformers.hub.DownloadProgress
import com.martinkorelic.mobiletransformers.hub.DownloadProgressListener
import com.martinkorelic.mobiletransformers.hub.HubDownloader
import com.martinkorelic.mobiletransformers.hub.HubResolver
import com.martinkorelic.mobiletransformers.packages.CacheIndex
import com.martinkorelic.mobiletransformers.internal.runtime.RepositoryBackedModelSession
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.PackageTask
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.runtime.GenAiSupport
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities
import java.io.File
import com.martinkorelic.mobiletransformers.packages.PackagePaths

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
    suspend fun fromPretrained(
        context: Context,
        repoId: String,
        cacheDir: String = context.filesDir.absolutePath,
        revision: String = "main",
        variant: String? = null,
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
        engine: InferenceEngine = InferenceEngine.NATIVE,
        hubConfig: HubConfig? = null,
        onDownloadProgress: DownloadProgressListener? = null,
    ): MobileTransformerModel {
        val sanitized = PackageFormat.sanitizeRepoId(repoId)
        val modelDir = File(cacheDir, sanitized)
        if (!modelDir.isDirectory) {
            // #21: not installed -> manifest-first pull + atomic install, then load.
            val genaiRequested = engine == InferenceEngine.GENAI || features.any { it == ModelFeature.GenAI }
            val featureGroups = buildSet {
                add("inference")
                if (ModelFeature.Training in features) add("train")
                if (ModelFeature.Rag in features || ModelFeature.Embedding in features) add("rag")
            }
            try {
                HubDownloader.downloadAndInstall(
                    cacheDir = File(cacheDir),
                    repoId = repoId,
                    revision = revision,
                    variant = variant,
                    features = featureGroups,
                    genai = genaiRequested,
                    // #21: real device capabilities so VariantSelector can reject an incompatible
                    // variant BEFORE downloading it, instead of taking manifest.defaultVariant blind.
                    abis = Build.SUPPORTED_ABIS?.toList() ?: emptyList(),
                    totalMemMb = deviceMemoryMb(context),
                    endpoint = hubConfig?.endpoint ?: HubResolver.DEFAULT_ENDPOINT,
                    token = hubConfig?.token,
                    // Forwarded whole: the pull reports bytes, rate and phase, and re-deriving a
                    // narrower triple here is what left the facade unable to say anything useful
                    // about a multi-gigabyte transfer.
                    onProgress = { progress -> onDownloadProgress?.onProgress(progress) },
                )
            } catch (e: Exception) {
                throw ModelNotInstalledException(
                    "package '$repoId' is not installed at $modelDir and the Hub pull failed: ${e.message}",
                )
            }
            if (!modelDir.isDirectory) {
                throw ModelNotInstalledException(repoId, cacheDir)
            }
        }

        val repo = LLMRepository(context.applicationContext, cacheDir, initialModel = sanitized)
        if (!repo.isGenerationAvailable && !repo.isTrainingAvailable && !repo.isRagAvailable) {
            throw MissingArtifactException(
                "package '$repoId' has no usable train/inference/embedding config under $modelDir",
            )
        }

        // #19: fail at construction (not first use) if a requested genuine feature isn't installed.
        // Engine selectors (GenAI/ManualInference) are not downloads — skip them here.
        val installed = detectFeatures(repo)
        features.filterNot { it.isEngineSelector }.forEach { requested ->
            if (requested !in installed) throw FeatureNotInstalledException(requested, installed)
        }

        // GenAI is a selectable engine over the one shared package; #11 owns real GenAI wiring + fallback.
        // Requesting the GenAI/ManualInference feature only sets the engine — it never downloads a 2nd group.
        val resolvedEngine =
            if (features.any { it == ModelFeature.GenAI }) InferenceEngine.GENAI else engine

        // #19: GenAI needs the genai config in the shared package; else fail closed (no silent fallback here).
        if (resolvedEngine == InferenceEngine.GENAI &&
            !File(modelDir, "inference/genai_config.json").isFile
        ) {
            throw EngineUnavailableException(
                InferenceEngine.GENAI,
                "inference/genai_config.json not found in the installed package (re-export with GenAI).",
            )
        }

        // #17/#19: what a picker may offer. Native is the floor; GenAI needs both the package
        // side-car and the native probe — exactly ModelRuntimeFactory's two conditions.
        val genaiInstalled = File(modelDir, "inference/genai_config.json").isFile
        val availableEngines = buildSet {
            add(InferenceEngine.NATIVE)
            if (genaiInstalled && GenAiSupport.available()) add(InferenceEngine.GENAI)
        }

        val capabilities =
            RuntimeCapabilities(
                engine = resolvedEngine,
                availableEngines = availableEngines,
                supportsTraining = repo.isTrainingAvailable,
                supportsMerge = repo.isTrainingAvailable,
                supportsRag = repo.isRagAvailable,
                supportsEmbedding = repo.isRagAvailable,
                // #34: scheduled training is exactly as available as training is — the scheduler is
                // a WorkManager wrapper over the same TrainingJob, with no extra package requirement.
                supportsScheduledTraining = repo.isTrainingAvailable,
                availableFeatures = detectFeatures(repo),
                // What the package IS, not only what it can do. Without this a caller cannot tell a
                // classification encoder from a chat decoder, and can only discover the difference by
                // asking for generation and reading the failure.
                task = PackageTask.read(
                    PackagePaths.forCache(modelDir.parentFile, modelDir.name).inference,
                ),
            )

        val session =
            RepositoryBackedModelSession(
                repo = repo,
                capabilities = capabilities,
                modelDir = modelDir,
                inferencePackagePath = PackagePaths.forCache(modelDir.parentFile, modelDir.name).inference.absolutePath,
            )
        return MobileTransformerModel(session, capabilities, repoId)
    }

    /**
     * The model packages already installed in [cacheDir], newest-agnostic and cheap enough to call
     * from a screen's initial load.
     *
     * #17/#19 gap found building the showcase app: `CacheIndex.list` existed, but nothing on the
     * public entry point exposed it, so an app could not answer "what do I already have?" — which is
     * the first question a Models screen has to answer, and the reason the old sample app simply
     * assumed an `adb push`ed package that no real user can produce.
     *
     * Returns an empty list when [cacheDir] does not exist yet: "nothing installed" is a normal
     * first-run state, not an error.
     */
    @JvmStatic
    fun installed(cacheDir: String): List<CacheIndex.InstalledPackage> =
        CacheIndex.list(File(cacheDir))

    /** Convenience overload using the same default [cacheDir] as [fromPretrained]. */
    @JvmStatic
    fun installed(context: Context): List<CacheIndex.InstalledPackage> =
        installed(context.filesDir.absolutePath)

    /** Total device RAM in MB, for the #13 `recommendedDeviceMemoryMb` filter. Null when unavailable. */
    private fun deviceMemoryMb(context: Context): Int? =
        runCatching {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val info = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
            (info.totalMem / (1024L * 1024L)).toInt()
        }.getOrNull()

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

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
import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest
import com.martinkorelic.mobiletransformers.packages.DeviceCapabilities
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.PackageTask
import com.martinkorelic.mobiletransformers.packages.ToolCallSupport
import com.martinkorelic.mobiletransformers.repository.LLMRepository
import com.martinkorelic.mobiletransformers.runtime.GenAiSupport
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import com.martinkorelic.mobiletransformers.runtime.ModelRuntimeFactory
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
            // Shared with PackageDownloadWorker: the two download paths must agree on which groups a
            // feature set implies, or a background pull installs a different package than this one.
            val featureGroups = DeviceCapabilities.downloadGroups(features)
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
                    abis = DeviceCapabilities.abis(),
                    totalMemMb = DeviceCapabilities.totalMemoryMb(context),
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
        val genaiInstalled = File(modelDir, "inference/genai_config.json").isFile
        if (resolvedEngine == InferenceEngine.GENAI && !genaiInstalled) {
            throw EngineUnavailableException(
                InferenceEngine.GENAI,
                "inference/genai_config.json not found in the installed package (re-export with GenAI).",
            )
        }

        // #17/#19: what a picker may offer — ModelRuntimeFactory's OWN rule, asked ahead of time.
        //
        // This used to apply two of the factory's three conditions, omitting the manifest's
        // `supportedEngines`. FunctionGemma ships a genai_config.json but declares
        // `supportedEngines: ["native"]`, so the facade advertised GenAI, the app offered it, and the
        // factory then refused it mid-load with "explicitly requested but not selectable". See
        // ModelRuntimeFactory.enginesAvailableFor.
        val declaredEngines = declaredEnginesFor(modelDir)
        val availableEngines = ModelRuntimeFactory.enginesAvailableFor(
            declaredEngines = declaredEngines,
            genaiConfigPresent = genaiInstalled,
            genaiAvailable = GenAiSupport.available(),
        )
        if (resolvedEngine == InferenceEngine.GENAI && InferenceEngine.GENAI !in availableEngines) {
            // Named at load, where the package is in hand, rather than as a null runtime discovered
            // at the first generate() — which is what "Generation session was never created" was.
            throw EngineUnavailableException(
                InferenceEngine.GENAI,
                "the installed package declares supportedEngines=${declaredEngines ?: "(none)"} for " +
                    "its variant, so GenAI is not a valid engine for it. Gemma-3 packages are " +
                    "exported through optimum rather than the GenAI builder and are Native-only. " +
                    "Load it with engine=NATIVE.",
            )
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
                // From the model's own chat template, with the names as a fallback. `repoId` is
                // included because the manifest's `baseModelId` is provenance, not an install key —
                // and because passing only the architecture (`gemma3_text`) is exactly the bug this
                // replaces.
                trainingParameterCount = manifestOf(modelDir)?.trainingParameterCount ?: 0L,
            // Which fine-tuning technique this package carries. Recorded by the exporter since
            // the training stage existed; read here for the first time.
            peftMethods = manifestOf(modelDir)?.peftMethods.orEmpty().toSet(),
                toolCalling = ToolCallSupport.read(
                    tokenizerDir = PackagePaths.forCache(modelDir.parentFile, modelDir.name).tokenizer,
                    hints = listOf(repoId, sanitized),
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

    /**
     * The installed variant's declared engines, or `null` when the package declares none.
     *
     * Mirrors `LLMRepository.installedSupportedEngines` deliberately: the facade must answer "may I
     * offer GenAI" from the same declaration the loader will judge the request against.
     */
    private fun declaredEnginesFor(modelDir: File): Set<String>? =
        manifestOf(modelDir)?.supportedEnginesFor()

    /** The installed manifest, or null when absent or unreadable. Never throws: it is a hint source. */
    private fun manifestOf(modelDir: File): MobileTransformersManifest? {
        val manifestFile = File(modelDir, PackageFormat.MANIFEST_FILENAME)
        if (!manifestFile.isFile) return null
        return runCatching { MobileTransformersManifest.load(manifestFile) }.getOrNull()
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

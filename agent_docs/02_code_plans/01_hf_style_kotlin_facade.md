# HuggingFace-Style Kotlin Facade

**Priority #19 | Prerequisites: #11 (`01_code_plans/03_inference_engine_abstraction_native_and_genai.md`), #17 (`00_code_plans/05_android_facade_foundation.md`) | Blocks: #21 (`02_code_plans/04_hub_pull_and_cache_flow.md`), #22 (`02_code_plans/06_adapter_pushback.md`)**

## Purpose

Give Android developers a single HuggingFace-style entry point — `MobileTransformers.fromPretrained(...)` → `MobileTransformerModel` — that wraps the existing repository stack (`LLMRepository`, `TrainingRepository`, `InferenceRepository`, `RagRepository`) without rewriting the JNI/C++ engine. The facade owns: package validation, repository wiring, engine selection (Native default / GenAI opt-in), translation of user-facing config objects into the existing `ORT*Config` types, and friendly errors for missing artifacts. The current `MainActivity` (`/Users/martinkorelic/Developer/Other/mobiletransformers/app/src/main/java/com/martinkorelic/orttransformer/MainActivity.kt`) constructs repositories directly; this plan replaces that wiring with the facade.

This is the public surface that Tier 1 docs (`02_tier1_hf_integrated_core.md` §"Kotlin Facade", §"Config Objects", §"PEFT Surface") promise. It assumes #17 has already renamed the SDK module to `:MobileTransformers`, moved the library namespace to `com.martinkorelic.mobiletransformers`, and stood up the manifest-first package validator + cache installer from `00_code_plans/06_manifest_first_package_and_cache_bridge.md`.

## Touched / new files

All new facade code lives under the post-rename namespace `com.martinkorelic.mobiletransformers` in the SDK module (currently `android/ORTransformer/ORTransformersMobile/`, renamed by #16/#17 to `android/MobileTransformers/MobileTransformers/`).

New files:

- `…/com/martinkorelic/mobiletransformers/MobileTransformers.kt` — the static entry object (`fromPretrained`).
- `…/com/martinkorelic/mobiletransformers/MobileTransformerModel.kt` — the model handle (instance methods).
- `…/com/martinkorelic/mobiletransformers/config/TrainConfig.kt`
- `…/com/martinkorelic/mobiletransformers/config/GenerationConfig.kt`
- `…/com/martinkorelic/mobiletransformers/config/RagConfig.kt`
- `…/com/martinkorelic/mobiletransformers/config/PeftConfig.kt`
- `…/com/martinkorelic/mobiletransformers/config/DeviceConfig.kt`
- `…/com/martinkorelic/mobiletransformers/config/DatasetConfig.kt`
- `…/com/martinkorelic/mobiletransformers/config/HubConfig.kt`
- `…/com/martinkorelic/mobiletransformers/runtime/RepositoryBackedModelRuntime.kt` — the adapter over the four repositories.
- `…/com/martinkorelic/mobiletransformers/runtime/ConfigMapping.kt` — `ORT*Config` translation (kept `internal`).
- `…/com/martinkorelic/mobiletransformers/MobileTransformersException.kt` — friendly error hierarchy.
- `…/com/martinkorelic/mobiletransformers/InferenceEngine.kt` — `enum class InferenceEngine { Native, GenAI }`.
- `…/com/martinkorelic/mobiletransformers/ModelFeature.kt` — `enum class ModelFeature { Inference, Training, Rag }` (may already exist from #17; reuse).
- `…/com/martinkorelic/mobiletransformers/compat/LegacyAliases.kt` — deprecated `typealias` shims under `com.martinkorelic.ortmobile`.

Touched files:

- `app/src/main/java/com/martinkorelic/orttransformer/MainActivity.kt` (→ renamed `…/mobiletransformers/app/MainActivity.kt`) — swap direct repository construction for the facade.
- The three ViewModels (`InferenceViewModel`, `TrainingViewModel`, `ConfigurationViewModel`) — accept a `MobileTransformerModel` instead of raw repositories (see Migration section).
- `…/repository/LLMRepository.kt`, `TrainingRepository.kt`, `InferenceRepository.kt`, `RagRepository.kt` — keep `public` for Tier-1 compat; no API changes required by this plan.

The existing config classes `ORTTrainingConfig`, `ORTGenerationConfig`, `ORTRagConfig` (and `ORTRagArguments`) stay where they are and become **internal-by-convention** (kept public only for the deprecation window — see Compatibility).

## Data contracts / interfaces

### Entry object

```kotlin
object MobileTransformers {
    @JvmStatic
    suspend fun fromPretrained(
        context: Context,
        repoId: String,                      // e.g. "mobiletransformers/Qwen2-0.5B-mobile" OR a local cache folder name
        cacheDir: String = context.filesDir.absolutePath,
        revision: String? = null,            // Hub revision; ignored for already-cached local packages in this plan
        variant: String? = null,             // manifest variant id, e.g. "cpu-int4"; null => manifest defaultVariant
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
        engine: InferenceEngine = InferenceEngine.Native,
        hubConfig: HubConfig? = null         // token + endpoint; only needed for remote pull (#21)
    ): MobileTransformerModel
}
```

In this plan `fromPretrained` resolves an **already-installed** package in `<cacheDir>/<sanitizedRepoId>/` (the cache layout `LLMRepository` discovers via `availableModels` and the `train/ inference/ embedding/` config paths). Remote download is #21; here we validate + wire only. `hubConfig`/`revision` are carried through but a missing local package raises `ModelNotInstalledException` rather than downloading.

### Model handle

```kotlin
class MobileTransformerModel internal constructor(
    private val runtime: RepositoryBackedModelRuntime,
    val repoId: String,
    val installedFeatures: Set<ModelFeature>,
    val engine: InferenceEngine
) {
    suspend fun applyPeft(peft: PeftConfig)
    suspend fun train(dataset: DatasetConfig, config: TrainConfig = TrainConfig(), callback: TrainCallback? = null)
    suspend fun merge()
    suspend fun generate(prompt: String, config: GenerationConfig = GenerationConfig(), callback: GenerateCallback? = null): String
    suspend fun retrieve(query: String, config: RagConfig = RagConfig(), callback: RetrieveCallback? = null): List<RetrievedDocument>
    suspend fun pushAdapter(hubConfig: HubConfig, repoId: String): PushResult   // optional; throws NotImplementedFeature in Tier 1 unless #22 lands
    fun close()
}
```

Callbacks are thin public re-exposures of the existing repository callbacks so app code does not import `ORT*`/repository types. The facade defines:

- `interface TrainCallback` — mirrors `TrainingRepository.TrainingCallback` (`onModelLoadStart/End`, `onDataLoadEnd(totalSteps, stepsPerEpoch)`, `onStepEnd(progress)`, `onEpochEnd(progress)`, `onMergeStart/End(progress)`, `onCompletion(progress)`, `onError(throwable)`), passing a public `TrainProgress` data class that maps 1:1 from `TrainingProgress`.
- `interface GenerateCallback` — mirrors `InferenceRepository.GenerationCallback` (`onStartGeneration`, `onPartialResult(progress)`, `onCompletion(progress)`, `onError`), public `GenerateProgress` mapped from `InferenceProgress`.
- `interface RetrieveCallback` — mirrors `RagRepository.RagCallback` (`onQueryResults(result)`, `onQueryEnd`, `onError`), public `RetrievedDocument` mapped from `RagResult`.

### PeftConfig surface and mapping

`PeftConfig` is a sealed class. **Important repo fact:** the Kotlin side has *no* MARS/LoRA fields today — PEFT is a Python-export-time concern (`trainer/builder.py` `optimum_hf_export(train_method, lora_rank, lora_alpha, …)` and `peft_models/mars/config.py` `MarsConfig.optimization_level`). On-device, the package's `train/training_config.json` is already produced with the chosen method baked in (`requires_grad`, `frozen_params`, `peft_mapping`, `rank`, `alpha`, `peft_target`). So `applyPeft` on Android is a **selection/validation** step against what the installed package supports, plus stashing rank/alpha overrides into the runtime's `ORTTrainingConfig` where applicable — not a graph rewrite.

```kotlin
sealed class PeftConfig {
    abstract val rank: Int
    abstract val alpha: Int
    open val targetModules: List<String>? = null

    data class Lora(override val rank: Int = 16, override val alpha: Int = 32,
                    override val targetModules: List<String>? = null) : PeftConfig()
    data class MarsOpt0(override val rank: Int = 8, override val alpha: Int = 8,
                        override val targetModules: List<String>? = null) : PeftConfig()
    data class MarsOpt1(override val rank: Int = 8, override val alpha: Int = 8,
                        override val targetModules: List<String>? = null) : PeftConfig()
    data class MarsQuantized(override val rank: Int = 8, override val alpha: Int = 8,
                             val optimizationLevel: Int = 4,    // 2,3,4 from MarsConfig
                             val quantNBits: Int = 8,           // 8 or 4 (MarsConfig.quant_n_bits)
                             override val targetModules: List<String>? = null) : PeftConfig()
}
```

Mapping to the Python `train_method` / `MarsConfig.optimization_level` taxonomy (the values the *package* was exported with — verified in `trainer/builder.py` argparse `choices=["lora","lora-xs","mars","nolora"]` and `peft_models/mars/config.py`):

| PeftConfig variant | `train_method` | `mars.optimization_level` | `lora_rank` | `lora_alpha` | quant |
| --- | --- | --- | --- | --- | --- |
| `Lora` | `lora` | n/a | `rank` | `alpha` | none |
| `MarsOpt0` | `mars` | `0` (fully trainable, no quant) | `rank` (→ `r`) | `alpha` | none |
| `MarsOpt1` | `mars` | `1` (partial trainable, frozen+fused down proj, no quant) | `rank` | `alpha` | none |
| `MarsQuantized` | `mars` | `optimizationLevel` (2/3/4) | `rank` | `alpha` | `quantNBits` (8/4), levels 2=partial, 3=full, 4=partial+full |

`applyPeft` reads the installed `training_config.json` (already in cache) plus the manifest's `peftMethods` list. If the requested variant's `(train_method, optimization_level)` does not match what the package shipped, throw `PeftMismatchException` listing what the package supports. Rank/alpha overrides that exceed the exported adapter shape are rejected the same way — on-device training cannot change the exported PEFT topology, only re-run within it.

### Public → ORT* config mapping (`ConfigMapping.kt`, `internal`)

`TrainConfig` → `ORTTrainingConfig` (verified fields from `ORTTrainingConfig.kt`):

```kotlin
data class TrainConfig(
    val epochs: Int = 1,
    val batchSize: Int = 4,
    val maxSteps: Int? = 10,
    val saveSteps: Int = 100,
    val gradientAccumulationSteps: Int = 4,
    val learningRate: Float = 1e-4f,
    val scheduler: Scheduler = Scheduler.Linear(),     // -> ORTTrainingConfig.schedulerConfig
    val mergeAtEnd: Boolean = true,
    val saveAtEnd: Boolean = true,
    val resumeFromState: Boolean = true,
    val device: DeviceConfig = DeviceConfig()
) {
    sealed class Scheduler {
        data class Linear(val startFactor: Float = 1.0f, val endFactor: Float = 0.333f) : Scheduler()
        data class Cosine(val minLearningRate: Float = 0f, val warmupSteps: Int = 10) : Scheduler()
    }
}
```

| TrainConfig | ORTTrainingConfig |
| --- | --- |
| `epochs` | `numTrainEpochs` |
| `batchSize` | `batchSize` |
| `maxSteps` | `maxSteps` |
| `saveSteps` | `saveSteps` |
| `gradientAccumulationSteps` | `gradAccumSteps` |
| `mergeAtEnd` | `mergeWeightsAtEnd` |
| `saveAtEnd` | `saveModelAtEnd` |
| `resumeFromState` | `loadFromState` |
| `scheduler=Linear` + `learningRate` | `schedulerType="linear"`, `schedulerConfig=SchedulerConfig.Linear(learningRate, startFactor, endFactor)` |
| `scheduler=Cosine` + `learningRate` | `schedulerType="cosine"`, `schedulerConfig=SchedulerConfig.Cosine(learningRate, minLearningRate, warmupSteps)` |
| `device` | `deviceOptions` (`DeviceOptions`) |
| `dataset` (passed to `train(dataset, …)`) | `datasetOptions` (`DatasetOptions`) |
| facade sets `repoName/onnxName/taskName` | from manifest/runtime, not user-set |

`DatasetConfig` → `ORTTrainingConfig.DatasetOptions`:

```kotlin
data class DatasetConfig(
    val trainFile: String = "arc_e",
    val batchSize: Int? = 64,                 // -> datasetBatchSize
    val maxSequenceLength: Int? = 512,
    val maxSamples: Int? = 256,               // -> maxDatasetLength
    val dropLongSamples: Boolean = false,     // -> removeLongSamples
    val split: Boolean = false,               // -> datasetSplit
    val shuffle: Boolean = false,             // -> datasetShuffle
    val testRatio: Float = 0.0f
)
```

`GenerationConfig` → `ORTGenerationConfig` (verified):

```kotlin
data class GenerationConfig(
    val maxSequenceLength: Int = 128,
    val systemPrompt: String? = null,
    val trackMetrics: Boolean = true,
    val sampling: Sampling = Sampling.Greedy,
    val device: DeviceConfig = DeviceConfig()
) {
    sealed class Sampling {
        object Greedy : Sampling()                                  // method="greedy"
        data class TopK(val k: Int = 10, val temperature: Float = 1f, val seed: Int = 42) : Sampling()
        data class TopP(val p: Float = 0.9f, val temperature: Float = 1f, val seed: Int = 42) : Sampling()
    }
}
```

| GenerationConfig | ORTGenerationConfig |
| --- | --- |
| `maxSequenceLength` | `maxSequenceLength` |
| `systemPrompt` | `systemPrompt` |
| `trackMetrics` | `trackMetrics` |
| `sampling` | `sampling` (`SamplingOptions(method, temperature, topK, topP, seed)`) |
| `device` | `deviceOptions` |
| `engine` (from model handle) | `type` = `"native"` or `"genai"` |
| runtime sets after merge | `loadMergedWeights` (set true once `merge()` has run) |
| from manifest/runtime | `repoName`, `onnxName` |

`RagConfig` → `ORTRagConfig` (verified):

```kotlin
data class RagConfig(
    val topK: Int = 10,
    val searchType: SearchType = SearchType.Semantic,   // Semantic|Text -> "semantic"|"text"
    val embeddingDimension: Int = 256,
    val maxTextLength: Int = 1024,
    val chunkSize: Int = 512,
    val chunkOverlap: Int = 50,
    val device: DeviceConfig = DeviceConfig()
) { enum class SearchType { Semantic, Text } }
```

Maps 1:1 to `ORTRagConfig` (`topK`, `searchType`, `embeddingDimension`, `maxTextLength`, `chunkSize`, `chunkOverlap`, `deviceOptions`; `repoName`/`onnxName` from runtime). Per-call overrides translate to `ORTRagArguments`.

`DeviceConfig` → `ORTTrainingConfig.DeviceOptions` (shared shape across all three):

```kotlin
data class DeviceConfig(
    val executionProvider: String = "cpu",     // -> executionProvider
    val coreConfigId: String = "opt1",
    val memoryConfigId: String = "high_perf",
    val enableProfiling: Boolean = false
)
```

`HubConfig`:

```kotlin
data class HubConfig(val token: String? = null, val endpoint: String = "https://huggingface.co")
```

### Error hierarchy (`MobileTransformersException.kt`)

```kotlin
sealed class MobileTransformersException(message: String, cause: Throwable? = null) : Exception(message, cause)
class ModelNotInstalledException(repoId: String, cacheDir: String) : MobileTransformersException(...)
class MissingArtifactException(feature: ModelFeature, expectedPath: String) : MobileTransformersException(...)
class PeftMismatchException(requested: String, supported: List<String>) : MobileTransformersException(...)
class FeatureNotInstalledException(feature: ModelFeature, installed: Set<ModelFeature>) : MobileTransformersException(...)
class EngineUnavailableException(engine: InferenceEngine, reason: String) : MobileTransformersException(...)
class NotImplementedFeatureException(name: String) : MobileTransformersException(...)
```

Friendly messages name the exact missing path. Examples: `MissingArtifactException(Training, "<cacheDir>/<repo>/train/training_config.json")` → *"Training is not available for 'mobiletransformers/Qwen2-0.5B-mobile': expected train/training_config.json was not found in the installed package. Re-export with --training_mode true or install a package whose manifest lists 'train' in downloadPlan."*

## Implementation steps

1. **Feature detection in `fromPretrained`.** Sanitize `repoId` → cache folder, confirm `<cacheDir>/<sanitizedRepoId>/` exists (else `ModelNotInstalledException`). Run the #17 manifest validator. Derive `installedFeatures` from presence of `train/training_config.json`, `inference/generation_config.json`, `embedding/rag_config.json` (the exact paths `LLMRepository` resolves). Intersect with the requested `features`; if a requested feature's artifacts are absent, throw `FeatureNotInstalledException` early so the failure is at construction, not first use.
2. **Engine selection.** Native is always available. If `engine == GenAI`, verify the package has `inference/genai_config.json` (manifest `downloadPlan.genai` group); else `EngineUnavailableException`. Carry the choice into `RepositoryBackedModelRuntime` so `ORTGenerationConfig.type` is set to `"native"`/`"genai"` per call. (The engine abstraction itself is #11; this facade only selects.)
3. **Build `RepositoryBackedModelRuntime`.** Construct exactly one `LLMRepository(context, cacheDir, initialModel = sanitizedRepoId)` and the three repositories over it (`TrainingRepository(llm)`, `InferenceRepository(llm)`, `RagRepository(llm)`) — the same objects `MainActivity` builds today. The runtime holds them plus the resolved engine and a mutable `mergedWeightsLoaded: Boolean`.
4. **`applyPeft`.** Read `train/training_config.json` (already parsed by `LLMRepository`) + manifest `peftMethods`. Validate the requested `PeftConfig` variant against `(train_method, optimization_level, rank, alpha)`. Store an effective `ORTTrainingConfig` template (rank/alpha cannot exceed exported shape). No native call here.
5. **`train`.** Map `TrainConfig` + `DatasetConfig` → `ORTTrainingConfig` via `ConfigMapping`. Call `LLMRepository.prepareTraining(ortTrainingConfig, dataPreprocessFunction = config.customPreprocess?)`, then `TrainingRepository.performTraining(ortTrainingConfig, trainingCallback = adapter, dataPreprocessFunction)`. Wrap the public `TrainCallback` in a `TrainingRepository.TrainingCallback` adapter that maps `TrainingProgress` → `TrainProgress`. If `TrainConfig.mergeAtEnd`, the underlying flow already merges; set `runtime.mergedWeightsLoaded = true` on `onMergeEnd`.
6. **`merge`.** If training already merged, no-op + log. Otherwise call `TrainingRepository.endTraining(saveModel = true)` (the merge/save path) and set `mergedWeightsLoaded = true`. Map errors to `MissingArtifactException`/generic `MobileTransformersException`.
7. **`generate`.** Map `GenerationConfig` → `ORTGenerationConfig` (set `type` from engine, `loadMergedWeights = runtime.mergedWeightsLoaded`). Call `LLMRepository.prepareGeneration(cfg)` then `InferenceRepository.generate(prompt, cfg, callback adapter)`. Accumulate streamed `onPartialResult` text and return the final string; still forward each partial to the user callback.
8. **`retrieve`.** Require `ModelFeature.Rag`. Map `RagConfig` → `ORTRagConfig`, call `LLMRepository.prepareRetriever(cfg)` then `RagRepository.query(query, ragArgs, callback adapter)`; collect `RagResult` → `List<RetrievedDocument>`.
9. **`pushAdapter`.** Tier-1 stub: throw `NotImplementedFeatureException("pushAdapter")` unless #22 has landed; the signature exists so the public API is stable. Wire to #22's adapter export when available.
10. **`close`.** Call `LLMRepository.resetInference()` and `resetTraining()` to release native sessions.
11. **Compatibility aliases.** In `compat/LegacyAliases.kt`, under `package com.martinkorelic.ortmobile`, add `@Deprecated("Use com.martinkorelic.mobiletransformers.* facade", level = WARNING) typealias` shims for the moved repository classes if/when their package moves, per `02_tier1_hf_integrated_core.md` §Compatibility. Keep `ORT*Config` names referenced only inside `ConfigMapping`/runtime; do not surface them in facade signatures.

## Interactions

- **#17 (facade foundation):** reuses its manifest validator, cache installer, `ModelFeature`, and the renamed module/namespace. This plan assumes #17 merged.
- **#11 (engine abstraction):** the `InferenceEngine` enum and the `ORTGenerationConfig.type` switch are the selector; actual Native/GenAI session handling is owned by #11.
- **#18 (training lifecycle/checkpoint contracts):** `TrainProgress`/`TrainCallback` should align with #18's job/progress events; if #18 introduced a `TrainingJob`, `train()` may return it instead of `Unit` (note for the implementer — keep the callback path regardless).
- **Repositories stay intact:** no change to `LLMRepository`/`TrainingRepository`/`InferenceRepository`/`RagRepository` signatures. The facade is purely additive (decision in `02_tier1_hf_integrated_core.md` §Recommended Decision).
- **#21 (hub pull):** `fromPretrained`'s `revision`/`variant`/`hubConfig` parameters are the seam; #21 fills in download-then-install before validation.
- **Python export (`trainer/builder.py`, `peft_models/mars/config.py`):** the PEFT mapping table is the on-device mirror of the export-time `train_method`/`optimization_level`; keep them in sync.

## Sample-app migration

Today `MainActivity.onCreate` builds the stack directly (verified):

```kotlin
llmRepository = LLMRepository(applicationContext, filesDir.absolutePath)
inferenceRepository = InferenceRepository(llmRepository)
trainingRepository = TrainingRepository(llmRepository)
ragRepository = RagRepository(llmRepository)
// …
val inferenceViewModel = remember { InferenceViewModel(llmRepository, inferenceRepository, ragRepository) }
val trainingViewModel = remember { TrainingViewModel(llmRepository, trainingRepository) }
val configurationViewModel = remember { ConfigurationViewModel(llmRepository) }
```

After migration (lifecycleScope because `fromPretrained` is `suspend`):

```kotlin
val model = lifecycleScope.async {
    MobileTransformers.fromPretrained(
        context = applicationContext,
        repoId = "mobiletransformers/Qwen2-0.5B-mobile",
        cacheDir = filesDir.absolutePath,
        features = setOf(ModelFeature.Inference, ModelFeature.Training, ModelFeature.Rag),
        engine = InferenceEngine.Native
    )
}
// …
val inferenceViewModel = remember { InferenceViewModel(model) }
val trainingViewModel = remember { TrainingViewModel(model) }
val configurationViewModel = remember { ConfigurationViewModel(model) }
```

ViewModels are reworked to depend on the single `MobileTransformerModel` and call `model.train/generate/retrieve` with the public config objects, passing the public callbacks (no `ORT*` imports). A typical generate call: `model.generate(prompt, GenerationConfig(sampling = GenerationConfig.Sampling.TopK(k = 10)), callback)`. Keep the old direct-repository path compilable for one release (regression smoke below) so existing demos do not break.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Config mapping unit tests** (`ConfigMappingTest.kt`): assert every public→`ORT*` field maps (e.g. `TrainConfig(epochs=4,batchSize=4)` → `ORTTrainingConfig(numTrainEpochs=4,batchSize=4)`; `Scheduler.Cosine` → `schedulerType="cosine"` + `SchedulerConfig.Cosine`; `Sampling.TopK` → `SamplingOptions(method="topk"|"greedy"…, topK=…)`; `RagConfig` 1:1).
- **PEFT mapping unit tests:** `MarsOpt1` → `(train_method="mars", optimization_level=1)`; `MarsQuantized(optimizationLevel=4, quantNBits=4)` → quant level 4 / 4-bit; `PeftMismatchException` when requested method not in manifest `peftMethods`.
- **Feature-gate tests:** `fromPretrained(features={Training})` over an inference-only fixture throws `FeatureNotInstalledException`; `engine=GenAI` without `genai_config.json` throws `EngineUnavailableException`.
- **Friendly-error tests:** missing `train/training_config.json` → `MissingArtifactException` whose message contains the exact path.
- Plus the module **compiles** (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Namespace smoke:** a tiny consumer importing `com.martinkorelic.mobiletransformers.MobileTransformers` compiles.
- **Migration regression:** old direct-repository wiring still compiles (`./gradlew :MobileTransformersApp:assembleDebug`); deprecated `com.martinkorelic.ortmobile` aliases resolve with a deprecation warning.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Facade smoke (instrumentation, local fixture package):** `fromPretrained` → `applyPeft(MarsOpt1)` → `train(DatasetConfig(maxSamples=…), TrainConfig(maxSteps=1))` → `merge()` → `generate("hi", GenerationConfig(maxSequenceLength=1))` returns a non-empty string; verify `ORTGenerationConfig.loadMergedWeights==true` after merge.
- **Engine-selector smoke:** same flow with `engine=Native` and (if fixture has genai config) `engine=GenAI`; assert `ORTGenerationConfig.type` is `"native"`/`"genai"` respectively.
- **Migration regression (run leg):** the old direct-repository demo still runs on a device.

**Workflow (end-to-end)** — *(CHECKPOINT #19, device/manual)* `fromPretrained` → `applyPeft` → `train` → `merge` → `generate` on a device, asserting that the merged adapter changes output: capture a `generate` result before training, run `train(maxSteps≥1)` + `merge()`, then `generate` the same prompt with `loadMergedWeights==true` and assert the output differs from the pre-train baseline.

**Definition of done** — `MobileTransformers.fromPretrained` returns a `MobileTransformerModel` for an installed package; the public config objects map 1:1 to the verified `ORT*Config` fields; `applyPeft`/`train`/`merge`/`generate`/`retrieve` drive the unchanged repositories with no `ORT*` types in facade signatures; missing artifacts fail at construction with a path-naming friendly error; the sample app and a deprecated-alias consumer both still compile; and the device workflow above shows merged output diverging from the baseline.

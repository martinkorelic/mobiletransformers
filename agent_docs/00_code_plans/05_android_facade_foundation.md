# Android Facade Foundation (MobileTransformers SDK Public API)
**Priority (global #):** 16  |  **Prerequisites:** #12 (`06_manifest_first_package_and_cache_bridge.md`), #15 (`04_android_gradle_rename_migration.md`)  |  **Blocks:** #17 (`08_training_lifecycle_and_checkpoint_contracts.md`), #18 (`02_code_plans/01_hf_style_kotlin_facade.md`)

## Purpose

Introduce the stable, HF-style public Kotlin API for the renamed `MobileTransformers` SDK module **without rewriting the engine**. The facade `MobileTransformers.fromPretrained(...)` returns a `MobileTransformerModel` whose work is delegated to a `RepositoryBackedModelRuntime` adapter that wraps the existing `LLMRepository` / `TrainingRepository` / `InferenceRepository` / `RagRepository`. Public types use neutral names (`TrainConfig`, `GenerationConfig`, …); the `ORT*` names stay internal.

This realizes the "Mobile Foundation Architecture" section of `00_repository_restructure_plan.md` (lines 380–484) and inherits the IMPLEMENTATION_ORDER canonical decisions: a **dual inference engine over one package** (Native is default/guaranteed; GenAI is a *selectable engine*, not a separate package), and a manifest/cache contract supplied by #12.

Guiding rule (doc 00, line 382): *wrap the current working repositories first, then move implementation behind stable contracts.* First commit is a namespace + adapter layer, not a physical relocation of every class.

## Touched / new files

All under `android/MobileTransformers/MobileTransformers/src/main/java/com/martinkorelic/mobiletransformers/` (post-#15 rename).

**New public surface:**
- `MobileTransformers.kt` — object with `fromPretrained(...)`.
- `MobileTransformerModel.kt` — the returned handle (`train`/`merge`/`generate`/`retrieve`/`capabilities`/`close`).
- `config/TrainConfig.kt`, `config/GenerationConfig.kt`, `config/RagConfig.kt`, `config/PeftConfig.kt`, `config/DeviceConfig.kt`, `config/HubConfig.kt` — public typed configs.
- `runtime/ModelRuntime.kt` — internal-facing runtime contract (interface).
- `runtime/RuntimeCapabilities.kt` — capability flags **plus the `InferenceEngine` selector enum**.
- `packages/ModelFeature.kt`, `packages/ModelVariant.kt`, `packages/MobileTransformersManifest.kt`, `packages/ModelPackageValidator.kt`, `packages/ModelPackageInstaller.kt`, `packages/ChecksumVerifier.kt`, `packages/CacheIndex.kt` — manifest/package contracts (data classes + interfaces; full impl shared with #12).
- `hub/HuggingFaceHubClient.kt`, `hub/HubDownloadRequest.kt`, `hub/HubAuthProvider.kt` — minimal manifest-first resolver/downloader interfaces (impl can be stubbed; real downloader is #20).

**New internal adapters:**
- `internal/runtime/RepositoryBackedModelRuntime.kt` — adapts `LLMRepository` + the three sub-repositories to `ModelRuntime`.
- `internal/config/ConfigMappers.kt` — `TrainConfig.toOrt()` / `GenerationConfig.toOrt()` / `RagConfig.toOrt()` mapping to `ORTTrainingConfig` / `ORTGenerationConfig` / `ORTRagConfig`.
- `internal/repositories/` — keeps references to the existing repository classes (no move required first pass).

**Compatibility:**
- `internal/legacy/Aliases.kt` (extends the #15 `Aliases.kt`) — deprecated typealiases under `com.martinkorelic.ortmobile`.

**Reused existing (NOT rewritten):** `repository/LLMRepository.kt`, `repository/{Training,Inference,Rag}Repository.kt`, `ORTGeneratorNative.kt`, `ORTTrainerNative.kt`, `ORTRetriever.kt`, `ORTGenAINative.kt`, `ORT{Training,Generation,Rag}Config.kt`, `ORTVectorDatabase.kt`.

## Data contracts / interfaces

### Public facade
```kotlin
object MobileTransformers {
    suspend fun fromPretrained(
        context: Context,
        repoId: String,
        revision: String = "main",
        variant: String? = null,
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
        cacheDir: File = context.filesDir,
        engine: InferenceEngine = InferenceEngine.Native,   // selectable; Native default
        hubConfig: HubConfig = HubConfig(token = null),
    ): MobileTransformerModel
}

class MobileTransformerModel internal constructor(
    private val runtime: ModelRuntime,
    val capabilities: RuntimeCapabilities,
) {
    suspend fun train(dataset: DatasetSource, config: TrainConfig = TrainConfig()): TrainingResult
    suspend fun merge(): MergeResult
    suspend fun generate(prompt: String, config: GenerationConfig = GenerationConfig()): GenerationResult
    suspend fun retrieve(query: String, config: RagConfig = RagConfig()): RetrievalResult
    fun close()
}
```
Stable surface to preserve (doc 00, lines 467–482): `fromPretrained → train → merge → generate → retrieve`. Keep it small; do not surface `ORT*` types, `Job`, or `*Native` handles.

### Runtime contract
```kotlin
internal interface ModelRuntime {
    val capabilities: RuntimeCapabilities
    suspend fun train(dataset: DatasetSource, config: TrainConfig): TrainingResult
    suspend fun merge(): MergeResult
    suspend fun generate(prompt: String, config: GenerationConfig): GenerationResult
    suspend fun retrieve(query: String, config: RagConfig): RetrievalResult
    fun close()
}
```
`RepositoryBackedModelRuntime` is the only impl this pass.

### Engine selector + capabilities (`runtime/RuntimeCapabilities.kt`)
```kotlin
enum class InferenceEngine { Native, GenAI }   // engines over ONE shared package

data class RuntimeCapabilities(
    val engine: InferenceEngine,
    val supportsTraining: Boolean,
    val supportsMerge: Boolean,
    val supportsRag: Boolean,
    val supportsEmbedding: Boolean,
    val supportsScheduledTraining: Boolean = false,   // future (#17)
    val supportsAdapterTensorExport: Boolean = false, // future (federation)
    val availableFeatures: Set<ModelFeature>,
)
```
Maps directly onto the existing availability flags `LLMRepository.isTrainingAvailable / isGenerationAvailable / isRagAvailable`.

### Feature enum (`packages/ModelFeature.kt`)
```kotlin
enum class ModelFeature { Inference, Training, Rag, Embedding, GenAI, ManualInference, Adapter }
```
**Critical semantics** (IMPLEMENTATION_ORDER canonical decision #10): `GenAI` and `ManualInference` are **engine selectors over the same shared package**, not separate downloadable packages. The package on disk (`train/`, `inference/`, `embedding/`) is consumable by both the Native ORT engine and the GenAI engine. When `ModelFeature.GenAI`/`ManualInference` appears, it sets/validates `InferenceEngine`, it does not request a different feature-group download. `Inference`, `Training`, `Rag`, `Embedding`, `Adapter` are genuine feature groups.

### Public configs and internal mapping (`internal/config/ConfigMappers.kt`)
Public configs use HF-flavored names; mappers translate to the existing `data class` configs verbatim.

| Public config | Maps to existing | Notable field renames |
| --- | --- | --- |
| `TrainConfig(maxSteps, numEpochs, batchSize, learningRate, schedule, mergeAtEnd, saveAtEnd, device, peft)` | `ORTTrainingConfig` | `maxSteps→maxSteps`, `numEpochs→numTrainEpochs`, `schedule→schedulerType`+`SchedulerConfig`, `mergeAtEnd→mergeWeightsAtEnd`, `saveAtEnd→saveModelAtEnd`, `device→deviceOptions` |
| `GenerationConfig(maxNewTokens, sampling, systemPrompt, device, loadMerged)` | `ORTGenerationConfig` | `maxNewTokens→maxSequenceLength`, `sampling→SamplingOptions`, `device→deviceOptions`, `loadMerged→loadMergedWeights`; `type` fixed to `"native"` (or set from `InferenceEngine`) |
| `RagConfig(topK, searchType, embeddingDimension, chunkSize, chunkOverlap, maxTextLength, device)` | `ORTRagConfig` | 1:1 names; `device→deviceOptions` |
| `PeftConfig(method, rank, alpha, targetModules)` | folds into `ORTTrainingConfig`/handoff metadata | LoRA/MARS trainable-tensor selection; first pass may be metadata-only |
| `DeviceConfig(executionProvider, coreConfigId, memoryConfigId, enableProfiling)` | `DeviceOptions` | 1:1 |
| `HubConfig(token, endpoint?)` | consumed by `hub/` | maps to `HubAuthProvider` |

`InferenceEngine.GenAI` → `ORTGenerationConfig.type = "genai"` is currently a deprecated/disabled native path (`ORTGenAINative`); validate against `RuntimeCapabilities.engine` and fall back to `Native` with a clear error if GenAI is unavailable in the build. (Real GenAI engine selection is #18 / `01_code_plans/03`.)

### Manifest / package contract (shared with #12)
- `MobileTransformersManifest` parses `mobiletransformers_manifest.json`: model id, task, tokenizer metadata, ORT/GenAI versions, `variants: List<ModelVariant>`, per-feature file groups, checksums, Android runtime requirements, and a reference to `weight_handoff_map.json`.
- `ModelVariant(abi, quantization, memoryEstimateBytes, features, engineSupport, isDefault)` — variant selection by ABI/quant/memory/requested features/engine support.
- `ModelPackageValidator` validates expected `<cacheDir>/<model>/train/`, `/inference/`, `/embedding/` paths **before** `fromPretrained` loads (doc 00, line 448 + 496). Preserves the cache layout `LLMRepository` already reads.
- `ModelPackageInstaller` installs a Hub package into that exact cache shape so the existing repositories keep working untouched.
- `ChecksumVerifier` (SHA256) + `CacheIndex` (manifest metadata + storage usage) per doc 00 line 450.
- `DatasetSource` (`training/DatasetSource.kt`): local JSONL/text/tokenized, mapped onto the existing `ORTDataCurator` dataset loader (`trainFile`, `maxSequenceLength`, `datasetBatchSize`, …). Result/event types `TrainingResult`, `MergeResult`, `GenerationResult`, `RetrievalResult` wrap the existing callback payloads (`TrainingProgress`, `InferenceProgress`, `RagResult`).

## Implementation steps

1. **Land #15 first.** Confirm the rename build gate is green; this plan only adds files under the renamed module.
2. **Add public config classes** (`config/*.kt`) as plain `data class`es with sane defaults matching the existing config defaults.
3. **Add `ConfigMappers.kt`** with `toOrt()` extensions; unit-test round-trip against the existing `ORT*Config` defaults so no behavior shifts.
4. **Add `ModelFeature`, `InferenceEngine`, `RuntimeCapabilities`** with the engine-selector semantics documented above.
5. **Add manifest/package types** (`packages/*.kt`) as the data-class + interface surface; share concrete parsing/validation with #12 (do not duplicate logic — depend on it). `ModelPackageValidator`/`Installer` may delegate to the #12 implementation.
6. **Implement `RepositoryBackedModelRuntime`**:
   - Construct `LLMRepository(applicationContext, cacheDir.path, initialModel = <selected model>)`.
   - `train(...)`: map `TrainConfig.toOrt()`, call `TrainingRepository.performTraining(...)`, surface progress via a `TrainingCallback` adapter → `TrainingProgress` events; `join()` internally so the public `suspend` fun returns a final `TrainingResult`.
   - `merge()`: drive the existing merge path (`mergeWeightsAtEnd` / `ORTTrainerNative.mergeExportWeights` via `endTraining(saveModel=true)`); return `MergeResult` with the inference-ready package path.
   - `generate(...)`: map `GenerationConfig.toOrt()`, call `InferenceRepository.generate(...)` with a `GenerationCallback` adapter; collect the stream into `GenerationResult` (and expose a token stream variant later in #18).
   - `retrieve(...)`: `RagRepository.initialize(...)` then `query(...)` via a `RagCallback` adapter → `RetrievalResult`.
   - `capabilities`: read `LLMRepository.isTrainingAvailable/isGenerationAvailable/isRagAvailable` + selected `InferenceEngine`.
   - `close()`: call `resetInference()` / `resetTraining()` and any retriever/DB close.
7. **Implement `MobileTransformers.fromPretrained`**:
   - Resolve manifest (via `hub/` + #12 installer/validator), select `ModelVariant`, validate cache paths, install if missing.
   - Validate requested `features` and `engine` against the variant; for `GenAI`/`ManualInference` set `InferenceEngine` (do not download a second package).
   - Construct `RepositoryBackedModelRuntime` and return `MobileTransformerModel(runtime, capabilities)`.
8. **Wire `MobileTransformerModel`** to delegate every method to `runtime`. No engine logic in this class.
9. **Add compatibility typealiases** under `com.martinkorelic.ortmobile` for any public type external demos referenced; keep for one release.
10. **Keep `MainActivity`/sample app unchanged** except optionally adding a tiny demo that calls the new facade alongside the existing repository-driven screens.

## Interactions

- **Engine boundary (canonical):** the facade must never imply two packages for Native vs GenAI. `InferenceEngine` selects which engine consumes the single `inference/` package. GenAI remains a *selectable* path; Native is the guaranteed default. Defer real GenAI wiring to #18 / `01_code_plans/03`; this pass only models the selector and validates it.
- **Engine internals stay internal:** `ORTGeneratorNative`, `ORTTrainerNative`, `ORTGenAINative`, `ORTRetriever`, `ORTVectorDatabase`, and the four repositories are reached only through `RepositoryBackedModelRuntime`. No `ORT*` type appears in any `public` signature.
- **JNI untouched:** `System.loadLibrary("ortmobile")` and the `Java_com_martinkorelic_ortmobile_*` C++ symbols are not renamed (per #15 option A). The facade sits well above the JNI layer. If #15 left `*Native` classes in the legacy `com.martinkorelic.ortmobile` package, the adapter imports them from there.
- **Cache contract (doc 00 line 496):** `ModelPackageInstaller` writes into `<cacheDir>/<model>/{train,inference,embedding}/…` so `LLMRepository.updatePaths()` (which probes `training_config.json`, `generation_config.json`, `rag_config.json`) keeps discovering models with zero change. Do not alter those JSON formats; the manifest references and validates them (doc 00 line 515).
- **Weight handoff (canonical decision #2/#3):** merge relies on per-tensor external initializers in `inference/` and the `weight_handoff_map.json` source of truth (#12 / `00_code_plans/07`), not on `weight_merger.cpp:904` string rewrites. `merge()` must surface the handoff-validated package path, not re-derive names.
- **Downstream:** #17 extends `MobileTransformerModel`/runtime with `TrainingJob`, scheduling, and checkpoint metadata; #18 builds the richer HF facade and token-stream generate on this base. Keep `ModelRuntime` open for those.

## Tests & smokes

1. **Config mapper round-trip (unit):** `TrainConfig().toOrt()` / `GenerationConfig().toOrt()` / `RagConfig().toOrt()` equal the existing `ORT*Config()` defaults; non-default fields propagate correctly.
2. **Feature/engine semantics (unit):** asserting `ModelFeature.GenAI`/`ManualInference` resolve to an `InferenceEngine` selection and never trigger a separate feature-group download; `RuntimeCapabilities.engine` reflects the request or falls back to `Native` with a typed error.
3. **Manifest parse + variant select (unit):** parse a `tests/fixtures/mobiletransformers_manifest.json`, select default variant, validate expected cache paths; reject a manifest missing required `inference/` group.
4. **Adapter contract (unit, mocked repositories):** `RepositoryBackedModelRuntime` calls the correct repository methods for `train`/`merge`/`generate`/`retrieve` and maps callbacks to result types; `close()` calls `resetInference`/`resetTraining`.
5. **Facade smoke (instrumented):** with a small pre-installed fixture package in `cacheDir`, `fromPretrained(...).generate("Hello", GenerationConfig(maxNewTokens = 16))` returns non-empty text via the Native engine and `libortmobile.so` (no `UnsatisfiedLinkError`).
6. **Public-surface lint:** a test/script asserting no `ORT*` symbol appears in the module's `public`/`internal-but-exported` API (e.g. grep the generated API signature); compat typealiases under `com.martinkorelic.ortmobile` still compile with deprecation warnings.
7. **Build gate:** `./gradlew :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug :MobileTransformers:testDebugUnitTest` green.

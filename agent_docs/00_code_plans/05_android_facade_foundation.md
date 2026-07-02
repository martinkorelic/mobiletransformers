# Android Facade Foundation (MobileTransformers SDK Public API)
**Priority (global #):** 17  |  **Prerequisites:** #13 (`06_manifest_first_package_and_cache_bridge.md`), #16 (`04_android_gradle_rename_migration.md`)  |  **Blocks:** #18 (`08_training_lifecycle_and_checkpoint_contracts.md`), #19 (`02_code_plans/01_hf_style_kotlin_facade.md`)

## Purpose

Introduce the stable, HF-style public Kotlin API for the renamed `MobileTransformers` SDK module **without rewriting the engine**. The facade `MobileTransformers.fromPretrained(...)` returns a `MobileTransformerModel` whose work is delegated to a `RepositoryBackedModelSession` adapter that wraps the existing `LLMRepository` / `TrainingRepository` / `InferenceRepository` / `RagRepository`. Public types use neutral names (`TrainConfig`, `GenerationConfig`, …); the `ORT*` names stay internal.

> **Naming (canonical, per IMPLEMENTATION_ORDER):** the name `ModelRuntime` is **reserved for the inference-engine boundary** owned by `01_code_plans/03` (#11: `load/generate/release`, impls `ORTGeneratorNative`/`ORTGeneratorGenAI`). This plan's facade-level, whole-model contract (`train/merge/generate/retrieve`) is **`ModelSession`**, implemented by `RepositoryBackedModelSession`. Likewise the engine-level capability flags are #11's `EngineCapabilities`; the model-level flags here keep the name `RuntimeCapabilities`. The `InferenceEngine { NATIVE, GENAI }` enum is declared **once, by #11** (`01_code_plans/03`, which lands first at order 11, in `runtime/InferenceEngine.kt`) and **reused here** — this plan must not declare a second engine enum.

This realizes the "Mobile Foundation Architecture" section of `00_repository_restructure_plan.md` (lines 380–484) and inherits the IMPLEMENTATION_ORDER canonical decisions: a **dual inference engine over one package** (Native is default/guaranteed; GenAI is a *selectable engine*, not a separate package), and a manifest/cache contract supplied by #13.

Guiding rule (doc 00, line 382): *wrap the current working repositories first, then move implementation behind stable contracts.* First commit is a namespace + adapter layer, not a physical relocation of every class.

## Touched / new files

All under `android/MobileTransformers/MobileTransformers/src/main/java/com/martinkorelic/mobiletransformers/` (post-#16 rename).

**New public surface:**
- `MobileTransformers.kt` — object with `fromPretrained(...)`.
- `MobileTransformerModel.kt` — the returned handle (`train`/`merge`/`generate`/`retrieve`/`capabilities`/`close`).
- `config/TrainConfig.kt`, `config/GenerationConfig.kt`, `config/RagConfig.kt`, `config/PeftConfig.kt`, `config/DeviceConfig.kt`, `config/DatasetConfig.kt`, `config/HubConfig.kt` — public typed configs.
- `runtime/ModelSession.kt` — internal-facing whole-model contract (interface; **not** `ModelRuntime`, which is #11's engine boundary).
- `runtime/InferenceEngine.kt` — REUSED from #11 (`01_code_plans/03` declares the single `InferenceEngine { NATIVE, GENAI }` enum; it already exists when this plan runs).
- `runtime/RuntimeCapabilities.kt` — model-level capability flags.
- `packages/ModelFeature.kt` — NEW here (the feature enum). The rest — `packages/{ModelVariant,MobileTransformersManifest,ModelPackageValidator,ModelPackageInstaller,ChecksumVerifier,CacheIndex}.kt` — **already exist**: #13 created them (order 13, in `com.martinkorelic.ortmobile.packages`) and the #16 rename moved them to `com.martinkorelic.mobiletransformers.packages`. This plan only *promotes* them into the public surface (visibility/API review + facade wiring) — do not re-create or duplicate them.
- `hub/HuggingFaceHubClient.kt`, `hub/HubDownloadRequest.kt`, `hub/HubAuthProvider.kt` — minimal manifest-first resolver/downloader interfaces (impl can be stubbed; real downloader is #21).

**New internal adapters:**
- `internal/runtime/RepositoryBackedModelSession.kt` — adapts `LLMRepository` + the three sub-repositories to `ModelSession`.
- `internal/config/ConfigMappers.kt` — `TrainConfig.toOrt()` / `GenerationConfig.toOrt()` / `RagConfig.toOrt()` mapping to `ORTTrainingConfig` / `ORTGenerationConfig` / `ORTRagConfig`.
- `internal/repositories/` — keeps references to the existing repository classes (no move required first pass).

**Compatibility:**
- `internal/legacy/Aliases.kt` (extends the #16 `Aliases.kt`) — deprecated typealiases under `com.martinkorelic.ortmobile`.

**Reused existing (NOT rewritten):** `repository/LLMRepository.kt`, `repository/{Training,Inference,Rag}Repository.kt`, `ORTGeneratorNative.kt`, `ORTTrainerNative.kt`, `ORTRetriever.kt`, `ORT{Training,Generation,Rag}Config.kt`, `ORTVectorDatabase.kt`. (`ORTGenAINative.kt` no longer exists — deleted by #11; the GenAI engine is `ORTGeneratorGenAI` behind #11's `ModelRuntime`.)

## Data contracts / interfaces

### Public facade — FINAL signature set (canonical for #19/#24 too)

This block is the **single definition site** of the facade signatures. `02_code_plans/01` (#19) *extends* these files (adds `applyPeft`, `pushAdapter`, public callbacks, sample-app migration) — it must not re-declare or re-shape them. `03_code_plans/02` (#24) locks sampling/streaming parity over the same names.

```kotlin
object MobileTransformers {
    @JvmStatic
    suspend fun fromPretrained(
        context: Context,
        repoId: String,
        cacheDir: String = context.filesDir.absolutePath,   // String — matches LLMRepository
        revision: String = "main",
        variant: String? = null,                            // manifest variant id; null => defaultVariant
        features: Set<ModelFeature> = setOf(ModelFeature.Inference),
        engine: InferenceEngine = InferenceEngine.NATIVE,   // selectable; Native default
        hubConfig: HubConfig? = null,                       // token/endpoint; only needed for remote pull (#21)
    ): MobileTransformerModel
}

class MobileTransformerModel internal constructor(
    private val session: ModelSession,
    val capabilities: RuntimeCapabilities,
) {
    suspend fun train(dataset: DatasetConfig, config: TrainConfig = TrainConfig()): TrainingResult
    suspend fun merge(): MergeResult
    suspend fun generate(prompt: String, config: GenerationConfig = GenerationConfig()): GenerationResult
    suspend fun retrieve(query: String, config: RagConfig = RagConfig()): RetrievalResult
    fun close()
}
```

Signature decisions (final; #19 inherits them verbatim):
- `generate` returns **`GenerationResult`** (text + metrics from `InferenceProgress`); plain-text callers use `result.text`.
- The dataset argument type is **`DatasetConfig`** (one name everywhere; there is no separate `DatasetSource` type).
- `cacheDir` is a **`String`** (what `LLMRepository` takes), not `File`.
- The public generation length field is **`maxNewTokens`** from day one (mapped to internal `maxSequenceLength`); sampling is `03_code_plans/02`'s `SamplingConfig(method: SamplingMethod, …)` — no plan introduces a competing sealed `Sampling` class.
- The public exception hierarchy (`MobileTransformersException` + `ModelNotInstalledException`/`MissingArtifactException`/`PeftMismatchException`/`FeatureNotInstalledException`/`EngineUnavailableException`/`NotImplementedFeatureException`) is defined in `02_code_plans/01`; this plan raises through it (stub the base + the two construction-time exceptions here if #19 has not landed). `00_code_plans/10`'s Python `exceptions.py` mirrors these names.

Stable surface to preserve (doc 00, lines 467–482): `fromPretrained → train → merge → generate → retrieve`. Keep it small; do not surface `ORT*` types, `Job`, or `*Native` handles.

### Whole-model contract (`ModelSession` — NOT #11's `ModelRuntime`)
```kotlin
internal interface ModelSession {
    val capabilities: RuntimeCapabilities
    suspend fun train(dataset: DatasetConfig, config: TrainConfig): TrainingResult
    suspend fun merge(): MergeResult
    suspend fun generate(prompt: String, config: GenerationConfig): GenerationResult
    suspend fun retrieve(query: String, config: RagConfig): RetrievalResult
    fun close()
}
```
`RepositoryBackedModelSession` is the only impl this pass. For generation it delegates (via the repositories) to whichever engine `ModelRuntimeFactory` (#11) selected — it never talks to a concrete engine.

### Engine selector + capabilities (`runtime/InferenceEngine.kt`, `runtime/RuntimeCapabilities.kt`)
```kotlin
// InferenceEngine { NATIVE, GENAI } is declared by #11 (01_code_plans/03) and REUSED here — engines over ONE shared package

data class RuntimeCapabilities(               // model-level flags; #11's engine-level flags are EngineCapabilities
    val engine: InferenceEngine,
    val supportsTraining: Boolean,
    val supportsMerge: Boolean,
    val supportsRag: Boolean,
    val supportsEmbedding: Boolean,
    val supportsScheduledTraining: Boolean = false,   // future (#18)
    val supportsAdapterTensorExport: Boolean = false, // future (federation)
    val availableFeatures: Set<ModelFeature>,
)
```
Maps directly onto the existing availability flags `LLMRepository.isTrainingAvailable / isGenerationAvailable / isRagAvailable`.

### Feature enum (`packages/ModelFeature.kt`)
```kotlin
enum class ModelFeature { Inference, Training, Rag, Embedding, GenAI, ManualInference, Adapter }
```
**Critical semantics** (IMPLEMENTATION_ORDER canonical decision #11): `GenAI` and `ManualInference` are **engine selectors over the same shared package**, not separate downloadable packages. The package on disk (`train/`, `inference/`, `embedding/`) is consumable by both the Native ORT engine and the GenAI engine. When `ModelFeature.GenAI`/`ManualInference` appears, it sets/validates `InferenceEngine`, it does not request a different feature-group download. `Inference`, `Training`, `Rag`, `Embedding`, `Adapter` are genuine feature groups.

### Public configs and internal mapping (`internal/config/ConfigMappers.kt`)
Public configs use HF-flavored names; mappers translate to the existing `data class` configs verbatim.

| Public config | Maps to existing | Notable field renames |
| --- | --- | --- |
| `TrainConfig(epochs, batchSize, maxSteps, saveSteps, gradientAccumulationSteps, learningRate, scheduler, mergeAtEnd, saveAtEnd, resumeFromState, device)` | `ORTTrainingConfig` | `epochs→numTrainEpochs`, `scheduler→schedulerType`+`SchedulerConfig`, `mergeAtEnd→mergeWeightsAtEnd`, `saveAtEnd→saveModelAtEnd`, `resumeFromState→loadFromState`, `device→deviceOptions` — the authoritative field-by-field table is in `02_code_plans/01` (#19); this row is the same shape, summarized |
| `GenerationConfig(maxNewTokens, sampling, systemPrompt, device, loadMerged)` | `ORTGenerationConfig` | `maxNewTokens→maxSequenceLength`, `sampling→SamplingOptions`, `device→deviceOptions`, `loadMerged→loadMergedWeights`; `type` fixed to `"native"` (or set from `InferenceEngine`) |
| `RagConfig(topK, searchType, embeddingDimension, chunkSize, chunkOverlap, maxTextLength, device)` | `ORTRagConfig` | 1:1 names; `device→deviceOptions` |
| `PeftConfig(method, rank, alpha, targetModules)` | folds into `ORTTrainingConfig`/handoff metadata | LoRA/MARS trainable-tensor selection; first pass may be metadata-only |
| `DeviceConfig(executionProvider, coreConfigId, memoryConfigId, enableProfiling)` | `DeviceOptions` | 1:1 |
| `HubConfig(token, endpoint?)` | consumed by `hub/` | maps to `HubAuthProvider` |

`InferenceEngine.GENAI` → `ORTGenerationConfig.type = "genai"` routes to `ORTGeneratorGenAI` via #11's `ModelRuntimeFactory`; validate against `RuntimeCapabilities.engine` and fall back to `NATIVE` with a clear error if GenAI is unavailable in the build. (Engine selection/fallback logic is owned by `01_code_plans/03`.)

### Manifest / package contract (shared with #13)
- `MobileTransformersManifest` parses `mobiletransformers_manifest.json`: model id, task, tokenizer metadata, ORT/GenAI versions, `variants: List<ModelVariant>`, per-feature file groups, checksums, Android runtime requirements, and a reference to `weight_handoff_map.json`.
- `ModelVariant(abi, quantization, memoryEstimateBytes, features, engineSupport, isDefault)` — variant selection by ABI/quant/memory/requested features/engine support.
- `ModelPackageValidator` validates expected `<cacheDir>/<model>/train/`, `/inference/`, `/embedding/` paths **before** `fromPretrained` loads (doc 00, line 448 + 496). Preserves the cache layout `LLMRepository` already reads.
- `ModelPackageInstaller` installs a Hub package into that exact cache shape so the existing repositories keep working untouched.
- `ChecksumVerifier` (SHA256) + `CacheIndex` (manifest metadata + storage usage) per doc 00 line 450.
- `DatasetConfig` (`config/DatasetConfig.kt`): local JSONL/text/tokenized dataset description, mapped onto the existing `ORTDataCurator` dataset loader (`trainFile`, `maxSequenceLength`, `datasetBatchSize`, …). Result/event types `TrainingResult`, `MergeResult`, `GenerationResult`, `RetrievalResult` wrap the existing callback payloads (`TrainingProgress`, `InferenceProgress`, `RagResult`).

## Implementation steps

1. **Land #16 first.** Confirm the rename build gate is green; this plan only adds files under the renamed module.
2. **Add public config classes** (`config/*.kt`) as plain `data class`es with sane defaults matching the existing config defaults.
3. **Add `ConfigMappers.kt`** with `toOrt()` extensions; unit-test round-trip against the existing `ORT*Config` defaults so no behavior shifts.
4. **Add `ModelFeature`, `InferenceEngine`, `RuntimeCapabilities`** with the engine-selector semantics documented above.
5. **Add manifest/package types** (`packages/*.kt`) as the data-class + interface surface; share concrete parsing/validation with #13 (do not duplicate logic — depend on it). `ModelPackageValidator`/`Installer` may delegate to the #13 implementation.
6. **Implement `RepositoryBackedModelSession`**:
   - Construct `LLMRepository(applicationContext, cacheDir.path, initialModel = <selected model>)`.
   - `train(...)`: map `TrainConfig.toOrt()`, call `TrainingRepository.performTraining(...)`, surface progress via a `TrainingCallback` adapter → `TrainingProgress` events; `join()` internally so the public `suspend` fun returns a final `TrainingResult`.
   - `merge()`: drive the existing merge path (`mergeWeightsAtEnd` / `ORTTrainerNative.mergeExportWeights` via `endTraining(saveModel=true)`); return `MergeResult` with the inference-ready package path.
   - `generate(...)`: map `GenerationConfig.toOrt()`, call `InferenceRepository.generate(...)` with a `GenerationCallback` adapter; collect the stream into `GenerationResult` (and expose a token stream variant later in #19).
   - `retrieve(...)`: `RagRepository.initialize(...)` then `query(...)` via a `RagCallback` adapter → `RetrievalResult`.
   - `capabilities`: read `LLMRepository.isTrainingAvailable/isGenerationAvailable/isRagAvailable` + selected `InferenceEngine`.
   - `close()`: call `resetInference()` / `resetTraining()` and any retriever/DB close.
7. **Implement `MobileTransformers.fromPretrained`**:
   - Resolve manifest (via `hub/` + #13 installer/validator), select `ModelVariant`, validate cache paths, install if missing.
   - Validate requested `features` and `engine` against the variant; for `GenAI`/`ManualInference` set `InferenceEngine` (do not download a second package).
   - Construct `RepositoryBackedModelSession` and return `MobileTransformerModel(session, capabilities)`.
8. **Wire `MobileTransformerModel`** to delegate every method to `runtime`. No engine logic in this class.
9. **Add compatibility typealiases** under `com.martinkorelic.ortmobile` for any public type external demos referenced; keep for one release.
10. **Keep `MainActivity`/sample app unchanged** except optionally adding a tiny demo that calls the new facade alongside the existing repository-driven screens.

## Interactions

- **Engine boundary (canonical):** the facade must never imply two packages for Native vs GenAI. `InferenceEngine` selects which engine consumes the single `inference/` package. GenAI remains a *selectable* path; Native is the guaranteed default. Defer real GenAI wiring to #19 / `01_code_plans/03`; this pass only models the selector and validates it.
- **Engine internals stay internal:** `ORTGeneratorNative`, `ORTTrainerNative`, `ORTGeneratorGenAI`, `ORTRetriever`, `ORTVectorDatabase`, and the four repositories are reached only through `RepositoryBackedModelSession`. No `ORT*` type appears in any `public` signature.
- **JNI untouched:** `System.loadLibrary("ortmobile")` and the `Java_com_martinkorelic_ortmobile_*` C++ symbols are not renamed (per #16 option A). The facade sits well above the JNI layer. If #16 left `*Native` classes in the legacy `com.martinkorelic.ortmobile` package, the adapter imports them from there.
- **Cache contract (doc 00 line 496):** `ModelPackageInstaller` writes into `<cacheDir>/<model>/{train,inference,embedding}/…` so `LLMRepository.updatePaths()` (which probes `training_config.json`, `generation_config.json`, `rag_config.json`) keeps discovering models with zero change. Do not alter those JSON formats; the manifest references and validates them (doc 00 line 515).
- **Weight handoff (canonical decision #2/#3):** merge relies on per-tensor external initializers in `inference/` and the `weight_handoff_map.json` source of truth (#13 / `00_code_plans/07`), not on `weight_merger.cpp:904` string rewrites. `merge()` must surface the handoff-validated package path, not re-derive names.
- **Downstream:** #18 extends `MobileTransformerModel`/runtime with `TrainingJob`, scheduling, and checkpoint metadata; #19 builds the richer HF facade and token-stream generate on this base. Keep `ModelSession` open for those.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Config mapper round-trip:** `TrainConfig().toOrt()` / `GenerationConfig().toOrt()` / `RagConfig().toOrt()` equal the existing `ORT*Config()` defaults; non-default fields propagate correctly.
- **Feature/engine semantics:** assert `ModelFeature.GenAI`/`ManualInference` resolve to an `InferenceEngine` selection and never trigger a separate feature-group download; `RuntimeCapabilities.engine` reflects the request or falls back to `Native` with a typed error.
- **Manifest parse + variant select:** parse a `tests/fixtures/mobiletransformers_manifest.json`, select default variant, validate expected cache paths; reject a manifest missing required `inference/` group.
- **Adapter contract (mocked repositories):** `RepositoryBackedModelSession` calls the correct repository methods for `train`/`merge`/`generate`/`retrieve` and maps callbacks to result types; `close()` calls `resetInference`/`resetTraining`.
- **Public-surface lint:** a test/script asserting no `ORT*` symbol appears in the module's `public`/`internal-but-exported` API (e.g. grep the generated API signature); compat typealiases under `com.martinkorelic.ortmobile` still compile with deprecation warnings.
- **Build gate:** `./gradlew :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug :MobileTransformers:testDebugUnitTest` green.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- From a tiny on-disk fixture package (Robolectric/JVM, no device), drive `ModelPackageValidator` over `<cacheDir>/<model>/{train,inference,embedding}` and assert it passes the layout that `LLMRepository.updatePaths()` probes; a fixture missing the `inference/generation_config.json` probe is rejected.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Facade smoke (instrumented):** with a small pre-installed fixture package in `cacheDir`, `fromPretrained(...).generate("Hello", GenerationConfig(maxNewTokens = 16))` returns non-empty text via the Native engine and `libortmobile.so` (no `UnsatisfiedLinkError`).

**Workflow (end-to-end)** — the #17 checkpoint scenario (device, user-run).
- Load a model via the facade and generate one token on a device: `MobileTransformers.fromPretrained(context, repoId, features = setOf(ModelFeature.Inference))` → `model.generate("Hello", GenerationConfig(maxNewTokens = 1))` returns a non-empty `GenerationResult` through the Native engine on a real device/emulator, with no `ORT*` type in the call path and no `UnsatisfiedLinkError`.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- `MobileTransformers.fromPretrained(...)` returns a `MobileTransformerModel` exposing the stable `train/merge/generate/retrieve/capabilities/close` surface; no `ORT*`/`*Native`/`Job` type leaks into any public signature.
- All work is delegated through `RepositoryBackedModelSession` to the existing repositories; engine selection is modeled via `InferenceEngine` (Native default, GenAI selectable over the **one** shared package) and validated against the variant.
- Public configs map 1:1 to the existing `ORT*Config` defaults (round-trip unit test green); compat typealiases under `com.martinkorelic.ortmobile` compile.
- The #17 workflow passes on a device: facade load → one-token generation via Native engine.

# Inference Engine Abstraction — Native And GenAI

**Priority #11 | Prerequisites: #10 (`01_code_plans/02_genai_external_data_swap_spike.md`, which feeds Gate 0.1) | Blocks: #19 (`02_code_plans/01_hf_style_kotlin_facade.md`)**

## Purpose

Introduce a single `ModelRuntime` engine interface with two implementations — **Native** (default, guaranteed) and **GenAI** (opt-in, gated on Gate 0.1) — both operating over the **same package / same `inference/` folder** produced by File #9. The engine is a *selection over one package*, never a separate package or build.

Key principles:
- **Native is the floor.** If GenAI is unavailable, fails to load, or its symbol/ABI is missing, the runtime **falls back to Native** transparently.
- **`ModelFeature.GenAI` / `ModelFeature.ManualInference` are engine selectors, not package variants.** The manifest declares which engines a variant supports; the caller (or auto-selection) picks one.
- **Streaming callback parity.** Both engines emit the same `GenerationCallback` / `InferenceProgress` sequence so the facade and UI are engine-agnostic.

This wraps the existing `ORTGeneratorNative` (Native) and the new GenAI JNI wrapper seeded by File #10's `genai_spike.cpp` (replacing the commented-out `onnx-genai.cpp` and the dead `ORTGenAINative`).

## Touched / new files

Kotlin:
- NEW `android/.../ModelRuntime.kt` — the engine interface + `RuntimeCapabilities` + `Engine` enum + factory/selector.
- `android/.../ORTGeneratorNative.kt` — adapt to implement `ModelRuntime` (Native). Existing `generate(promptText, generationArgs, callback)` (lines 80-244), `createInferenceSession(...)` (JNI, lines ~271-277), `loadMergedWeights` check (lines 41-48) stay; wrap behind the interface.
- NEW `android/.../ORTGeneratorGenAI.kt` — the GenAI `ModelRuntime` impl. Calls new JNI into `genai_runtime.cpp`. Mirrors the `generate(...)` signature and emits the same callbacks.
- `android/.../repository/LLMRepository.kt` — `GenerationCallback` (lines 40-47) and `InferenceProgress` (`ORTProgress.kt:18-27) are the parity contract; repository selects the engine via `RuntimeCapabilities` instead of hard-instantiating `ORTGeneratorNative`.
- `android/.../ORTGenAINative.kt` — **delete** (deprecated, all methods `throw NotImplementedError`); superseded by `ORTGeneratorGenAI` + `genai_runtime.cpp`.
- `android/.../ORTGenerationConfig.kt` — add an optional `engine: Engine? = null` (null = auto-select). Keep `repoName`, `loadMergedWeights`.

C++:
- NEW `android/.../cpp/genai_runtime.cpp` — promote `genai_spike.cpp` (File #10) to a real wrapper: `OgaCreateModel(<inference dir>)`, tokenizer, generator loop with token-by-token streaming, sampling config, release. Uses stable APIs only (no `OgaCreateModelWithInitializers`).
- `android/.../cpp/onnx-genai.cpp` — **remove** (the commented-out dead attempt).
- `android/.../cpp/CMakeLists.txt` — keep linking `onnxruntime-genai` (line ~62); register `genai_runtime.cpp`.

Python (parity, optional): `inference/generator_genai.py` stays as the desktop reference for the GenAI loop.

Manifest:
- The package manifest (owned by `00_code_plans/06`) gains `supported_engines: ["native", "genai"]` per variant + a default engine. This plan *reads* it.

## Data contracts / interfaces

### `Engine` and `RuntimeCapabilities`

```kotlin
enum class Engine { NATIVE, GENAI }

data class RuntimeCapabilities(
    val engine: Engine,
    val supportsStreaming: Boolean,
    val supportsLoadMergedWeights: Boolean,   // external-initializer swap consumed
    val maxContextLength: Int,
)
```

**Execution-provider registry (F3).** The engine and its ORT execution providers are **data-driven, not an `if/elif` over EP names**. Introduce an `EXECUTION_PROVIDER_REGISTRY` keyed by provider (`cpu`, `xnnpack`, `nnapi`, `genai`, future-NPU) whose rows carry the EP's append function + availability probe + an `Engine` affinity; `ModelRuntimeFactory` resolves the ordered provider list for a chosen `Engine` from this registry rather than branching on string names. A new provider (e.g. an NPU EP) is a **registry row + enum member, no business-logic edit** — the same closed-set-is-data principle the PEFT/architecture/merger registries follow. Native composes its `SessionOptions` EP append-order from the registry; GenAI's provider is the `genai` row. `genaiAvailable()` becomes that row's availability probe.

### `ModelRuntime` interface (engine-agnostic)

```kotlin
interface ModelRuntime {
    val capabilities: RuntimeCapabilities
    fun load(cacheDir: String, config: ORTGenerationConfig)      // open session over inference/ dir
    fun generate(
        promptText: String,
        generationArgs: ORTGenerationConfig,
        callback: GenerationCallback? = null,
    ): String                                                    // same return + callback as ORTGeneratorNative.generate
    fun release()
}
```

`generate` MUST drive the existing `GenerationCallback` exactly: `onModelLoadStart` -> `onModelLoadEnd` -> `onStartGeneration(InferenceProgress)` -> N x `onPartialResult(InferenceProgress)` -> `onCompletion(InferenceProgress)`, and `onError(Throwable)` on failure. `InferenceProgress` fields (`token`, `tokenId`, `totalDecodedTokens`, `prefillTimeMs`, `timeToLoadModelMs`, `generationTimeMs`, `avgTokensPerSecond`, `isCompleted`) are populated by both engines so UI code never branches on engine.

### Both engines, one folder

- Native: `createInferenceSession(inferenceModelPath="<cacheDir>/<repoName>/inference", ...)`, then `AddExternalInitializers` (`session_cache.h:702`) for the swapped per-tensor `.bin` files; `session.use_ort_model_bytes_for_initializers=0` (`session_cache.h:717`).
- GenAI: `OgaCreateModel("<cacheDir>/<repoName>/inference")`; external data resolves from that dir; `genai_config.json` `session_options.config_entries` carries `session.model_external_initializers_file_folder_path`.
- The `loadMergedWeights` precondition (File #9) is "handoff map present + all `external_file`s exist with valid checksums" — checked once and exposed via `RuntimeCapabilities.supportsLoadMergedWeights`.

### Engine selection / fallback

```
requested = config.engine ?? manifest.default_engine
candidates = manifest.supported_engines
if requested == GENAI and GENAI in candidates and genaiAvailable():
    try ORTGeneratorGenAI.load(); return it
    catch -> log, fall through
return ORTGeneratorNative.load()   // guaranteed floor
```

`genaiAvailable()` = Gate 0.1 passed AND the GenAI `.so` is linked AND `OgaCreateModel` symbol present (the File #10 symbol check, run once at init). Never throw to the caller for a GenAI-load failure; fall back.

## Implementation steps

1. **Define `ModelRuntime` + `Engine` + `RuntimeCapabilities`** in `ModelRuntime.kt`.
2. **Adapt Native**: make `ORTGeneratorNative` implement `ModelRuntime`. Factor its session open into `load(...)`, keep `generate(...)` and the callback emission as-is. Populate `capabilities` with `engine = NATIVE`, streaming true. Move the `inference/merged` probe (lines 41-48) to the new "handoff map + checksums" check from File #9.
3. **Promote the spike to `genai_runtime.cpp`**: token-by-token loop, `OgaTokenizerStream` decode per token, expose JNI `createGenAISession(inferenceDir, configIds...)`, `genaiGenerateStep(handle): tokenString`, `setSamplingConfig(...)` (mirror Native's), `releaseGenAISession(handle)`.
4. **Implement `ORTGeneratorGenAI.kt`** over those JNI methods; emit the **same** `GenerationCallback` sequence and fill `InferenceProgress` identically (measure `prefillTimeMs`, `timeToLoadModelMs`, per-token timing). `capabilities.engine = GENAI`.
5. **Manifest**: read `supported_engines` + `default_engine`; add `engine` to `ORTGenerationConfig` and to `FileUtil` parsing.
6. **Selector + fallback** in `LLMRepository`: replace direct `ORTGeneratorNative(...)` construction with a `ModelRuntimeFactory.create(config, manifest)` that implements the selection/fallback above.
7. **`genaiAvailable()` init check**: run the File #10 symbol check once (cache result); also verify the Gate 0.1 flag.
8. **Delete dead code**: remove `ORTGenAINative.kt` and `onnx-genai.cpp`; update `CMakeLists.txt` and any imports.
9. **Streaming parity test harness**: a shared test that records the callback event sequence for a fixed prompt and asserts Native and GenAI emit the same ordered events (token strings may differ only if sampling differs; use greedy/temp 0 for the assertion).
10. **Wire the facade**: File #19 (`02_code_plans/01`) calls only `ModelRuntime`, never a concrete engine.

## Interactions

- **File #10 / Gate 0.1**: gates whether GenAI is ever selected; provides the symbol check and the `genai_runtime.cpp` seed.
- **File #9**: provides the single `inference/` folder + handoff map + checksums that both engines consume; defines the `loadMergedWeights` precondition.
- **`00_code_plans/06` (manifest)**: declares `supported_engines` / `default_engine`; this plan reads them.
- **File #19 (HF-style facade)**: `fromPretrained`/generate route through `ModelRuntime`; `ModelFeature.GenAI` / `ModelFeature.ManualInference` map to `Engine.GENAI` / `Engine.NATIVE` as selectors, not as separate downloads.
- **`GenerationCallback` / `InferenceProgress`** (`LLMRepository.kt:40`, `ORTProgress.kt:18`) are the immutable parity contract; do not fork them per engine.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Interface conformance** (`ModelRuntimeTest.kt`): both `ORTGeneratorNative` and `ORTGeneratorGenAI` satisfy `ModelRuntime`; `capabilities.engine` is correct for each.
- **Manifest selection smoke** (JVM): a variant declaring `supported_engines: ["native"]` rejects/ignores a GENAI request and `ModelRuntimeFactory` selects Native (pure selection logic, no device).
- **`genaiAvailable()` / fallback selection** (JVM): with the symbol probe stubbed absent (Gate 0.1 fail simulation), GenAI is never offered even if `config.engine = GENAI`; with a stubbed GenAI-load failure the factory returns Native and the caller sees no error, only a logged warning.
- **EP registry parity** (JVM): every `EXECUTION_PROVIDER_REGISTRY` row resolves to a provider for its `Engine` and the requested EP order is honored (F3 — no `if/elif`).
- Module **compiles** with the dead code removed: `./gradlew :MobileTransformers:compileDebugKotlin` succeeds with `ORTGenAINative.kt` and `onnx-genai.cpp` gone and `CMakeLists.txt` registering `genai_runtime.cpp`; no dangling references/imports.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Streaming-parity harness (logic)**: a shared recorder asserts both engines emit the same ordered callback event *types* (`onStartGeneration` -> `onPartialResult`* -> `onCompletion`) for a fixed scripted token stream — the engine-agnostic sequence contract, exercised without a real device by feeding a fake generator step.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Same-folder dual-engine smoke** (device): load the same File #9 `inference/` dir under each engine, generate one token, both succeed (the runtime expression of Gate 0.1 equivalence).
- **Streaming parity smoke** (device): identical ordered callback events and identical token sequence at temp 0 for a greedy fixed prompt across both engines.
- **Merged-weights reflected per engine** (device): after train->merge (File #9), both engines reflect the changed output on reload.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- A single `ModelRuntime` interface with `ORTGeneratorNative` (Native) and `ORTGeneratorGenAI` (GenAI) implementations, selected over **one** `inference/` package via `ModelRuntimeFactory` with transparent fallback to Native (GenAI failure never reaches the caller).
- Both engines drive the identical `GenerationCallback`/`InferenceProgress` sequence; UI/facade never branch on engine.
- Provider/engine selection is data-driven via `EXECUTION_PROVIDER_REGISTRY` (F3); `ORTGenAINative.kt` and `onnx-genai.cpp` are deleted and the project builds clean.

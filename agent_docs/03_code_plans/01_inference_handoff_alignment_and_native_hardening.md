# Inference Handoff Alignment & Native Path Hardening

**Priority #23 | Prerequisites: #11 (`01_code_plans/03_inference_engine_abstraction_native_and_genai.md`), #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`), #8 (`00_code_plans/07_weight_handoff_map_and_tensor_codec.md`) | Blocks: #24 (`03_code_plans/02`), #26 (`03_code_plans/04`)**

## Purpose

Make the Native inference engine a deliberate, documented, fail-closed component sitting behind the already-defined `ModelRuntime` boundary (#11). This plan does **not** define an engine interface — that is owned by #11. It (a) wires `ORTGeneratorNative` into `ModelRuntime`, (b) retires the legacy `inference/merged/` handoff in favor of the external-initializer + `weight_handoff_map.json` contract (#8/#9), (c) documents the real graph I/O contract, and (d) fixes the conversation-state prepend bug and adds train↔generate lifecycle tests.

This is the Tier-2 expression of the canonical decision: **trained/merged weights are flat per-tensor external initializers in `<cacheDir>/<model>/inference/`, loaded via `AddExternalInitializers`, keyed by `weight_handoff_map.json`.** No `inference/merged/` subdirectory survives as a contract.

## Touched / new files

Kotlin:
- `android/.../ORTGeneratorNative.kt` — replace the `inference/merged` directory probe in the `loadMergedWeights` setter (`ORTGeneratorNative.kt:41-47`) with the handoff-map precondition (map present + all `external_file`s exist + checksums valid). Keep `generate(promptText, generationArgs, callback)` (`:80-82`) and the callback emission (`:123-237`) unchanged. Adapt the class to implement `ModelRuntime` (per #11).
- `android/.../ORTGenerationConfig.kt` — keep `loadMergedWeights` (`:19`) as the public flag; its *meaning* changes to "consume merged external initializers," not "load from `merged/`".
- `android/.../repository/LLMRepository.kt` — `makeOrtNativeInference()` (`:262-282`) and `prepareGeneration` (`:334-380`) route through `ModelRuntimeFactory` (#11), not a direct `ORTGeneratorNative(...)`.
- `android/.../ORTGenAINative.kt` — **delete the six dead `NotImplementedError` stubs** (`:74-94`: `releaseWeightSession`/`releaseGenAISession`/`initializeGenAIInference`/`performGenAIInferenceStep`/`cacheSessionWeights`/`createGenAISession`). This abandoned JNI-GenAI bridge is superseded by the real GenAI engine (`ORTGeneratorGenAI`, owned by #11); it must not linger as half-implemented surface. Retire the file (and `onnx-genai.cpp`) as part of this cleanup.

C++:
- `android/.../cpp/inference.cpp` — no behavior change; document the input/output contract (below). The dynamic input-name resolution (`inference.cpp:112`) and optional `position_ids` handling (`:122-124, 139-144`) are the source of truth for the documented names.
- `android/.../cpp/session_cache.h` — the external-initializer load path (`session_cache.h:662-709`, `AddExternalInitializers` at `:702`) is the handoff consumer; gains the map-driven naming from #8 (do not re-derive names from `<dirname>.<filestem>`).

Docs: this plan feeds `docs/ARCHITECTURE.md` and `docs/PUBLIC_API.md` (owned by `05_code_plans/04`).

## Data contracts / interfaces

### Native graph I/O contract (documented, observed from `inference.cpp`)

| Tensor | Direction | Shape | Notes |
| --- | --- | --- | --- |
| `input_ids` | in | `[batch, seq]` | required (`inference.cpp:19,25`) |
| `attention_mask` | in | `[batch, seq]` | required |
| `position_ids` | in | `[batch, seq]` | optional; present iff in `session_cache->input_names` (`:122-124`) |
| `past_key_values.*` | in | `[batch, num_kv_heads, past_len, head_dim]` | KV-cache, dynamic per layer (`session_cache.h:462-505`) |
| `logits` | out | `[batch, seq, vocab]` | required (`inference.cpp:22`) |
| `present.*` | out | KV update | written back via `updatePastKeyValues` (`inference.cpp:176`) |

KV geometry comes from model metadata `head_dim` / `num_kv_heads` / `num_layers` (`session_cache.h:726-753`).

### Handoff precondition (replaces the `merged/` probe)

```
loadMergedWeightsReady(cacheDir, model) :=
    exists(inference/weight_handoff_map.json)
    AND for each entry.externalDataLocation[role]: file exists in inference/
    AND checksum(file) == manifest checksum
    AND entry dtype/shape match the loaded TensorProto
→ exposed as RuntimeCapabilities.supportsLoadMergedWeights (#11)
```

Fail closed: if the map is present but any file/checksum/dtype/shape fails, raise **before** `createInferenceSession`, with the offending tensor name.

## Implementation steps

1. **Retire the `merged/` probe.** In `ORTGeneratorNative.kt:41-47`, replace the directory existence check with `loadMergedWeightsReady(...)` above. Remove any path construction pointing at `inference/merged/`.
2. **Implement `ModelRuntime`** on `ORTGeneratorNative` (factor session open into `load(...)`, keep `generate(...)`); populate `RuntimeCapabilities(engine=NATIVE, supportsStreaming=true, supportsLoadMergedWeights=<precondition>)`.
3. **Map-driven load** in `session_cache.h:662-709`: take initializer names from `inferenceInitializerNames[role]` (#8), not filesystem reconstruction; verify dtype/shape; then `AddExternalInitializers`.
4. **Fix conversation-state prepend** (`ORTGeneratorNative.kt:96-97`, the `// NOTE: Sometimes one token from the previous assistant message keeps prepending` / `// TODO: Will need fix`). Audit the prompt assembly in `generate(...)` (`:80-237`) for the case where the previous assistant token leaks into the next prompt; add a `resetConversation()` that clears KV/position/attention state and is called on new sessions.
5. **Route via factory** in `LLMRepository.kt:262-282/334-380`.
6. **Error messages**: unsupported model (missing required input), missing/mismatched handoff, shape mismatch — all before session creation.

## Interactions

- **#11 (`ModelRuntime`)**: this plan supplies the Native impl; #11 owns selection/fallback/parity.
- **#8/#9 (handoff map / merger)**: this plan is the load-side consumer; names/dtypes/order come from there.
- **#24 (`03_code_plans/02`)**: depends on the hardened `generate(...)` for the sampling/streaming public-name mapping.
- **`05_code_plans/04`**: the documented I/O contract becomes `docs/ARCHITECTURE.md` content.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Handoff precondition unit test (JVM)** `ORTGeneratorNativeHandoffTest.kt`: map present + all files → ready; missing file / bad checksum / wrong dtype / wrong shape → not ready and a fail-closed error naming the tensor.
- **No `merged/` regression** (grep): assert nothing in the build output / runtime paths constructs or reads `inference/merged/`.
- **No dead-stub regression** (grep): assert `NotImplementedError` no longer appears in `ORTGenAINative`, and the file (and `onnx-genai.cpp`) are gone, not left half-implemented.
- Module **compiles**: `./gradlew :MobileTransformers:compileDebugKotlin` after `ORTGeneratorNative` implements `ModelRuntime`.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Map-driven load assertion (JVM, fixture)**: a tiny `weight_handoff_map.json` + flat per-tensor files → assert initializer names come from `inferenceInitializerNames[role]` (not `<dirname>.<filestem>` reconstruction) and dtype/shape validation passes before any session open.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Native graph I/O smoke (Android)**: load a tiny fixture, assert required inputs present, generate one token.
- **Conversation reset test (Android)**: two sequential prompts; assert no token from prompt 1 leaks into prompt 2.
- **Train↔generate lifecycle test (Android)**: open training session → close → open generation session over the same cache → generate; assert no session/handle leak and merged weights are reflected.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- `ORTGeneratorNative` implements `ModelRuntime` (per #11) and is selected via `ModelRuntimeFactory`, not constructed directly.
- Load is map-driven and fail-closed: `weight_handoff_map.json` present + all `external_file`s exist + checksums/dtype/shape valid, else it raises (naming the offending tensor) **before** `createInferenceSession`.
- Zero references to `inference/merged/` survive anywhere in the load path or build output.
- The six dead `NotImplementedError` GenAI stubs (and `ORTGenAINative` / `onnx-genai.cpp`) are deleted, not left half-implemented.
- The conversation-state prepend bug is fixed and `resetConversation()` clears KV/position/attention state on new sessions.

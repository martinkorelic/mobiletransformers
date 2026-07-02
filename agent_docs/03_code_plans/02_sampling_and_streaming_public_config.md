# Sampling & Streaming Public Config Alignment

**Priority #24 | Prerequisites: #23 (`03_code_plans/01`), #19 (`02_code_plans/01_hf_style_kotlin_facade.md`) | Blocks: #27 (`03_code_plans/05`), `05_code_plans/04` (docs)**

> **Consumes `00_code_plans/09`.** The sampling method is the `SamplingMethod` enum (09), not a bare `String` + the `methodMap = mapOf("greedy" to 0, …)` magic at `ORTGeneratorNative.kt:266`. The public `SamplingConfig` here is the HF-named facade; its `method` field is typed to the enum, and the enum owns the wire-string→native-ordinal mapping (replacing `methodMap`). The HF-aligned numeric rename (`maxNewTokens`) is the alias on 09's `GenerationConfig` Pydantic model on the Python side; this plan defines the Kotlin mirror.

## Purpose

Give the public facade HuggingFace-aligned sampling/generation config names while keeping the existing internal `ORTGenerationConfig` / `SamplingOptions` and the native `SamplingConfig` C++ struct intact. Both engines (Native, GenAI) must emit the **same** `GenerationCallback` / `InferenceProgress` sequence so UI code never branches on engine. No generation behavior changes — this is a naming/mapping + parity-lock plan.

## Touched / new files

Kotlin:
- `android/.../config/SamplingConfig.kt` (public; the type #17's signature set names — created there if not already present, finalized here) — HF-named data class mapped into the internal `SamplingOptions` (`ORTGenerationConfig.kt:3-9`). `method: SamplingMethod` (the 09 enum), not `String`.
- `android/.../ORTGeneratorNative.kt` — replace `methodMap = mapOf("greedy" to 0, "top_k" to 1, "top_p" to 2)` + `methodMap[args.method] ?: 0` (`:266-267`) with `SamplingMethod.fromWire(args.method).nativeOrdinal` (the enum owns the ordinal; fail-closed on unknown).
- `android/.../config/GenerationConfig.kt` (public; created by #17 with `maxNewTokens` from day one) — this plan verifies/locks the `maxNewTokens` → internal `ORTGenerationConfig.maxSequenceLength` (`ORTGenerationConfig.kt:15`) mapping and its new-tokens semantics; no rename happens here.
- `android/.../ORTGenerationConfig.kt` — unchanged internally; gains a `fromPublic(GenerationConfig)` mapper. Keep `overrideConfig` (`:23-42`).
- `android/.../repository/LLMRepository.kt` — `GenerationCallback` (`:40-47`) is the immutable parity contract; do not fork per engine.
- `android/.../ORTProgress.kt` — `InferenceProgress` (`:18-27`) fields are the parity payload; both engines populate all of them.

C++ (reference only, no change): `sampling.h` `SamplingConfig` (`sampling.h:25-36`: `method`, `temperature`, `top_k`, `top_p`, `random_seed`) and `SamplingMethod` enum (`sampling.h:16-20`: GREEDY/TOP_K/TOP_P).

## Data contracts / interfaces

### Public ↔ internal mapping (CANONICAL)

| HF concept | Public (new) | Internal | C++ |
| --- | --- | --- | --- |
| `max_new_tokens` | `GenerationConfig.maxNewTokens` | `ORTGenerationConfig.maxSequenceLength` (`:15`) | — |
| `do_sample`/method | `SamplingConfig.method` (`greedy`/`top_k`/`top_p`) | `SamplingOptions.method` (`:4`) | `SamplingMethod` (`sampling.h:16-20`) |
| `temperature` | `SamplingConfig.temperature` | `SamplingOptions.temperature` (`:5`) | `temperature` |
| `top_k` | `SamplingConfig.topK` | `SamplingOptions.topK` (`:6`) | `top_k` |
| `top_p` | `SamplingConfig.topP` | `SamplingOptions.topP` (`:7`) | `top_p` |
| `seed` | `SamplingConfig.seed` | `SamplingOptions.seed` (`:8`) | `random_seed` |

Defaults must match the existing internal defaults exactly (`greedy`, temp `1.0`, topK `10`, topP `0.9`, seed `42`) so behavior is unchanged when callers omit values.

### Callback sequence (parity contract, both engines)

```
onModelLoadStart → onModelLoadEnd → onStartGeneration(InferenceProgress)
  → N × onPartialResult(InferenceProgress) → onCompletion(InferenceProgress)
onError(Throwable) on failure
```

`InferenceProgress` fields populated by both engines: `token`, `tokenId`, `totalDecodedTokens`, `prefillTimeMs`, `timeToLoadModelMs`, `generationTimeMs`, `avgTokensPerSecond`, `isCompleted` (`ORTProgress.kt:18-27`).

## Implementation steps

1. Add public `SamplingConfig` / `GenerationConfig` data classes with HF names + matching defaults.
2. Add `ORTGenerationConfig.fromPublic(...)` + `SamplingOptions.fromPublic(...)` mappers; the facade (#19) accepts only the public types and maps inward.
3. Confirm `maxNewTokens → maxSequenceLength` is the only renamed numeric; document the semantic (new tokens vs. total sequence) and adjust the mapping if the internal value is total-length rather than new-tokens (verify against `generate(...)` loop bound in `ORTGeneratorNative.kt`).
4. Lock the callback parity: a shared interface, no per-engine subclassing of `GenerationCallback`/`InferenceProgress`.
5. Update the HF mapping table in `03_tier2_inference_and_rag.md` / `docs/PUBLIC_API.md`.

## Interactions

- **#19 (facade)**: consumes the public config types; this plan defines them.
- **#11 / #23**: both `ModelRuntime` impls fill `InferenceProgress` identically.
- **#27 (`03_code_plans/05`)**: grounded generation reuses the public `GenerationConfig`.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Mapping round-trip (JVM)** `SamplingConfigMapperTest.kt`: public→internal→(values) preserves every field and default.
- **Default-equivalence test (JVM)**: a default public `GenerationConfig` produces an internal config byte-equal to today's default `ORTGenerationConfig()` (`greedy`, temp `1.0`, topK `10`, topP `0.9`, seed `42`).
- **Enum mapping test (JVM)**: `SamplingMethod.fromWire("greedy"/"top_k"/"top_p").nativeOrdinal` equals the legacy `methodMap` ordinals (`0`/`1`/`2`); an unknown wire string fails closed.
- Module **compiles**: `./gradlew :MobileTransformers:compileDebugKotlin`.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **`maxNewTokens` semantics check**: drive the `generate(...)` loop bound with a fixed `maxNewTokens` against a tiny fixture and assert it stops at the intended token count (confirms `maxNewTokens → maxSequenceLength` is new-tokens, not total-length).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Streaming parity smoke (Android)**: identical ordered callback events for a greedy/temp-0 fixed prompt across Native and GenAI (shared harness with #11).

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- Public `SamplingConfig` / `GenerationConfig` expose HF-aligned names, with `method: SamplingMethod` (09 enum) and defaults byte-equal to the internal ones.
- The `methodMap` magic at `ORTGeneratorNative.kt:266` is replaced by `SamplingMethod.fromWire(...).nativeOrdinal`; no generation behavior changes.
- Both engines emit the identical `GenerationCallback` / `InferenceProgress` sequence — a single shared interface, no per-engine subclassing.

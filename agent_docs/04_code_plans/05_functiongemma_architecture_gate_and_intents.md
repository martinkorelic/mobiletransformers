# FunctionGemma Architecture Gate & Android Intent Binding

**Priority #37 | Prerequisites: #11 (`01_code_plans/03`, inference engine decision), #7 (`01_code_plans/05`, export), #9 (`01_code_plans/01`, merge) | Blocks: —**

> Tier-3, highest visibility / highest risk. **Gate hard.** If the architecture gate fails, defer to future work — do not let it sink the release. Must not block v1.0.

## Purpose

Personalize an on-device function-calling model (FunctionGemma, Gemma-3, 270M) that emits validated function calls bound to real Android intents — all trained and run locally. The defensible differentiators are on-device *training*, *personalized* per-user action sets, and *real intent binding* (Google's canonical path trains off-device and deploys via LiteRT-LM).

## The real blocker (sharpened)

The gate is the **inference graph**, not training:
- Training export already detects `Gemma3ForCausalLM` → `GemmaOnnxConfig` (`trainer/builder.py:263`); under `00_code_plans/09` that is an architecture-registry entry.
- The inference graph builder only handles `GemmaForCausalLM` / `Gemma2ForCausalLM` (`inference/builder.py:3234-3236`) — **no Gemma-3 inference-model class exists.** The pre-built inference graph (the OOM-avoidance core trick, `00_code_plans/06`/`07`) therefore can't be produced for Gemma-3 yet.

> **Consumes `00_code_plans/09`.** The deliverable is a `Gemma3Model` inference class wired into the **architecture registry** (set `ArchitectureSpec.inference_model_class = Gemma3Model` on the Gemma-3 entry) — **not** a new `elif config.architectures[0] == "Gemma3ForCausalLM"` at `inference/builder.py:3234`. Passing the gate = a registry entry gains a non-`None` `inference_model_class`.

## Touched / new files

Python:
- `inference/builder.py` — add a `Gemma3Model` class and register it as the Gemma-3 entry's `inference_model_class` in 09's architecture registry (no new `elif` at `:3234-3236`). This is the architecture-gate spike's main deliverable.
- `trainer/builder.py` — `:263` (now a registry entry) already covers training export; verify end-to-end with the new inference class.
- NEW `tools/functiongemma/mobile_actions.py` — synthetic per-user mobile-actions dataset generator (mirrors how MiniPersonalQA was built).

Kotlin:
- NEW `android/.../agent/FunctionCallValidator.kt` — structured-output validation (GenAI grammar if Gate 0.1 adopted GenAI; else JSON/function-call validation around the manual loop).
- NEW `android/.../agent/IntentBinder.kt` — maps validated calls → allowlisted Android intents; **dry-run mode** that returns the intended action without executing.

## Data contracts / interfaces

### Action schema

```json
{
  "actionName": "set_alarm",
  "parameters": {"time": "string", "label": "string"},
  "allowedIntent": "android.intent.action.SET_ALARM",
  "validationRules": {"time": "HH:mm"},
  "privacyClass": "harmless-demo"
}
```

### Hard gates (all must be addressed)

1. **Architecture gate** — pass conditions: HF model loads; training artifacts generate (`:263`); **inference graph exports for Gemma-3** (new branch); Android one-step training works or documents a blocker; inference emits valid structured call text.
2. **Differentiation gate** — must show ≥2 of: on-device training, personalized per-user action sets, local validated tool-call generation, Android intent binding, privacy-preserving local data. Running Google's off-device tutorial is not a contribution.
3. **Tool-grammar gate** — if GenAI won Gate 0.1, test grammar-constrained generation to validate calls; else implement lightweight JSON/function-call validation and document the limitation.

### Safety contract

- Never execute arbitrary model output. Validate against a local allowlist.
- Bind only harmless demo intents first.
- Dry-run mode is the default; execution is opt-in and allowlisted.

## Implementation steps

1. **Architecture spike (gate 1):** add the `Gemma3Model` inference class and set it as the registry entry's `inference_model_class` (09); export + validate end-to-end; record pass/fail in the support matrix (`02_code_plans/02`).
2. If gate 1 passes: build the synthetic per-user mobile-actions dataset.
3. Fine-tune FunctionGemma on-device on the personalized action set.
4. `FunctionCallValidator` (grammar or JSON allowlist per gate 3).
5. `IntentBinder` with dry-run + allowlist; bind harmless demo intents.
6. End-to-end demo only after gates 1 + 2 pass.

## Interactions

- **#11 / Gate 0.1**: grammar-constrained generation availability decides the validator strategy.
- **#7 / #9 / #13**: Gemma-3 must flow through the same export/merge/manifest contracts once the inference branch exists.
- **`02_code_plans/02`**: record the Gemma-3 gate result in the support matrix.

## References

- ONNX Runtime GenAI docs (constrained decoding is preview/source-build-only → ship post-hoc JSON validation in v1): https://onnxruntime.ai/docs/genai/
- onnxruntime-genai (constrained-decoding status): https://github.com/microsoft/onnxruntime-genai

## Worked example

A per-user action schema, an allowlist validator, and a dry-run intent binder that never executes:

```json
{
  "actionName": "set_alarm",
  "parameters": {"time": "HH:mm", "label": "string"},
  "allowedIntent": "android.intent.action.SET_ALARM",
  "validationRules": {"time": "HH:mm"},
  "privacyClass": "harmless-demo"
}
```

```kotlin
// FunctionCallValidator.kt — post-hoc JSON validation (GenAI constrained decoding is preview-only)
fun validate(raw: String): ValidatedCall {
    val call = gson.fromJson(raw, ToolCall::class.java)        // parse model output (Gson — the module's single JSON library, per the typed-parsing canonical decision)
    val spec = allowlist[call.actionName]                      // allowlist check
        ?: throw RejectedCallException("action not allowlisted: ${call.actionName}")
    require(spec.validate(call.parameters)) { "parameters fail validationRules" }
    return ValidatedCall(spec.allowedIntent, call.parameters)
}
```

```kotlin
// IntentBinder.kt — dry-run is the default; returns intended action, does NOT startActivity
fun dryRun(call: ValidatedCall): IntendedAction =
    IntendedAction(intent = Intent(call.allowedIntent).apply { putExtras(call.parameters) },
                   willExecute = false)        // execution is opt-in + allowlisted, never here
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/inference/test_gemma3_registry.py` — the Gemma-3 `ArchitectureSpec` entry gains a non-`None` `inference_model_class = Gemma3Model` (the gate-1 pass condition, expressed as registry data — no new `elif` at `inference/builder.py:3234-3236`).
- Structured-output validation (`FunctionCallValidatorTest.kt`): valid tool-call JSON parses and passes; off-allowlist `actionName` and parameters violating `validationRules` are rejected (`RejectedCallException`).
- Intent dry-run (`IntentBinderTest.kt`, JVM/Robolectric): a validated call returns the intended `allowedIntent` action with `willExecute=false`; assert `startActivity` is never called.
- Plus the module **compiles** (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `pytest tests/functiongemma/test_actions_dataset.py` — the synthetic per-user mobile-actions generator (`tools/functiongemma/mobile_actions.py`) emits well-formed action-schema records (each with `actionName`/`allowedIntent`/`validationRules`/`privacyClass`).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Architecture export smoke (gate 1, first):** the Gemma-3 inference graph exports end-to-end via the new `Gemma3Model` class; fail with a documented blocker otherwise (record pass/fail in `02_code_plans/02`).
- Training one-step smoke for FunctionGemma on-device (requires the source-built ORT Training wheel; or document a blocker).
- Tool-grammar gate: if GenAI won Gate 0.1, exercise grammar-constrained generation; else confirm post-hoc JSON validation and document the limitation.

**Workflow (end-to-end)** — *(CHECKPOINT #37, device/manual — gated behind architecture gate 1 + differentiation gate 2)* train on the per-user action set → the model emits a structured tool call → `FunctionCallValidator` accepts it against the allowlist → `IntentBinder` dry-runs the allowlisted intent and returns the intended action **without executing** it (`willExecute=false`, no `startActivity`).

**Definition of done** — gate 1 passes as a registry entry (the Gemma-3 `ArchitectureSpec` has `inference_model_class = Gemma3Model`; the pre-built inference graph exports), validated end-to-end with the existing export/merge/manifest contracts and recorded in the support matrix; FunctionGemma fine-tunes on-device on a synthetic per-user action set (or a documented blocker); `FunctionCallValidator` accepts allowlisted calls and rejects off-allowlist/invalid ones; `IntentBinder` returns the intended allowlisted action in dry-run without executing; and the end-to-end demo runs only after gates 1 + 2 pass. No arbitrary model output is ever executed.

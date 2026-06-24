# FunctionGemma Architecture Gate & Android Intent Binding

**Priority #36 | Prerequisites: #10 (`01_code_plans/03`, inference engine decision), #6 (`01_code_plans/05`, export), #8 (`01_code_plans/01`, merge) | Blocks: —**

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

- **#10 / Gate 0.1**: grammar-constrained generation availability decides the validator strategy.
- **#6 / #8 / #12**: Gemma-3 must flow through the same export/merge/manifest contracts once the inference branch exists.
- **`02_code_plans/02`**: record the Gemma-3 gate result in the support matrix.

## Tests & smokes

- **Architecture export smoke (gate 1, first):** Gemma-3 inference graph exports; fail with a documented blocker otherwise.
- Training one-step smoke for FunctionGemma.
- Structured-output validation tests (valid calls pass).
- Invalid-action rejection tests (off-allowlist calls rejected).
- Android intent dry-run test (intended action returned, not executed).
- End-to-end demo gated behind architecture + differentiation passes.

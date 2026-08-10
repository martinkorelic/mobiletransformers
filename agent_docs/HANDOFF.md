# Handoff — current cycle

**Branch:** `restructure` · **Reset:** 2026-08-10 (end of the second 2026-08-10 cycle)

This file is **only what the current cycle is doing**. It is deliberately short and gets rewritten each
cycle.

> ## Read these first
>
> 1. **`IMPLEMENTATION_ORDER.md` → "Operational knowledge (permanent)"** — environment and profiles,
>    the gotchas that each cost a cycle, the ORT engine separation, the device workflow, the
>    layer-identity problem, the recurring failure shape, the gate results, and the recorded device
>    suite. Everything a cold agent needs *regardless of cycle* lives there, not here.
> 2. **`IMPLEMENTATION_ORDER.md` → "How to execute a plan"** — the implementer protocol. Follow it.
> 3. The per-plan **self-check blocks** in the same file — each records what is proven, how, and what
>    is left, with the evidence.

---

## Where the project stands

**35 / 37 plans done · 108 / 111 self-check boxes.** The two open plans are **#32 (licence-gated, not
work)** and **#37**, whose one open box is the differentiation demo.

**#33 and #36 both closed this cycle**, each with device evidence, and each found a real defect on the
way (below). **No known defect is outstanding.**

**Exactly one real blocker for v1.0: the licence.** CC-BY-NC-4.0 contradicts the consumable-AAR goal.
It is a rights-holders decision (both authors in `CITATION.cff`), not engineering; the four sites are
listed in `docs/RELEASE_CHECKLIST.md`. **Reconfirmed 2026-08-10 that the second author has NOT
agreed** — do not touch licence files, do not add SPDX headers, do not set the `pyproject.toml` license
expression. Flag it, nothing more. The other two open #32 boxes are the tag itself and the release gate
that needs it.

**Host gates (measured 2026-08-10, do not copy forward — re-measure):**

```
Python 514 passed / 11 skipped · C++ 39 · Kotlin JVM 227 · guard 7 · parity OK · uv lock --check clean
```

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformers:testDebugUnitTest
```

> The previous handoff recorded **guard 10**; the tree produces **7** (`tests/unit/test_guards.py`).
> That is the same drift the Kotlin figure showed. Read the real numbers from
> `build/test-results/testDebugUnitTest/*.xml` and from the pytest summary; never carry a figure over.

**Device suite** (S21 FE `SM-G990B` / Android 15 / arm64-v8a) — the result depends on which package is
installed, and the cache holds one:

| package on device | result |
| --- | --- |
| SmolLM2-135M `TRAIN=1 RAG=1` (decoder) | 15 / 15 pass, 798.0 s |
| all-MiniLM-L6-v2 `TASK=text-classification TRAIN=1` (encoder) | 19 tests, 0 failures, 16 skipped |

**The device currently holds the ENCODER package.** Re-push the decoder one before any generation work:
`make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct TRAIN=1`.

---

## What this cycle settled (do not redo these)

- **#36's device round-trip PASSES.** Export adapter factors from a live checkpoint on hardware →
  `federated serve` on the host → import back → the checkpoint moved *and survived a reload*. 120
  factors, **1.78 MiB** upload payload, every tensor byte-equal to the aggregate. Driven by
  `scripts/federated_round_device.sh` (`make device-federated`). New production code:
  `NativeCheckpointTensorStore`, `FederatedTrainingRepository`, `AdapterExchange`.
  **Two real defects, both invisible to either half alone:**
  1. **`startTraining` released the session on every exit path**, so nothing could read the checkpoint
     it had just trained. Fixed with `ORTTrainingConfig.keepSessionAtEnd` (default false → every
     existing caller unchanged).
  2. **A bounded round applied no update at all**: `optimizerStep` fires on
     `globalStep % gradAccumSteps == 0` and `gradAccumSteps` defaults to **4**, so a `maxSteps = 1`
     round uploads the global adapter back unchanged while every callback reports success.
  The second was hidden by a *weak assertion* — comparing the export against the aggregate counted
  clipping as training and reported a comfortable "60/120 moved". Comparing against the previous
  **export** (same clipping both sides) reported 0/120, which is what exposed it. Full write-up:
  `IMPLEMENTATION_ORDER.md` → "The #36 device round-trip".
- **#33's encoder legs ALL PASS**, including the Android smoke. The encoder TRAINING export had never
  run past `gen_artifacts`, and it failed closed on the first try: **the peft→ORT name rewrite had a
  decoder's own module baked into it** (`base_model.model.model.` → `backbone.model.` instead of the
  wrapper pair `base_model.model.` → `backbone.`). Identical for every decoder, a silent no-op for
  BERT. Fixed in all four mirrors + the eight open-coded Python copies, with encoder cases pinned in
  Python, C++ and Kotlin tests. `EncoderTrainStepDeviceTest`: 2 real steps, finite losses,
  **24/24 adapter factors moved**.
- **The decoder-only device suites now skip on an encoder package** (`DeviceModel.requireDecoder`,
  reading the task from `optimum_config.json`) instead of failing at a `generate()` the graph cannot
  do. An unknown task counts as a decoder, so older packages behave exactly as before.
- **Two standing debts closed.** `transformersVersion` (and `architectures`) are no longer dropped by
  the `--stages training` re-export — `_carry_forward_inference_provenance` recovers them from the
  `inference/optimum_config.json` the inference stage wrote, rather than re-deriving them under a
  profile that pins a *different* transformers. And the stage-path guard's two RAG sites are **zero**:
  `PackagePaths` grew `embeddingDatabase`/`embeddingTokenizer` and the retriever resolves through it.

---

## What is left, in order

### 1. #37 — the differentiation gate (self-check 3), the only engineering item

Four of the five differentiators are in place; what is missing is **the demo wired end to end**:
per-user action set → on-device fine-tune → validated tool call → dry-run intent.

For FunctionGemma specifically it stays blocked by `ort-training-local`'s `transformers==4.46.2` — a
**#2/#3 dependency decision, not a #37 one**, and floating that pin has broken `get_peft_model` before.
The demo can be run on a trainable decoder plus the Gemma-3 inference graph.

Concretely, what a demo has to solve, and why it was not attempted this cycle:

- it needs the **decoder** package back on the device (the cache holds the encoder now);
- the assertion is **convergence-dependent**: a validated call means the model emitted JSON the
  `FunctionCallValidator` accepts. A handful of LoRA steps on a 135M base will not do that, so the demo
  needs enough steps on a small memorisable set (`agent/mobile_actions.py` generates it from the app's
  own allowlist) — plan for a long device run, and use `gradAccumSteps = 1` (see above, or it trains
  nothing);
- a demo that "passes" by exercising the *refusal* path would show nothing. Assert on an accepted call
  reaching `IntentBinder.dryRun`.

### 2. Decisions that are not engineering

- **CI provisioning** — either vendor the native deps to a runner, or formally define "CI green" in
  `docs/RELEASE_CHECKLIST.md` as a recorded manual run. Today it is neither, which is the actual debt.
- **The licence** — see the top of this file. Nothing to do until the rights holders decide.

---

## Standing debts (not blockers, do not lose)

- **CI** — workflows are `workflow_dispatch`-only by choice and their native-dep provisioning is
  unresolved. "CI green" in `docs/RELEASE_CHECKLIST.md` means a recorded manual run, not a badge.
- **#34's multi-hour run** — a recorded DEBT, not work. Doze deferral, the notification's appearance,
  and multi-hour behaviour under Android 16's FGS quotas remain unproven. They are Android's behaviour
  rather than this library's, which is why the automated test drives chunks directly.
- **Variant naming** — `cpu-int4` legitimately ships an fp32 inference graph, declared via the measured
  `inferenceGraphPrecision`. Renaming is a wire-contract change, deliberately not done.
- **`inference/builder.py`** carries ~10 upstream-derived TODOs. Vendored GenAI builder; treat as
  upstream, and it is allow-listed in the architecture-literal guard for that reason.
- **`peft/mapping.py` has two decoder-shaped prefix sites left, deliberately untouched.** One converts
  `base_layer_name` into checkpoint space for the **MARS** mapping only (LoRA's mapping keeps raw keys),
  so generalising it would change MARS's `peft_mapping` keys for encoders — and
  `test_mars_encoder_transfer.py` passes today against the current shape. The other
  (`module_name.replace(...)` with the result discarded) is **dead**, and load-bearing by being dead:
  the loop below it works in raw space, so assigning it would break `relative_path`. Both want a
  deliberate look with the MARS tests in hand, not a sweep.
- **`trainableParameterCount` / `trainingParameterCount` never reach the manifest** — the training
  stage reports them and `build_manifest` drops them (both the decoder and encoder packages read
  `null`). `train/trainable_parameters.json` does carry the real number. Manifest field list is #14's.
- **The stage-path guard carries 2 allow-listed sites**, both legitimate producers that write a layout
  down once (`export/pipeline.py`, `ModelPackageInstaller.kt`). The RAG pair is gone.

## Cycle protocol

Follow `IMPLEMENTATION_ORDER.md` "How to execute a plan". For this repo specifically:

1. **Reset the profile before `make check`** — `uv sync --frozen --group dev --python 3.10`. More
   "broken repo" reports have come from a leftover export/training venv than from any real defect.
   `scripts/device_package.sh` leaves the tree on the training profile.
2. **Flip a `Done` box only when the self-check holds**, with the evidence (device, OS, ABI, date)
   inline.
3. **Assert across the seam, and check the assertion can FAIL.** This cycle's most useful finding came
   from tightening an assertion that was already passing: it was measuring clipping and calling it
   training. Both device legs now compare against a self-calibrating baseline for that reason.
4. **When you correct an earlier claim, correct it where the claim lives**, not only in a new entry.

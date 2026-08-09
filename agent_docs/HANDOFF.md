# Handoff — current cycle

**Branch:** `restructure` · **Reset:** 2026-08-09

This file is **only what the current cycle is doing**. It is deliberately short and gets rewritten each
cycle.

> ## Read these first
>
> 1. **`IMPLEMENTATION_ORDER.md` → "Operational knowledge (permanent)"** — environment and profiles,
>    the gotchas that each cost a cycle, the ORT engine separation, the device workflow, the
>    layer-identity problem, the recurring failure shape, and the final gate results. Everything a cold
>    agent needs *regardless of cycle* lives there, not here.
> 2. **`IMPLEMENTATION_ORDER.md` → "How to execute a plan"** — the implementer protocol. Follow it.
> 3. The per-plan **self-check blocks** in the same file — each records what is proven, how, and what
>    is left, with the evidence.
>
> The previous 2,958-line session log was consolidated into that permanent section on 2026-08-09. Its
> durable content is preserved there; the rest is in git history.

---

## Where the project stands

**31 / 37 plans done · 97 / 111 self-check boxes.** Of the 14 unticked, **11 are Tier-3**
(#33–#37), which the plan states never block v1.0.

**Exactly one real blocker for v1.0: the licence.** CC-BY-NC-4.0 contradicts the consumable-AAR goal.
It is a rights-holders decision (both authors in `CITATION.cff`), not engineering; the four sites to
change are listed in `docs/RELEASE_CHECKLIST.md`. Nothing technical waits behind it. The other two open
#32 boxes are the tag itself and the release gate that needs it.

**Host gates (keep these green):**

```
Python 459 passed / 10 skipped · C++ 22 · Kotlin JVM 153 · guard 5 · parity OK · uv lock --check clean
```

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformers:testDebugUnitTest
```

---

## This cycle's work, in order

### A. Device re-runs — **do this first, the package is already staged**

A real `TRAIN=1 RAG=1` package for `HuggingFaceTB/SmolLM2-135M-Instruct` was exported and pushed to the
connected S21 FE (`SM-G990B`) on 2026-08-09, **already carrying the new PEFT targets**. So this is one
command:

```bash
make device-test          # connectedDebugAndroidTest
```

Two things need re-proving, and both are consequences of this cycle's changes:

1. **The decoder suite under `q_proj`/`v_proj`.** PEFT targets now come from the architecture registry
   instead of a hardcoded `q_proj`/`k_proj`, so **which modules the decoder trains has changed**. The
   2026-08-08 device evidence predates it and is stale until this run. Host-side the change is clean
   (checkpoint size and trainable-parameter count are byte-identical to the shipped package), but only
   the device proves the merge/load legs still hold.
2. **`TrainConvergenceTest.trainingStartsFromPretrainedWeightsNotRandomOnes`**, rewritten this cycle
   and **never run on hardware**. Its old absolute threshold (`< 6.0`) measured a one-token CoLA
   objective it was never valid for and was recorded as a v1 blocker on bad arithmetic. It now compares
   initial loss on **coherent English** vs **random token soup** through the same preprocessor
   (`mini_recommendation`, whose supervised field is free text), requiring a 15% margin — a comparison
   an untrained model cannot fake.

Record the result the way this repo does: device model, Android version, ABI, date, in the plan's
self-check. **Tick nothing on a skip.**

If either fails, that is a real finding — do not relax the assertion to green. The whole point of the
rewrite was that the previous suite could be green and mean nothing.

### B. #33 encoder — the last two legs

The host chain is **proven** (export → `generate_artifacts` → 30 train steps → metric: loss −21.8%,
accuracy 0.25 → 1.00), pinned by `tests/integration/test_encoder_training_gate.py`. Two legs remain:

**B1. MARS on encoder attention layers** — self-check 3, *"verified, not assumed"*.

`src/mobiletransformers/peft/mars/model.py:66` hardcodes the decoder's module name:

```python
# TODO: Here we assume the attention layer is named "self_attn"
if isinstance(module, type(model.model.layers[0].self_attn)) and any_qkv:
```

BERT's attention module is `attention`, and `ArchitectureSpec.attention_module_name` **already carries
the right value per architecture** (`"attention"` on every encoder row) — the data exists and the
consumer ignores it. Route this lookup through the spec, then prove the transfer actually applies
rather than silently no-op'ing (assert the wrapped module count, not just that the call returned).
Note `peft/mars/model.py:80,142` also make positional assumptions about hidden-states ordering that an
encoder may not satisfy.

**B2. Android smoke over an encoder package.** `make device-package` currently exports a decoder plus
the RAG embedder; a classification package is a different variant shape (per-sequence labels, a
classification head, no KV cache). Decide whether the device consumes it through the existing
`ModelFeature.Training` path or a new feature, then run one ingest-free train step on device.

### C. #34 — charging-cycle training scheduler

The prerequisites are all in place and named:

- `training/TrainingJobManager.kt` — `TrainingJobSpec` is documented as *"the seam a future WorkManager
  `Worker` (#34) uses"*, and **no WorkManager dependency was added to the training path**.
- WorkManager itself is already a dependency and already used correctly for #21
  (`hub/PackageDownloadWorker.kt`, `CoroutineWorker` + `enqueueUniqueWork`) — copy that shape.
- `runtime/RuntimeCapabilities.kt:16` — `supportsScheduledTraining = false, // future (#34)` is the
  flag to flip.
- The two blockers the plan was written against are **closed**: `LinearLRScheduler.stateDict()` /
  `loadFromState()` exist (2026-08-07), and `LLMRepository.sessionLock` (a `Mutex`) plus
  `ORTTrainerNative.@Volatile cancelRequested` give the scheduler mutual exclusion and cooperative
  cancellation.

What is genuinely new: a foreground `CoroutineWorker` with `charging + idle + battery-not-low`
constraints and a mandatory persistent notification, checkpoint/resume across chunk boundaries and
process death, and — per the plan's own framing — **the measurement is the contribution**, so thermal
and energy traces are part of the deliverable, not a nice-to-have. Verify a multi-hour job survives
Android 16's tightened job quotas.

### D. #35 — the N-client simulation (one manual leg from done)

Everything else is done: `federated/adapter_record.py` derives its ordering from
`TrainableTensorCodec` (invents none), the byte serialization is frozen by
`tests/fixtures/federated_record.golden.bin`, `federated_average` + dropout are tested in the core env,
and the role vocabulary is decided and enforced.

The one open box is *"does an N-client Flower simulation aggregate adapter tensors and **improve the
metric**"* — `flower_sim.py::run_simulation`, runnable under the ort-training-local profile. **flwr is
deliberately out of the universal lock** (it downgrades protobuf/rich/typer and bumps mypy, breaking
`make check`): `pip install "flwr[simulation]"` out-of-band, like the ORT wheel.

Note the honest open question already recorded in the docs: v1 exchanges merged-weight-shaped tensors
(`aggregation_role="merged_base_plus_adapter"`), so per-round traffic is the size of the adapted
weights, not the rank-r adapters. That reads against the tier doc's "do not aggregate merged base
weights" and is a **v2 decision left explicitly open** — do not silently resolve it.

### E. #36 — federated Android client & gateway

**No code exists.** `grep exportTrainableTensors\|importTrainableTensors` over the whole tree returns
nothing; the only JNI exports are `exportModelForInference` and `mergeExportWeights` (24 `extern "C"`
entry points in `cpp/native-lib.cpp`). `RuntimeCapabilities.supportsAdapterTensorExport = false` is the
flag.

It is **hard-gated on #35 passing first** (its own self-check says so), and it is no longer gated on the
role-vocabulary question, which was decided. The codec must be **byte-identical to Python**, and the
golden already exists to prove it — mirror `FederatedAdapterRecord`'s pinned format (uint32 LE header
length + JSON header + codec-order payloads) rather than inventing a wire format. The privacy/security
gates (consent, TLS, auth, clipping/DP) are a **precondition to any real-user run**, not a follow-up.

### F. #37 — FunctionGemma architecture gate

Spike-gated and highest-risk; the plan says defer rather than let it sink the release.

The registry already carries a `Gemma3ForCausalLM` row bound to `Gemma3OnnxConfig` with
`inference_model_class=None` and the comment *"Gemma3 export is supported; inference is the
FunctionGemma gate (#37)"*. Note the caveat recorded beside it: the Gemma2/Gemma3 config bindings were
corrected but **never exercised end to end**, because the dotted paths resolve lazily. So gate one is
simply: run a real Gemma-3 export under the export profile and see whether the graph is right.

If it exports, the differentiators the plan requires are (a) on-device training, (b) personalised
per-user tool sets, (c) real Android intent binding — and it must **never execute raw model output**
(allowlist + dry-run + validated tool calls).

---

## Standing debts (not blockers, do not lose)

- **CI** — the workflows are `workflow_dispatch`-only by choice and their native-dep provisioning is
  unresolved. Explicitly out of scope this cycle. Until it changes, "CI green" in
  `docs/RELEASE_CHECKLIST.md` means a recorded manual run, not a badge.
- **Post-merge numerical correctness on device** is still unasserted. The export now gates parameter
  budget and train-vs-inference loss delta on the host (`artifacts/parameter_budget.py`,
  `artifacts/train_inference_parity.py`), but nothing checks the numbers *after* an on-device merge.
- **Variant naming** — `--quant` drives the training stage and the variant id only; the inference
  export does not quantize, so `cpu-int4` legitimately ships an fp32 inference graph. Now declared via
  the measured `inferenceGraphPrecision` in `optimum_config.json` and documented in
  `docs/MODEL_FORMAT.md`. Renaming the variant is a wire-contract change and was deliberately not done.
- **`inference/builder.py`** carries ~10 upstream-derived TODOs (quantize-weights, LoRA-on-inference-
  graph, GQA/SparseAttention). It is the vendored GenAI builder; treat as upstream, not project debt.
- **`peft/ablation/layer.py:726`** — merging is genuinely unimplemented for ablation PEFT.

## Cycle protocol

Follow `IMPLEMENTATION_ORDER.md` "How to execute a plan". In particular, for this repo specifically:

1. **Reset the profile before `make check`** — `uv sync --frozen --group dev --python 3.10`. More
   "broken repo" reports have come from a leftover export/training venv than from any real defect.
2. **Flip a `Done` box only when the self-check holds**, and record the evidence (device, OS, ABI,
   date) inline. The column has drifted three times by ticking on intent.
3. **Assert across the seam.** See "The recurring failure shape" in the permanent section — every
   expensive defect here has been two halves each verified alone.
4. **When you correct an earlier claim, correct it where the claim lives**, not only in a new entry.

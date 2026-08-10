# Handoff — current cycle

**Branch:** `restructure` · **Reset:** 2026-08-09 (end of the 2026-08-09 cycle)

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

**31 / 37 plans done · 102 / 111 self-check boxes.** Of the 9 unticked, **6 are Tier-3** (#33–#37),
which the plan states never block v1.0.

**Exactly one real blocker for v1.0: the licence.** CC-BY-NC-4.0 contradicts the consumable-AAR goal.
It is a rights-holders decision (both authors in `CITATION.cff`), not engineering; the four sites are
listed in `docs/RELEASE_CHECKLIST.md`. **The user confirmed 2026-08-09 that the second author has NOT
agreed** — do not touch licence files, do not add SPDX headers. Flag it, nothing more. The other two
open #32 boxes are the tag itself and the release gate that needs it.

**Host gates (keep these green):**

```
Python 501 passed / 11 skipped · C++ 38 · Kotlin JVM 219 · guard 10 · parity OK · uv lock --check clean
```

> **Kotlin JVM was recorded as 168 and the tree has never produced that.** Counted 2026-08-10: HEAD
> carries 153 `@Test` in `src/test`, plus 9 from the uncommitted scheduler work = **162** before this
> cycle. The 168 was drift, of the same kind the `Done` column has shown three times. The number above
> is measured (`build/test-results/testDebugUnitTest/*.xml`), not carried forward — re-measure it
> rather than copying it.

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformers:testDebugUnitTest
```

**Device suite** (S21 FE `SM-G990B` / Android 15 / arm64-v8a): see "Device suite" in the permanent
section of `IMPLEMENTATION_ORDER.md` for the recorded table.

> ## ✅ SETTLED 2026-08-10: the `transformers` fork regression
>
> The 13/14 device run's one failure is **root-caused and fixed in the consumer**; the fork stays.
>
> ```
> ConversationResetTest — inference step failed: ... Gather node.
> Name:'/model/Gather_5'  indices element out of data bounds, idx=5 must be within [-5,4]
> ```
>
> **It was a short attention mask, not a position-index error.** `/model/Gather_5` exists **only** in
> graphs exported by transformers >= 4.57 (proven by exporting the same model under both lines and
> diffing: identical 63 inputs / 61 outputs, but two new `/model/Gather_*` nodes). It gathers the
> *flattened* `attention_mask` at absolute positions derived from the cache length, so a mask shorter
> than `cached + new` indexes past the end. 4.46.2 had no such node and tolerated the short mask — the
> bookkeeping was already wrong, just unobserved.
>
> Host reproduction against the real graph: `mask = past+new` passes (**with the old position ids
> too**), `mask = past+new-1` fails with the exact device error.
>
> Fixed structurally rather than by counting: `InferenceSessionCache::pastSequenceLength()` is now the
> single authority on the cache length, `generateWithKVCache` fails closed naming all three numbers,
> `resetConversation()` finally clears the **native** cache, and the planning moved to the ORT-free
> `internal/runtime/GenerationInputs.kt` with 5 JVM tests pinning the turn boundary.
>
> Full write-up + the graph diff: `IMPLEMENTATION_ORDER.md` → "The #37 architecture gate" → RESOLVED.
>
> ⚠️ **Methodology note that cost the first attempt:** `uv pip install transformers==4.46.2` then
> `uv run --extra export ...` silently exports under **4.57.6** — `uv run` re-syncs to the lock first.
> Call `.venv/bin/python -m mobiletransformers.cli.main ...` directly for a pinned leg.
>
> **Confirmed on device 2026-08-10: 15 / 15 pass, 0 failures, 798.0 s** on a freshly exported
> `TRAIN=1 RAG=1` package. (15 rather than 14 because `ConversationResetTest` gained a second case;
> both are now non-vacuous — the old one asserted `tokenCount >= 0`, true of every outcome.)
>
> (The earlier two-failure run was package mutation, not regression — both tests pass here. That is
> settled; do not re-investigate it.)

---

## What this cycle settled (do not redo these)

- **The PEFT-target change is device-proven.** `q_proj,k_proj` → `q_proj,v_proj` needs no re-run.
- **`TrainConvergence(pretrained-weights)` passes on hardware** — first ever run.
- **MARS transfers to encoders** (#33 self-check 3). Six hardcoded decoder sites, fixed as registry
  data (`ArchitectureSpec.projection_names`). Decoder mapping byte-identical.
- **MARS's shared adapter actually trains now.** It was being quantized — and quantized means frozen —
  on *every* architecture, so MARS was silently degraded to LoRA with a frozen random down-projection.
  Rule: *a declared-trainable tensor is never quantized*. LoRA graphs byte-identical; a 30-step encoder
  run moves 12/12 shared tensors, loss 17.4% vs 5.6% before.
- **Training inputs bind by name** from the graph, not positionally (#33 B2 half). 8 googletests.
- **#34's scheduler works on hardware** — chunk 1 `globalStep 0→2`, chunk 2 `2→4`, LR schedule
  continued, thermal/energy trace captured.
- **#35's simulation passes** — 4 clients × 3 rounds, aggregated-adapter eval loss 8.7353 → 8.5258 →
  8.2579, payload 1.87 MB/round. Rank-r vocabulary implemented, byte golden regenerated, handoff-map
  schema **1.1** (additive `adapterDtypes`/`adapterShapes`).
- **#37's architecture gate passes** — `google/gemma-3-270m` exports to a #13-valid package. Needed an
  export-profile-only `transformers` fork (`>=4.50`) and a real registry fix (the row bound the
  **multimodal** `Gemma3OnnxConfig` to the **text-only** `Gemma3ForCausalLM`). **The fork has a cost
  that was not anticipated — see the box at the top of this file.**
- **Five new guards**, each verified to fail when its defect is restored: architecture literals outside
  the registry, registry-vs-Optimum config bindings, declared-vs-realized trainable tensors, the
  dependency fork collapsing, and the cumulative chunk budget.

---

## This cycle's work, in order

### A. Settle the `transformers` fork regression (see the box at the top)

Everything else in this list is optional until this is decided: it affects every package the repo
exports today.

### B. #33 — task-driven package shape (DECIDED, not started)

**Scope decision made: the general path, not the minimal one.** Stage layout, label contract, KV-cache
presence and merger requirements become declared per task on `TaskSpec` rather than decoder-assumed, so
a future objective (masked-LM, contrastive) is a registry row. Closes #33's last box on the way.

Starting points, in order:

1. `TaskSpec` already holds `label_shape`, `auto_model_class`, `model_init_kwargs`,
   `quantization_exclude_layers`, `trainer_wrapper_class`. Extend it with the **package** facts: which
   stages a task's package has, whether a KV cache exists, whether merger models are needed.
2. `export/pipeline.py`'s `_full_export` assumes the decoder shape (inference graph with KV cache +
   merger models + handoff map). Make those decisions read from the task spec.
3. **The native side is already done** — `cpp/training_inputs.h` binds by name and handles the
   encoder's `token_type_ids` + per-sequence `labels`. That half needs nothing.
4. The Kotlin gap is `ORTDataCurator`, which builds **per-token** labels; classification needs one
   label per sequence. This is the piece with no groundwork yet.
5. Then `make device-package` for `sentence-transformers/all-MiniLM-L6-v2` and one ingest-free train
   step on device.

The #33 plan's DoD permits "one device step **or** a documented blocker" — if the package shape proves
larger than it looks, record the blocker with evidence rather than half-migrating.

### C. #36 — federated Android client & gateway (fully ungated)

The vocabulary question is **answered and shipped**, so this is now plain work:

- Mirror `tests/federated/fixtures/federated_record.golden.bin` **byte for byte** in Kotlin. The format
  is a uint32 LE header length + UTF-8 JSON header (`tensors[].byteOffset`/`byteLength`) + concatenated
  raw LE payloads in codec order.
- Names/dtypes/shapes come from `weight_handoff_map.json` **schema 1.1** — `adapterDtypes` /
  `adapterShapes` — so the Kotlin codec resolves them from data. **Do not infer shapes from the rank.**
- Codec order is (entries sorted by canonical weight name) × `HandoffEntry.ADAPTER_ROLE_ORDER`.
- Tensor names are the **ORT checkpoint parameter** names (`backbone.model…lora_A.lora.weight`), not
  the PEFT module paths — that is the identity a client looks up.
- JNI `exportTrainableTensors`/`importTrainableTensors` do not exist yet; 24 `extern "C"` entry points
  are in `cpp/native-lib.cpp`.
- **Privacy/security gates (consent, TLS, auth, clipping/DP) are a precondition to any real-user run**,
  not a follow-up.
- A package exported before schema 1.1 **fails closed** with a message naming the re-export. That is
  intended; do not add a fallback to merged weights.

### D. G2 — one package-path resolver (not started)

The #35 simulation lost time to `<package>/train/` versus the manifest's `variants/<id>/train`, and the
#34 worker needed the *flat cache* layout — two different layouts, both currently built by string
concatenation. This is the layer-identity problem in a different namespace, and `cpp/layer_name.h`
already shows how that class is solved structurally.

The manifest declares `paths` and `select_variant()` returns them. What is missing is the rule that
**no consumer builds a stage path by appending a string**, plus a guard enforcing it.
`TrainingWorker.checkpointOnDisk` carries a comment marking the one place that currently knows the
cache layout. Do this next time a package consumer is touched.

### E. #37 — the remaining two self-checks

The architecture gate no longer blocks them: `FunctionCallValidator` (allowlist + `validationRules`),
`IntentBinder` (dry-run default, `startActivity` never called), and the differentiation gate.

**On-device *training* of Gemma-3 is still blocked** by `ort-training-local`'s `transformers==4.46.2`,
which is part of the ORT wheel's paired stack. That is a #2/#3 dependency decision, not a #37 one.
Floating a pin in that block has broken `get_peft_model` before.

### F. #34 — the multi-hour run (optional)

Chunk/checkpoint/resume and the traces are proven. What is **not**: Doze deferral, the notification's
appearance, and multi-hour behaviour under Android 16's tightened FGS quotas. Those are Android's
behaviour rather than this library's, which is why the automated test drives chunks directly. Run it
only if the release story needs it.

---

## Standing debts (not blockers, do not lose)

- **CI** — workflows are `workflow_dispatch`-only by choice and their native-dep provisioning is
  unresolved. "CI green" in `docs/RELEASE_CHECKLIST.md` means a recorded manual run, not a badge.
- ~~**Post-merge numerical correctness on device**~~ — **CLOSED 2026-08-10**. `PostMergeNumericsTest`
  passes on hardware and found three real bugs getting there (a dangling logits pointer on every
  forward pass, a null-session release that killed whole instrumentation runs, and a training-profile
  probe that answered the wrong question). See IMPLEMENTATION_ORDER "Post-merge numerical correctness".
- **Variant naming** — `cpu-int4` legitimately ships an fp32 inference graph, declared via the measured
  `inferenceGraphPrecision`. Renaming is a wire-contract change, deliberately not done.
- **#34's multi-hour run** is a recorded DEBT, not work: Doze deferral, the notification's appearance,
  and multi-hour behaviour under Android 16's FGS quotas remain unproven. They are Android's behaviour
  rather than this library's, which is why the automated test drives chunks directly. Run it only if
  the release story needs it.
- **`inference/builder.py`** carries ~10 upstream-derived TODOs. Vendored GenAI builder; treat as
  upstream, and it is allow-listed in the architecture-literal guard for that reason.
- ~~**`artifacts/validation.py`** hardcodes `replace("self_attn", "attn")`~~ — **CLOSED 2026-08-10.**
  `ONNXModelGenerator` takes an `architecture_spec` and resolves the spelling through
  `_attention_module_name()`; the entry is **removed** from the architecture-literal allow-list rather
  than widened, so the guard now enforces it.
- **`peft/ablation/layer.py`** — merging is genuinely unimplemented for ablation PEFT, and now
  **fails closed**. `get_delta_weight` used to `pass` (return `None`), and PEFT's `merge()` adds that
  delta to the base weight — so the merge completed "successfully" and wrote nothing, handing back an
  untrained model reported as merged. It raises `NotImplementedError` naming the alternative.

## Cycle protocol

Follow `IMPLEMENTATION_ORDER.md` "How to execute a plan". For this repo specifically:

1. **Reset the profile before `make check`** — `uv sync --frozen --group dev --python 3.10`. More
   "broken repo" reports have come from a leftover export/training venv than from any real defect.
   This cycle also installed `flwr` and `pytest` out-of-band; the reset removes them.
2. **Flip a `Done` box only when the self-check holds**, with the evidence (device, OS, ABI, date)
   inline. The column has drifted three times by ticking on intent.
3. **Assert across the seam.** Every expensive defect here has been two halves each verified alone —
   including three found this cycle.
4. **When you correct an earlier claim, correct it where the claim lives**, not only in a new entry.

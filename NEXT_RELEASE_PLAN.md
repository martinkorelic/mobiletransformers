# MobileTransformers — Next Release Research & Development Plan

**Goal of this release:** turn MobileTransformers from a research prototype into a *portable, HuggingFace-integrated, self-contained on-device fine-tuning + inference framework* that a developer can adopt with their own dataset and minimal friction — then leave it as a complete, citable artifact.

**Strategic framing (read first):** A direct competitor, *MobileFineTuner* (arXiv 2512.08211, Dec 2025), now exists. It is a real C++ trainer with a clean developer API, AAR build, and HF-style `from_pretrained` loading — but it is **training-only**: no inference, no RAG, no quantization, no novel PEFT method, and its phone validation is a *synthetic smoke test*, not a demonstrated end-to-end personalization run. **We do not win by building a leaner trainer (they are ahead there). We win by being the only *complete on-device personalization system*: MARS + quantization + train→merge→**infer**→RAG, with real demonstrated deployment, wrapped in first-class HuggingFace ergonomics.** Every priority below serves that positioning.

---

## How to read this plan

Work is grouped into **four tiers**. Build tiers in order: each tier depends on the foundations laid by the one before it. Within a tier, items are ordered by importance.

- **Tier 0 — Foundation & decisions** (must happen first; unblocks everything)
- **Tier 1 — The portable, HF-integrated core** (the headline of the release)
- **Tier 2 — Inference & retrieval as first-class, configurable subsystems**
- **Tier 3 — Reach extensions** (encoders, agentic/FunctionGemma) — do only after Tiers 0–2 are solid

A short **"Decision gate"** marks points where a research spike must conclude before committing engineering effort.

---

## Tier 0 — Foundation & Decisions

These are research/decision tasks that cost little but determine the architecture of everything else. **Do not start Tier 1 until these conclude.**

### 0.1 — ONNX Runtime GenAI re-evaluation *(highest-priority research spike)*

**Why now:** When this project started (early 2025) you evaluated `onnxruntime-genai` and chose a hand-rolled inference loop "for deeper control" over training/inference/RAG. That decision is now **18 months old and likely stale.** As of June 2026, `onnxruntime-genai` is at **v0.14.1**, actively released roughly monthly, and now ships: the full generate() loop, KV-cache management, search/sampling (greedy/beam/top-p/top-k), **grammar-constrained generation for tool calling**, and first-class support for Gemma, Qwen, Phi, SmolLM3, and multimodal models. This directly overlaps the manual inference + KV-cache code you maintain.

**Research questions:**
1. Can `onnxruntime-genai`'s `Generator` API **consume a model whose weights were just produced by on-device training + merge**, or does it assume an offline-built `genai_config.json` artifact? (This is the crux — your differentiator is *train-then-immediately-infer on device*.)
2. Does its model-loading path support the **memory-mapping** scheme you rely on (shipping the inference graph from prep, linking it to freshly trained weights), or would adopting it reintroduce the on-device graph-export OOM you originally solved?
3. What is the Android/JNI build story for the GenAI layer specifically (not just core ORT)? Binary size, EP support, NNAPI/XNNPACK behavior.
4. Does delegating generation to GenAI **free you from maintaining** the manual KV-cache + sampling code, reducing surface area and bugs?

**Decision gate → choose one:**
- **(A) Adopt GenAI for the inference half**, keep your training + merge + memory-mapping as the unique front half. Outcome: you shed maintenance of generation internals, gain robust sampling/tool-grammar for free, and stay differentiated on the *training-to-inference handoff*. **Recommended default unless the spike shows GenAI can't consume on-device-merged weights.**
- **(B) Keep the manual loop**, but document *why* (control over the train→merge→infer handoff and memory mapping that GenAI's offline-artifact assumption breaks). This is defensible only if (1) or (2) above fail.

**Deliverable:** a 1–2 page spike report answering Q1–Q4, a working proof-of-concept of GenAI loading an on-device-merged model (or a documented failure), and a committed A/B decision. Everything in Tier 2 depends on this.

### 0.2 — Memory-mapping integration research

Tightly coupled to 0.1. Document precisely how your "ship the inference graph from prep, memory-map trained weights at `InferenceSession` instantiation" mechanism interacts with whichever inference path 0.1 selects. If GenAI is adopted, determine whether its weight-loading can be pointed at your memory-mapped, freshly-merged weight blob. **This is the single most novel piece of engineering in the framework — protect it.**

### 0.3 — Optimum / ONNX-export toolchain migration *(CRITICAL BLOCKER — research first)*

**This is the highest-urgency item in the plan.** The HuggingFace export toolchain your preparation phase depends on has been restructured, and one change directly threatens the training pipeline.

**What changed (verified June 2026):**
1. **ONNX integration split out of `optimum` into a new `optimum-onnx` package.** The main `optimum` repo now carries an explicit banner: *"ONNX integration was moved to optimum-onnx."* Your old `pip install optimum` + `from optimum.onnxruntime import ...` flow is deprecated. The new install is `pip install --upgrade --upgrade-strategy eager optimum[onnx]`, and export now lives in `huggingface/optimum-onnx` (with `optimum.exporters.main_export` as the recommended programmatic entrypoint and `optimum-cli export onnx` as the CLI).
2. **🚨 ONNX Runtime *Training* is officially deprecated and slated for full removal in Optimum v2.** The release notes state: *"ONNX Runtime Training officially deprecated"* and *"Deprecated support for TFLite, BetterTransformer, and ONNXRuntime-Training; these integrations will be fully removed in v2."*

**Why this is a blocker, not a chore:** Your framework's entire on-device *training* capability is built on ONNX Runtime Training. The upstream HuggingFace tooling that prepares those training artifacts (forward+backward graphs, optimizer, checkpoint state) is being removed. If you pin to old versions you freeze the framework on a dead dependency; if you upgrade naively, the training export may stop working. **This must be resolved before any other engineering, because it determines whether the train-side pipeline even has a supported toolchain.**

**Research questions:**
1. Does `optimum-onnx` still expose the **training-graph export** (forward + backward + optimizer + checkpoint) your preparation phase needs, or only inference export? If only inference, where does training-artifact generation now live?
2. Is **ONNX Runtime Training itself** (the `onnxruntime-training` runtime package, separate from Optimum's wrapper) still maintained and shippable for Android, independent of Optimum's deprecation of its *wrapper*? (Optimum deprecating its convenience layer does not necessarily kill the underlying `onnxruntime-training` build — confirm the distinction.)
3. What is the **minimum viable export path** post-migration: `optimum-cli export onnx` + manual training-artifact construction? Direct `onnxruntime.training` APIs without Optimum? A pinned legacy Optimum version as a stopgap?
4. Pin a **known-good dependency set** (exact versions of `optimum`, `optimum-onnx`, `onnxruntime`, `onnxruntime-training`, `transformers`, `torch`) under which the full train→merge→infer export reproducibly works today.

**Decision gate → choose a training-export strategy:**
- **(A) Migrate to `optimum-onnx`** for inference export and **call `onnxruntime.training` directly** for training-artifact generation, decoupling from Optimum's deprecated training wrapper. *Likely the durable path.*
- **(B) Pin a legacy Optimum version** as a documented stopgap while (A) is built — buys time but accrues technical debt and a removal deadline at Optimum v2.
- **(C) Reduce dependence on Optimum entirely** by generating training/inference graphs through your own export layer over `torch.onnx` + `onnxruntime.training`, treating Optimum as optional convenience rather than a hard dependency. *Most work, most control, most future-proof.*

**Deliverable:** a spike report answering Q1–Q4, a **pinned, reproducible `requirements.txt`/`environment.yml`** that builds the full pipeline today, and a committed migration strategy (A/B/C). Tie the pinned set into the packaging work in 0.4 and the CI in the cross-cutting section.

### 0.4 — Dependency hygiene & easy install

Decide the adoption surface before building APIs around it, and make installation frictionless on both sides (Python preparation + Android runtime):

**Python preparation side (easy install):**
- Ship a **pinned, tested dependency set** (from 0.3) as the canonical install. The current implicit-dependency situation (bare `optimum`, unversioned `onnxruntime`, etc.) breaks now that the Optimum ONNX split has landed.
- Provide a proper **packaged installer**: a `pyproject.toml` with extras (`mobiletransformers[export]`, `mobiletransformers[train]`) so a user runs one `pip install` rather than hand-assembling onnxruntime/optimum/transformers versions.
- Guard against the classic ORT footgun: `onnxruntime` and `onnxruntime-gpu` conflict (Optimum's own docs warn to `pip uninstall onnxruntime` before installing the GPU variant). Encode the correct combination in the extras so users can't get a broken mix.
- Consider a **lockfile** (`uv`/`pip-tools`) so the export environment is byte-for-byte reproducible — important because the export toolchain is now split across `optimum` + `optimum-onnx` + `onnxruntime-training`.

**Android runtime side (easy install):**
- **Android dependency:** can the framework be consumed as a Maven/Gradle **AAR** (the way MobileFineTuner already ships)? Research the ORT-training AAR build constraints. This is table stakes for "portable" — without it, adoption friction stays high.
- Provide `scripts/android/build_aar.sh` and a local-Maven publish path so a consumer app adds one Gradle line.

**Release & licensing:**
- **Versioning & release:** introduce semantic version tags and a `CHANGELOG`. The repo currently has 0 tags; a "complete, leave-it-as-is" release needs a tagged `v1.0`.
- **License check:** the repo is currently CC-BY-NC-4.0. **Non-commercial licensing will block real-world adoption and HF ecosystem uptake.** Decide consciously whether to relicense (e.g. Apache-2.0, as MobileFineTuner uses) if "de-facto framework" is the goal. This is a values/strategy decision, not an engineering one — flag it to all authors.

---

## Tier 1 — The Portable, HuggingFace-Integrated Core

**This is the headline of the release.** The thesis: *a developer brings only their own dataset; everything else — model pull, export, train, merge — is handled by the framework with HF-native ergonomics.*

### 1.1 — Unified, HuggingFace-style developer API

Adopt the API vocabulary developers already know from `transformers` / `optimum`, so the learning curve is near zero. Target a surface like:

```kotlin
val model = MobileTransformers.fromPretrained("Qwen/Qwen2-0.5B")   // pulls + prepares
model.applyPeft(MARS.OPT1(rank = 8))                                // our novel method
model.train(localDataset, TrainConfig(epochs = 4, batchSize = 4))
model.merge()                                                       // on-device merge
val out = model.generate("…")                                      // immediate inference
```

**Design principles:**
- Mirror HF naming (`fromPretrained`, `applyPeft`, `generate`) so the mental model transfers.
- Expose **MARS** as a first-class PEFT option — this is the differentiator MobileFineTuner structurally lacks. `LoRA`, `MARS.OPT0`, `MARS.OPT1`, quantized variants all selectable here.
- High-level Kotlin facade over the existing C++/JNI engine; do not expose ONNX/JNI internals to the app developer.

**Deliverable:** a documented public API boundary (`PUBLIC_API.md`-style), a quick-start that goes from clone to a running fine-tune in <10 minutes, and a sample consumer app.

### 1.2 — HuggingFace Hub → device model pull

The "bring only your dataset" promise requires the *model* to arrive automatically. Research and implement direct Hub download to device:
- Mirror the established `onnxruntime-genai` convention (HF-hosted, pre-exported ONNX model directories pulled via the Hub) so you interoperate with the existing ONNX model ecosystem rather than inventing a parallel one.
- Resolve a HF repo id → download `config.json`, tokenizer, weights (SafeTensors / ONNX) → cache on device → prepare for training.
- Handle the **export gap**: most HF models are PyTorch, not the training-ready ONNX artifacts your framework needs. Decide where conversion happens (see 1.3).

### 1.3 — Standardized "MobileTransformers-ready" model format on the Hub

**This is the key insight that makes the framework self-serve.** Define and publish a canonical on-device-trainable model package, and host a collection of them under a HF org/account, so users pull a ready model and only supply data.

**Define the package contract** (what a "MobileTransformers-ready" HF repo contains):
- The training ONNX graph (forward + backward) and the **pre-built inference graph** (so on-device export/OOM is avoided — your core trick).
- Quantized frozen base layers.
- Tokenizer + your extended configuration file (scheduler, EP, threads, PEFT targets).
- A manifest declaring supported PEFT methods, ranks, and quantization formats.

**Publish a starter model zoo** on HF (e.g. `mobiletransformers/Qwen2-0.5B-mobile`, `…/SmolLM2-360M-mobile`, `…/TinyLlama-1.1B-mobile`) so a developer can `fromPretrained` one and immediately train on their own data with zero export work.

**Bonus (closes the ecosystem loop):** support **pushing fine-tuned adapters back to the Hub** in PEFT-compatible format. Pull base → train on device → publish adapter. This makes you a *participant* in the HF ecosystem, not just a consumer — a strong "de-facto" signal.

### 1.4 — Streamlined offline-model build/export pipeline

The export step (PyTorch HF model → MobileTransformers-ready ONNX package) is currently the hardest part of onboarding. Make it one command.
- Provide a `Makefile` / CLI: `make export MODEL=Qwen/Qwen2-0.5B PEFT=mars-opt1 QUANT=int8` → produces the Tier-1.3 package, ready to upload to the Hub or push to device.
- Wrap the existing Python preparation phase (Optimum conversion, quantization, graph generation) behind this single entrypoint.
- Document the supported-architecture list and the conversion contract (which HF model families convert cleanly).

**Deliverable:** reproducible `make`-driven export; CI that builds the starter zoo models from scratch and validates them end-to-end.

---

## Tier 2 — Inference & Retrieval as First-Class Subsystems

Depends on the Tier-0.1 inference decision. This is where you out-scope MobileFineTuner, which has **no inference layer at all**.

### 2.1 — Inference subsystem (per Tier-0.1 decision)

- If **GenAI adopted (Path A):** integrate `onnxruntime-genai`'s `Generator` for the generation loop, sampling, and KV-cache, fed by your on-device-merged, memory-mapped weights. Retire the hand-rolled loop where GenAI covers it. Net: less code to maintain, more robust generation, tool-calling grammar available for Tier 3.
- If **manual retained (Path B):** consolidate and document the existing KV-cache + sampling implementation as a deliberate, justified component.

Either way: the train→merge→**infer** handoff on one device, with no server, remains the headline capability.

### 2.2 — Configurable RAG / vector-database layer

Your RAG + ObjectBox vector store already exists; this release makes it **easy to configure and swap**, which is what "leave it as a complete framework" requires.
- Abstract the vector store behind an interface so ObjectBox is the default but alternatives are pluggable.
- Expose a simple config surface: embedding model choice, chunking, top-k, similarity metric, precompute-vs-dynamic indexing.
- Document a "bring your own documents" path that mirrors the "bring your own dataset" training path — symmetry users will appreciate.
- Let the embedding model itself be pulled from the Hub (ties back to 1.2).

**Deliverable:** a `RagConfig` object, a documented default, and one worked example (local docs → embedded → retrieved → grounded generation) on device.

### 2.3 — HuggingFace API-surface alignment audit

Cross-cutting task: audit every public entrypoint against the equivalent HF/`optimum` API and align naming, defaults, and config object shapes. The goal is that a developer fluent in `transformers` can use MobileTransformers without re-learning concepts. Produce a short "HF ↔ MobileTransformers" mapping table in the docs.

---

## Tier 3 — Reach Extensions

Do these **only after Tiers 0–2 are solid.** Each broadens the claim but adds risk/scope. Gate each behind a spike.

### 3.1 — Encoder-model support (best value/effort in this tier)

Add support for smaller **encoder** models and non-generative tasks (text classification, embedding similarity).
- **Why it's cheap:** MARS's shared-projection design ports almost directly (the projections exist in encoders too); encoders do a single forward pass, so you can **delete the fragile autoregressive KV-cache path** for these tasks rather than maintain it.
- **Why it's valuable:** broadens the claim from "decoder LLM fine-tuner" to "general on-device transformer PEFT framework," and classification/embedding are natural, well-metricized IoT/mobile tasks (on-device intent classification, semantic search). MobileFineTuner is decoder-only too — this is open ground.
- **Spike first:** confirm your ONNX export + training graph generation handles an encoder architecture (e.g. a small BERT/MiniLM) end-to-end.

### 3.2 — Sleep / charging-cycle training *(systems realization, not algorithm)*

Use Android `WorkManager` to run fine-tuning during overnight charge/idle windows.
- **Frame as a measured systems capability**, not an ML contribution: the replay/continual-learning *algorithm* literature is crowded; your contribution is the *robust on-device realization* — checkpointing across interrupted charge cycles, constraint-gated scheduling, measured energy/thermal behavior.
- **Constraints to research:** `WorkManager` requires `charging + idle + battery-not-low` constraints; long-running training needs a **foreground service with a mandatory persistent notification**; **Android 16 tightened job quotas** — verify a multi-hour training job survives. Your existing checkpoint/resume logic is the enabler.
- **Deliverable:** a scheduler that trains across charge cycles with measured energy/thermal traces. The *measurement* is the contribution.

### 3.3 — Agentic personalization: FunctionGemma + Mobile Actions *(highest visibility, highest risk — gate hard)*

On-device fine-tuning of a **function-calling** model so the agent calls *personalized* tools/intents, all trained and run locally.

**The opportunity:** FunctionGemma (Google, Dec 2025, 270M, Gemma-3 arch) is purpose-built for on-device function calling, ships with a public **Mobile Actions** HF dataset, and fine-tuning lifts its accuracy from ~58% → ~85%. Your angle: **on-device** fine-tuning + your full pipeline, binding outputs to **real Android intents**.

**The risks (gate each before committing):**
1. **Architecture gate:** FunctionGemma is **Gemma-3**. Confirm your ONNX export + training-graph pipeline handles Gemma-3 (270M) end-to-end. If it doesn't export cleanly, this is a multi-week detour. **Spike this first; everything else here depends on it.**
2. **Differentiation gate:** Google already ships the canonical path (fine-tune via HF/Unsloth → deploy via **LiteRT-LM** in the AI Edge Gallery). Running their tutorial on your framework is not a contribution. Your defensible differentiators must be: (a) on-device *training* (Google's recipe trains off-device), (b) **personalized** tool sets per user (build a synthetic per-user mobile-actions dataset, mirroring how you built MiniPersonalQA), and/or (c) real Android **intent binding** so calls actually execute.
3. **Tool-grammar leverage:** if Tier-0.1 adopted `onnxruntime-genai`, its **grammar-constrained generation for tool calling** is directly useful here — another reason Path A pays off.

**Deliverable (if both gates pass):** FunctionGemma fine-tuned on-device on a personalized action set, emitting validated function calls bound to real Android intents. **If the architecture gate fails, defer to future work** — do not let it sink the release.

---

## Cross-Cutting: Repository & Build Modernization

Run alongside all tiers; required for a "complete, leave-it-as-is" release.

- **One-command build/export/run.** `Makefile` targets for: build, export-model, run-smoke, build-aar, publish-local. Lower the clone-to-running bar to minutes.
- **CI that builds from scratch.** GitHub/GitLab CI that compiles the core, builds the starter zoo models, and runs an end-to-end smoke (train→merge→infer) so the "it actually works" claim is continuously verified — this is your edge over a smoke-test-only competitor.
- **Documentation set:** `PUBLIC_API.md`, `MODEL_FORMAT.md` (the Tier-1.3 contract), `ANDROID_SDK.md`, `RAG.md`, an `ARCHITECTURE.md`, and a compatibility matrix (model family × PEFT method × quantization × task × device).
- **Versioned `v1.0` release** with `CHANGELOG`, tags, and the licensing decision from 0.3 resolved.

---

## Explicit Non-Goals (scope discipline)

- **GPU/NPU training.** Out of scope, by design. Justification: all three mobile accelerator stacks (NNAPI, XNNPACK, QNN) are **inference-only** — none exposes backward/optimizer kernels. Enabling GPU training means authoring a new execution provider with backward + optimizer kernels per vendor (Mali/Adreno/Xclipse) in OpenCL/Vulkan: a multi-month, per-vendor, fragile effort with a likely "works but not faster" outcome due to GPU↔CPU activation-transfer cost. **Documented as future work, not a missing feature.**
- **Multimodal (text+image) training.** Tempting but high-risk; defer beyond this release. Encoder *text* tasks (3.1) are the sane first step toward broader modality.
- **Beating MobileFineTuner on raw trainer engineering.** We do not compete on "leaner C++ trainer." We compete on *complete personalization system + MARS + HF integration*.

---

## Recommended Sequence (one-glance)

1. **0.3 Optimum / ORT-Training migration** (the critical blocker — the training toolchain is being removed upstream) → pinned, reproducible export environment locked **first**.
2. **0.1 ONNX Runtime GenAI spike** + **0.2 memory-mapping** + **0.4 packaging/license decision** → architecture and distribution locked.
3. **1.1 HF-style API** + **1.4 one-command export** → developer can drive the framework.
4. **1.2 Hub pull** + **1.3 model-format + starter zoo** → "bring only your dataset" becomes real.
5. **2.1 inference** (per 0.1) + **2.2 configurable RAG** + **2.3 HF API audit** → complete, configurable system.
6. **3.1 encoders** (cheap reach) → broadened claim.
7. **3.2 charging-cycle training** (measured) and **3.3 FunctionGemma** (gated) → headline extensions, risk-managed.
8. **Cross-cutting build/CI/docs/versioning** throughout → ship `v1.0`, leave it complete.

---

## What "complete" looks like at release

A developer, starting from nothing, can:
1. Add MobileTransformers as an Android dependency (AAR).
2. `fromPretrained("…")` to pull a ready model from the Hub.
3. Point it at *their own* dataset.
4. Fine-tune on device with MARS, fully offline.
5. Merge, then immediately generate — with optional, configurable RAG.
6. Optionally push the adapter back to the Hub.

…with no server, no emulation, no manual export, and no ML-framework expertise required. That is the artifact to ship and leave standing.

# Implementation Order — Code Plans Index

This index orders every feature code-plan under `agent_docs/00_code_plans/` … `agent_docs/05_code_plans/` from **most foundational first**. Implement in this order; each plan file repeats its own Prerequisites / Blocks header.

The high-level "what & why" lives in the six tier docs (`00_repository_restructure_plan.md`, `01_tier0_foundation_decisions.md`, `02_tier1_hf_integrated_core.md`, `03_tier2_inference_and_rag.md`, `04_tier3_reach_extensions.md`, `05_cross_cutting_release_modernization.md`). These code plans are the "how" — concrete enough for another agent to implement.

**Tracking:** the `Done` column is the live checklist — flip `[ ]` → `[x]` only when the plan's **Definition of done** (in its `## Tests & acceptance` section) holds *and* the matching block in [Per-plan completion self-checks](#per-plan-completion-self-checks) below all answer "yes".

## Canonical decisions every plan inherits

- **Dual inference engine over one package.** The same exported package + same Android-cache folder is consumable by **both** the native ORT engine and the ONNX Runtime GenAI engine. GenAI is a *selectable* engine (Native is the default/guaranteed path), not a separate package. See `01_code_plans/03_inference_engine_abstraction_native_and_genai.md`.
- **Unified weight handoff via external initializers.** Trainable/merged weights are ONNX **external initializers, one-file-per-tensor**, in `<cacheDir>/<model>/inference/`. The frozen quantized base is a separate immutable external blob. On-device merge overwrites the per-tensor files (atomic rename + checksum). No graph rewrite, no weights-as-inputs, no genai fork. See `01_code_plans/01_unified_merger_and_external_data_export.md`.
- **`weight_handoff_map.json` is the single source of truth** that replaces hard-coded name rewrites (`weight_merger.cpp:904`). Schema lives in `00_code_plans/07_weight_handoff_map_and_tensor_codec.md`.
- **Dependency profile isolation** (uv groups/extras): `onnxruntime-training` (source-built, provides `onnxruntime`), public `onnxruntime`, `onnxruntime-genai`, and `optimum-onnx[onnxruntime]` collide on the `onnxruntime` import and must never share one env. See `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`.
- **Apache-2.0** is the target framework license (model weights keep upstream licenses).
- **External-initializer handoff is the only merge path.** The legacy `inference/merged/` subdirectory is retired; merged weights are flat per-tensor external initializers in `inference/`. `HandoffMode` enumerates `external_initializer` / `model_input` / `adapter`, but **only `external_initializer` is supported in v1** — the others are registry stubs that fail closed with "not supported in this version". Tier 2 consumes the existing `ModelRuntime` engine boundary and must not define a competing `InferenceEngine`. See `03_code_plans/01`.
- **`FederatedAdapterRecord` derives from `TrainableTensorCodec`.** Federated exchange (Tier 3) is a thin wrapper over `weight_handoff_map.json` + the codec from `00_code_plans/07`; it never invents its own tensor ordering, and its `adapterFormatVersion` tracks the handoff-map `schemaVersion`. See `04_code_plans/03`.
- **Registries replace hardcoded dispatch.** PEFT methods, model architectures, and merger variants are declared in data-driven registries; no `if x == "lora"/elif "mars"` or `architectures[0] == "..."` chains in business logic. Adding a method/architecture/merger is a registry entry + an enum member, not a new `elif`. The same pattern also owns **task-type, execution-provider, document-loader (RAG), and export-front-door** registries, added as their consumers land. Owned by `00_code_plans/09_typed_models_enums_and_registries.md`.
- **Pydantic v2 is the typed config contract.** Every tunable config object is a Pydantic model with camelCase aliases; cross-boundary JSON is `model_dump(by_alias=True)` + a generated `schemas/*.schema.json` the Kotlin parser and C++ loader validate against (fail-closed). Secrets stay in `Settings` (`00_code_plans/02`). `pydantic>=2` is a core dependency (`00_code_plans/03`). Owned by `00_code_plans/09`.
- **Enums own every closed string set**, mirrored Python (`config/constants.py`) ↔ Kotlin (`.../constants/*.kt`): sampling method, scheduler type, execution provider, core/memory config id, search type, quantization type, PEFT method, task type, handoff mode, merger variant. Owned by `00_code_plans/09`.
- **Full merger unification.** A single parameterized `build_merger_model(MergerSpec)` replaces `create_lora_merger_model{,_2}` / `create_mars_merger_model{,_2}` (the "merger 1/2" duplication); C++ `get_merger_type`/`run_merger_model` resolve the spec from the handoff map + merger registry instead of branching on `"lora_q"/"mars_q"`. Owned by `00_code_plans/09`, wired by `01_code_plans/01`.
- **Schema versioning is uniform and forward-compatible.** Every JSON contract (`mobiletransformers_manifest.json`, `weight_handoff_map.json`, `model_support_matrix.json`, `FederatedAdapterRecord`, generated `schemas/*.schema.json`) carries `schemaVersion` (`MAJOR.MINOR`) + `minReaderVersion`. Readers **preserve unknown fields** (additive minor bumps are non-breaking) and **fail closed with a "needs newer SDK" message** only when `major` exceeds support or `minReaderVersion` is unmet. One `check_compat()` helper mirrored Python↔Kotlin. Owned by `00_code_plans/07`; consumed by `00/06`, `02/02`, `02/03`, `04_code_plans/03`.
- **Cross-language parity is CI-enforced, not manual.** Pydantic models (`config/models.py`) + enums (`config/constants.py`) are the single source of truth; `schemas/*.schema.json` and a golden `enums.json` are generated from them, and a CI `parity` job fails if the checked-in Kotlin/C++ mirrors drift. Owned by `00_code_plans/09`, gated by `05_code_plans/02`.
- **The Python library API is a declared, SemVer-governed surface** (`mobiletransformers.__all__`), peer to the Kotlin facade and the CLI — not just the CLI. Owned by `00_code_plans/10`, documented by `05_code_plans/04`, versioned by `05_code_plans/05`.
- **Tests follow a fixed taxonomy in every plan**: **unit** (automated; `pytest` / Android JVM, incl. a compile/assemble check) · **integration** (automated; run → expected output) · **manual** (user-run; long/intensive or device-specific) · a **Definition of done**. End-to-end **workflow** tests live only on feature-complete checkpoint plans (#15, #17, #19, #21, #27, #30, #32, #34, #35, #37). Android device/emulator behaviour is always manual (user-run); CI runs only compile + `assembleDebug`.

## Global order

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [ ] | 1 | `00_code_plans/01_python_package_and_uv_scaffolding.md` | Unblocks all Python work | — |
| [ ] | 2 | `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md` | Lock the toolchain before building on it (incl. `pydantic>=2` core) | 1 |
| [ ] | 3 | `01_code_plans/06_source_built_ort_training_pipeline.md` | Prove the train toolchain is alive (`generate_artifacts`) | 2 |
| [ ] | 4 | `00_code_plans/02_config_layering_settings_constants.md` | Three config layers + secrets; needed by export/CLI | 1 |
| [ ] | 5 | `00_code_plans/10_python_code_quality_and_module_health.md` | **Conventions, typing, lint, shared logging/exceptions + the monolith-decomposition strategy** — established before registries/merger/builders so they inherit it | 1, 2, 4 |
| [ ] | 6 | `00_code_plans/09_typed_models_enums_and_registries.md` | **Typed config + enums + PEFT/arch/merger registries** — kills hardcoded dispatch before builders are written | 2, 4, 5 |
| [ ] | 7 | `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md` | Inference-export front door (arch via registry) | 2, 4, 6 |
| [ ] | 8 | `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` | Data contract every later piece reads (codec consumes registries) | 4, 6 |
| [ ] | 9 | `01_code_plans/01_unified_merger_and_external_data_export.md` | **Dual-engine core** + full merger unification | 6, 7, 8 |
| [ ] | 10 | `01_code_plans/02_genai_external_data_swap_spike.md` | Feeds Gate 0.1 | 9 |
| [ ] | 11 | `01_code_plans/03_inference_engine_abstraction_native_and_genai.md` | Engine selection | 10 |
| [ ] | 12 | `01_code_plans/04_memory_mapping_experiments.md` | Optimizes 9–11 (non-blocking) | 9 |
| [ ] | 13 | `00_code_plans/06_manifest_first_package_and_cache_bridge.md` | Package contract | 8, 9 |
| [ ] | 14 | `02_code_plans/03_hub_model_package_format.md` | Hub repo shape | 13 |
| [ ] | 15 | `02_code_plans/05_one_command_export_cli.md` | Wraps 7–9 + 13 *(checkpoint: export E2E)* | 7, 9, 13 |
| [ ] | 16 | `00_code_plans/04_android_gradle_rename_migration.md` | Isolated, verified rename | — |
| [ ] | 17 | `00_code_plans/05_android_facade_foundation.md` | Public SDK facade *(checkpoint: load→generate)* | 13, 16 |
| [ ] | 18 | `00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md` | Job/progress/checkpoint API | 17 |
| [ ] | 19 | `02_code_plans/01_hf_style_kotlin_facade.md` | `fromPretrained`/train/merge/generate *(checkpoint: train→merge→generate)* | 11, 17 |
| [ ] | 20 | `02_code_plans/02_optimum_support_matrix.md` | Reporting layer | 7 |
| [ ] | 21 | `02_code_plans/04_hub_pull_and_cache_flow.md` | Python pull first, Android downloader next *(checkpoint: pull→load)* | 13, 14 |
| [ ] | 22 | `02_code_plans/06_adapter_pushback.md` | Last Tier-1 piece | 9, 14 |

### Tier 2 — Inference & RAG (Phase 7)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [ ] | 23 | `03_code_plans/01_inference_handoff_alignment_and_native_hardening.md` | Wire Native into `ModelRuntime`, retire `inference/merged/` | 11, 9, 8 |
| [ ] | 24 | `03_code_plans/02_sampling_and_streaming_public_config.md` | HF-aligned generation config (enum-typed) + callback parity | 23, 19 |
| [ ] | 25 | `03_code_plans/03_vector_store_boundary_and_inmemory.md` | Testable `VectorStore`; `SearchType` enum; dynamic dimension registry | 17 |
| [ ] | 26 | `03_code_plans/04_rag_ingestion_and_chunking.md` | Implement `ingestData()` | 25, 23 |
| [ ] | 27 | `03_code_plans/05_rag_config_and_grounded_generation.md` | Public `RagConfig` + grounded flow *(checkpoint: ingest→retrieve→generate)* | 26, 24, 21 |

### Cross-cutting — Release modernization (Phase 8; runs alongside, gated)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [ ] | 28 | `05_code_plans/01_makefile_and_cli_entrypoints.md` | One-command path | 1, 15, 2 |
| [ ] | 29 | `05_code_plans/02_ci_staged_pipeline.md` | Standing "it works" proof (incl. lint/typecheck/parity gates) | 28, 1, 3 |
| [ ] | 30 | `05_code_plans/03_aar_maven_publication.md` | Portable Android consumption *(checkpoint: consumer app builds)* | 16, 28 |
| [ ] | 31 | `05_code_plans/04_docs_set_and_compatibility_matrix.md` | Public docs + registry-driven matrix as contracts stabilize | 23-27, 19, 13 |
| [ ] | 32 | `05_code_plans/05_versioning_license_release.md` | v1.0 release gate *(checkpoint: full release)* | 29, 30, 31 |

### Tier 3 — Reach extensions (Phase 9; each spike-gated, never blocks v1.0)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [ ] | 33 | `04_code_plans/01_encoder_model_support.md` | Cheapest reach; arch-registry entry + codec/manifest reuse | 13, 7, 9 |
| [ ] | 34 | `04_code_plans/02_training_scheduler_workmanager.md` | Charging-cycle training; closes `ORTScheduler` state-persistence gap *(checkpoint: scheduled train→resume)* | 18, 17 |
| [ ] | 35 | `04_code_plans/03_federated_codec_and_python_simulation.md` | Python Flower sim (Option A first) *(checkpoint: N-client sim)* | 8, 9, 13 |
| [ ] | 36 | `04_code_plans/04_federated_android_gateway.md` | Android client + gateway (Option B) | 35, 34, 18 |
| [ ] | 37 | `04_code_plans/05_functiongemma_architecture_gate_and_intents.md` | Gemma-3 inference-graph as arch-registry entry + intents *(checkpoint: train→tool-call→intent)* | 11, 7, 9 |

## Per-plan completion self-checks

Reflective "is it really done?" questions, complementary to each plan's `## Tests & acceptance` Definition of done. Answer **all yes** before flipping the `Done` box.

### #1 — Python package & uv scaffolding (`00_code_plans/01`)
- [ ] Does `uv sync` succeed and expose the `mobiletransformers` console script?
- [ ] Does `python -c "import mobiletransformers"` import the whole package tree cleanly?
- [ ] Are generated artifacts (`build/`, `models/`, caches) gitignored and outside the package?

### #2 — Dependency profiles & ORT-training wheel (`00_code_plans/03`)
- [ ] Do the four `onnxruntime`-bearing profiles resolve in mutually-exclusive uv groups that never co-install?
- [ ] Is the torch ABI for the source-built ORT-training wheel pinned and recorded?
- [ ] Does `third_party/onnxruntime/manifest.json` capture ORT SHA + build flags + wheel checksum?

### #3 — Source-built ORT-training pipeline (`01_code_plans/06`)
- [ ] Can `generate_artifacts` run from the source-built wheel on a tiny fixture (toolchain proven alive)?
- [ ] Is the rebuild reproducible from `BUILD.md` with recorded checksums?
- [ ] Does the wheel CI smoke pass without ever pulling a public `onnxruntime-training`?

### #4 — Config layering (`00_code_plans/02`)
- [ ] Does precedence resolve CLI > env > YAML > package default, with secrets only in `Settings`?
- [ ] Do the deprecation shims for `config.py` / `tools/parser_config.py` still import and warn?
- [ ] Are non-secret constants in `constants.py` and user knobs in `config.yml`, with **no** secrets in YAML?

### #5 — Python code quality & module health (`00_code_plans/10`)
- [ ] Are `ruff` + `mypy` green in CI at the agreed ratchet, wired into `make lint` / `make typecheck`?
- [ ] Does new code use `get_logger()` + the `exceptions.py` hierarchy (no `print`, no bare `raise Exception`)?
- [ ] Is `mobiletransformers.__all__` declared, and does the enums/schema parity test pass?
- [ ] Does each remaining monolith carry a decomposition note tying its split to the registry/merger work?

### #6 — Typed models, enums & registries (`00_code_plans/09`)
- [ ] Can a new PEFT / architecture / merger be added with **only** a registry entry + enum member (zero new `if/elif`; grep for survivors)?
- [ ] Is every closed string set an enum mirrored Python↔Kotlin, proven by the CI parity test?
- [ ] Does `model_dump(by_alias=True)` round-trip through the generated schema that Kotlin/C++ validate?
- [ ] Did `build_merger_model(MergerSpec)` replace the `create_*_merger_model{,_2}` duplication?

### #7 — Optimum ONNX export & TasksManager (`01_code_plans/05`)
- [ ] Does TasksManager-driven discovery + task auto-select work with no per-architecture `if/elif`?
- [ ] Is the Optimum-v2 / `optimum-onnx` migration risk covered by the symbol-check spike with a documented fallback?
- [ ] Is the export front door behind `EXPORT_FRONTEND_REGISTRY` (F3)?

### #8 — Weight handoff map & tensor codec (`00_code_plans/07`)
- [ ] Is `weight_handoff_map.json` the sole source of tensor identity (no hard-coded `replace_prefix` on the load side)?
- [ ] Does the codec resolve names/dtype/shape/order deterministically from the map + registries?
- [ ] Are `schemaVersion` + `minReaderVersion` present and is an unsupported `handoffMode` fail-closed (F1/F7)?
- [ ] Is the quantized triple (weight/scale/zero_point) naming inconsistency resolved and regression-tested?

### #9 — Unified merger & external-data export (`01_code_plans/01`)
- [ ] Do offline and on-device merge emit **identical** external-initializer filenames keyed by the map?
- [ ] Is the frozen base one immutable blob + per-tensor trainables (no graph rewrite)?
- [ ] Are atomic rename + checksum enforced on both the Python and C++ sides?

### #10 — GenAI external-data swap spike (`01_code_plans/02`)
- [ ] Does the spike show a per-tensor `.bin` swap is observable in GenAI output (or a clear FAIL)?
- [ ] Is RSS measured and the `OgaCreateModelWithInitializers` symbol presence verified on the Android lib?
- [ ] Is the Gate 0.1 PASS/FAIL decision recorded with evidence?

### #11 — Inference engine abstraction (`01_code_plans/03`)
- [ ] Is there **one** `ModelRuntime` with Native (guaranteed) + GenAI (opt-in) over the **same** package?
- [ ] Do both engines emit an identical callback sequence (parity lock)?
- [ ] Does engine selection fall back to Native transparently when GenAI is unavailable?

### #12 — Memory-mapping experiments (`01_code_plans/04`)
- [ ] Do the experiments produce RSS numbers feeding the Gate 0.2 decision?
- [ ] Is mmap kept non-blocking (an optimization, not a v1 requirement)?

### #13 — Manifest-first package & cache bridge (`00_code_plans/06`)
- [ ] Does the installer materialize the Hub layout into the `LLMRepository` cache shape **atomically**?
- [ ] Does variant selection (ABI / memory / feature) pick deterministically and tie-break stably?
- [ ] Are `schemaVersion` + `minReaderVersion` honored with unknown-field tolerance (F1)?

### #14 — Hub model package format (`02_code_plans/03`)
- [ ] Is it **one** shared package (variant declares engines), not separate per-engine packages?
- [ ] Do `sanitize_repo_id` + the feature-group download plan match the cache-bridge contract?
- [ ] Are checksums + per-file sizes present and schema-versioned?

### #15 — One-command export CLI (`02_code_plans/05`) · checkpoint
- [ ] Does one command go HF model → validated device-ready package (inference + train + merger + manifest)?
- [ ] Does it delegate to existing modules (no reimplementation), with CLI > env > YAML > default overlay?
- [ ] Does the **export E2E workflow** test pass on a tiny model end to end?

### #16 — Android Gradle rename (`00_code_plans/04`)
- [ ] Do `:MobileTransformers` + `:MobileTransformersApp` assemble after the rename?
- [ ] Is ObjectBox generated code regenerated under the new namespace without breakage?
- [ ] Are the JNI symbol decisions (rename vs alias) documented and does the app still link?

### #17 — Android facade foundation (`00_code_plans/05`) · checkpoint
- [ ] Does `MobileTransformers.fromPretrained(...)` return a working `MobileTransformerModel` wrapping the existing repositories (no JNI rewrite)?
- [ ] Are the engine selector + exception hierarchy in place?
- [ ] Does the facade workflow (load → generate one token, device-manual) pass?

### #18 — Training lifecycle & checkpoint contracts (`00_code_plans/08`)
- [ ] Does `TrainingJob` expose status/events/checkpoint without hiding the native lifecycle?
- [ ] Is the callback→event adapter complete and the checkpoint file format preserved?
- [ ] Are the session lock + cooperative cancellation defined for reuse by the scheduler (#34)?

### #19 — HF-style Kotlin facade (`02_code_plans/01`) · checkpoint
- [ ] Do `applyPeft`/`train`/`merge`/`generate`/`retrieve` map cleanly onto existing repositories?
- [ ] Are public config names HF-aligned and mapped to internal config (mapping table present)?
- [ ] Does the **train→merge→generate** workflow (device-manual) pass end to end?

### #20 — Optimum support matrix (`02_code_plans/02`)
- [ ] Is `model_support_matrix.json` generated truth with inherited statuses + earliest-blocker attribution?
- [ ] Does the matrix **feed** (not duplicate) the compatibility doc (F6)?

### #21 — Hub pull & cache flow (`02_code_plans/04`) · checkpoint
- [ ] Does Python `pull_package` / `install_package` produce the exact cache layout the SDK loads?
- [ ] Is variant selection identical Python↔Kotlin (deterministic)?
- [ ] Does the **pull → materialize → load** workflow pass?

### #22 — Adapter push-back (`02_code_plans/06`)
- [ ] Does export produce a PEFT-compatible layout when clean, else a documented native fallback?
- [ ] Is Android upload gated / disabled by default with a privacy warning?

### #23 — Inference handoff alignment & native hardening (`03_code_plans/01`)
- [ ] Does Native implement `ModelRuntime` with map-driven, **fail-closed** external-initializer load?
- [ ] Are there zero `inference/merged/` references left, and the dead GenAI stubs deleted?
- [ ] Is the conversation-state prepend bug fixed with a reset test?

### #24 — Sampling & streaming public config (`03_code_plans/02`)
- [ ] Are public sampling names HF-aligned (`SamplingMethod` enum) and mapped to internal with exact defaults?
- [ ] Is the callback sequence identical across engines (parity)?

### #25 — Vector store boundary & in-memory test (`03_code_plans/03`)
- [ ] Does `InMemoryVectorStore` let RAG logic be unit-tested on the JVM with no ObjectBox/device?
- [ ] Is the `1 - score` distance→similarity conversion covered by an explicit test?
- [ ] Are unsupported embedding dimensions rejected fail-closed, and backends pluggable via the registry (F4)?

### #26 — RAG ingestion & chunking (`03_code_plans/04`)
- [ ] Does `ingestData()` chunk + embed + store `.txt`/`.md`/`.jsonl` with progress, replacing the TODO?
- [ ] Are loaders behind `DOCUMENT_LOADER_REGISTRY` so PDF/HTML slot in later (F3)?
- [ ] Is PDF/Word explicitly out of v1 scope and documented?

### #27 — RAG config & grounded generation (`03_code_plans/05`) · checkpoint
- [ ] Is the grounded flow inspectable (retrieve → assemble → generate, prompt visible)?
- [ ] Is `RagConfig` public and the default template overridable?
- [ ] Does the **ingest → retrieve → grounded-generate** workflow pass?

### #28 — Makefile & CLI entrypoints (`05_code_plans/01`)
- [ ] Are all targets thin wrappers over the CLI/Gradle (no logic), respecting profile isolation?
- [ ] Does `clean-generated` never touch `cache_dir/`?
- [ ] Are `lint` / `typecheck` targets present (for #5)?

### #29 — Staged CI pipeline (`05_code_plans/02`)
- [ ] Is CI staged cheapest-first (fast → export-smoke → android-assemble), with zoo/device nightly only?
- [ ] Does the Android assemble job work **before and after** the rename?
- [ ] Are the lint / typecheck / parity gates wired (F2)?

### #30 — AAR & local-Maven publication (`05_code_plans/03`) · checkpoint
- [ ] Is the missing `aarLibs/` / `libs` native input resolved first?
- [ ] Does the AAR publish to mavenLocal and an **external consumer app build** against it (workflow)?
- [ ] Are third-party AARs (`onnxruntime-genai`) handled (vendored `.so` or explicit consumer dep)?

### #31 — Docs set & compatibility matrix (`05_code_plans/04`)
- [ ] Is each doc written only when its contract locks (no drift), sourced from the owning plan?
- [ ] Is the compatibility matrix **rendered from** `model_support_matrix.json` (F6)?
- [ ] Is the Python public API (F5) documented in `PUBLIC_API.md` alongside the Kotlin facade + CLI?

### #32 — Versioning, license & v1.0 release (`05_code_plans/05`) · checkpoint
- [ ] Is the code relicensed to Apache-2.0 with SPDX headers and all rights-holders' agreement?
- [ ] Are `CITATION.cff` version/date == the tag, and are CHANGELOG non-goals listed?
- [ ] Does the **full release gate** (CI green + AAR + consumer smoke + docs + tag) pass?

### #33 — Encoder-model support (`04_code_plans/01`)
- [ ] Did the spike prove export + a train step + Android smoke + a metric for one small encoder?
- [ ] Is encoder support a `TASK_REGISTRY`/architecture-registry entry (no new `if/elif`, no KV-cache) (F3)?
- [ ] Was MARS-transfer-to-encoder-linear-layers **verified**, not assumed?

### #34 — Charging-cycle training scheduler (`04_code_plans/02`) · checkpoint
- [ ] Is `ORTScheduler.stateDict()`/`loadFromState()` implemented so the LR schedule survives chunk boundaries?
- [ ] Does a charging-constrained foreground `CoroutineWorker` checkpoint + resume cleanly across Doze?
- [ ] Does the **multi-chunk scheduled-train → resume** workflow pass (thermal/energy logged)?

### #35 — Federated codec & Python Flower simulation (`04_code_plans/03`) · checkpoint
- [ ] Does `FederatedAdapterRecord` derive ordering from `TrainableTensorCodec` (no new ordering) (F8)?
- [ ] Does an N-client Flower simulation aggregate adapter tensors and improve the metric?
- [ ] Is `adapterFormatVersion` linked to `weight_handoff_map.schemaVersion` (F1/F8)?

### #36 — Federated Android client & gateway (`04_code_plans/04`)
- [ ] Are `exportTrainableTensors`/`importTrainableTensors` JNI added, and is the codec byte-identical to Python (golden test)?
- [ ] Are the privacy/security gates (consent, TLS, auth, clipping/DP) addressed before any real-user run?
- [ ] Is it hard-gated on #35 passing first?

### #37 — FunctionGemma architecture gate & intents (`04_code_plans/05`) · checkpoint
- [ ] Did the architecture gate pass — Gemma-3 **inference**-graph export added as a registry entry?
- [ ] Does it **never** execute raw model output (allowlist + dry-run + validated tool calls)?
- [ ] Does the train → validated-tool-call → dry-run-intent demo show ≥2 differentiators?

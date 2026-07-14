# Implementation Order — Code Plans Index

This index orders every feature code-plan under `agent_docs/00_code_plans/` … `agent_docs/05_code_plans/` from **most foundational first**. Implement in this order; each plan file repeats its own Prerequisites / Blocks header.

The high-level "what & why" lives in the six tier docs (`00_repository_restructure_plan.md`, `01_tier0_foundation_decisions.md`, `02_tier1_hf_integrated_core.md`, `03_tier2_inference_and_rag.md`, `04_tier3_reach_extensions.md`, `05_cross_cutting_release_modernization.md`). These code plans are the "how" — concrete enough for another agent to implement.

**Tracking:** the `Done` column is the live checklist — flip `[ ]` → `[x]` only when the plan's **Definition of done** (in its `## Tests & acceptance` section) holds *and* the matching block in [Per-plan completion self-checks](#per-plan-completion-self-checks) below all answer "yes".

## Canonical decisions every plan inherits

- **Dual inference engine over one package.** The same exported package + same Android-cache folder is consumable by **both** the native ORT engine and the ONNX Runtime GenAI engine. GenAI is a *selectable* engine (Native is the default/guaranteed path), not a separate package. See `01_code_plans/03_inference_engine_abstraction_native_and_genai.md`.
- **Unified weight handoff via external initializers.** Trainable/merged weights are ONNX **external initializers, one-file-per-tensor**, in `<cacheDir>/<model>/inference/`. The frozen quantized base is a separate immutable external blob. On-device merge overwrites the per-tensor files (atomic rename + checksum). No graph rewrite, no weights-as-inputs, no genai fork. See `01_code_plans/01_unified_merger_and_external_data_export.md`.
- **Canonical `inference/` layout is FLAT.** `inference/model.onnx` + `inference/frozen_base.onnx.data` (the immutable base blob — this exact filename, matching the handoff map's `frozenBaseBlob`) + per-tensor `<name>.bin` (+ sibling `.sha256`) directly in `inference/`, beside `generation_config.json` / `genai_config.json` / `session_options.json` / `weight_handoff_map.json`. **No `inference/base/` and no `inference/merged/` subdirectories** — any plan text mentioning them describes the legacy behavior being retired.
- **`weight_handoff_map.json` is the single source of truth** that replaces hard-coded name rewrites (`weight_merger.cpp:904`). Schema lives in `00_code_plans/07_weight_handoff_map_and_tensor_codec.md`.
- **Dependency profile isolation** (uv groups/extras): `onnxruntime-training` (source-built, provides `onnxruntime`), public `onnxruntime`, `onnxruntime-genai`, and `optimum-onnx[onnxruntime]` collide on the `onnxruntime` import and must never share one env. See `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`.
- **Apache-2.0** is the target framework license (model weights keep upstream licenses).
- **External-initializer handoff is the only merge path.** The legacy `inference/merged/` subdirectory is retired; merged weights are flat per-tensor external initializers in `inference/`. `HandoffMode` enumerates `external_initializer` / `model_input` / `adapter`, but **only `external_initializer` is supported in v1** — the others are registry stubs that fail closed with "not supported in this version". Tier 2 consumes the existing `ModelRuntime` engine boundary and must not define a competing engine *interface* (`InferenceEngine` is only the selector *enum* declared by `01_code_plans/03`). See `03_code_plans/01`.
- **`FederatedAdapterRecord` derives from `TrainableTensorCodec`.** Federated exchange (Tier 3) is a thin wrapper over `weight_handoff_map.json` + the codec from `00_code_plans/07`; it never invents its own tensor ordering, and its `adapterFormatVersion` tracks the handoff-map `schemaVersion`. See `04_code_plans/03`.
- **One canonical manifest schema, one owner.** `02_code_plans/03` owns the `mobiletransformers_manifest.json` **field list** (camelCase wire names: `supportedEngines`, `defaultVariant`, `recommendedDeviceMemoryMb`, per-variant `paths`, `downloadPlan` keyed `[variantId][group]`); `00_code_plans/06` owns the **validator / variant-selection / cache-install semantics** and references that schema instead of redefining it. `sanitize_repo_id` maps `/` → `__` (double underscore) and is defined once in `02_code_plans/03`, mirrored byte-identically in Kotlin.
- **Kotlin naming split (no duplicate contracts).** `ModelRuntime` is the **inference-engine boundary** (`01_code_plans/03`: `load/generate/release`, impls `ORTGeneratorNative`/`ORTGeneratorGenAI`, with engine-level `EngineCapabilities`). The facade-level whole-model adapter (`00_code_plans/05`) is **`ModelSession`** (impl `RepositoryBackedModelSession`, model-level `RuntimeCapabilities`). One `InferenceEngine { NATIVE, GENAI }` enum, declared once by `01_code_plans/03` (#11 — it lands first, order 11) and reused verbatim by #17/#19; no plan declares a second engine enum.
- **Python config lives inside the package.** `src/mobiletransformers/config/` holds `constants.py` (enums), `settings.py` (secrets), `models.py` (Pydantic), `registry/` — all shipped in the wheel. Repo-root `config/` holds only user-editable YAML (`config.yml`, `logging.yml`). Root `config.py` / `tools/parser_config.py` become deprecation shims importing from `mobiletransformers.config.*`. Owned by `00_code_plans/02` + `00_code_plans/09`.
- **Cross-boundary readers use typed fail-closed parsing, not on-device JSON-Schema validation.** Kotlin/C++ parse into typed classes (Gson on Android — already a dependency); closed-set values go through enum `fromWire()` (throws on unknown **value**); `check_compat()` gates `schemaVersion`/`minReaderVersion`; unknown **fields** are tolerated (additive minor bumps). Pydantic read models use `extra="ignore"`. `schemas/*.schema.json` + golden `enums.json` are **CI parity artifacts only**, never a device dependency.
- **Public generation config uses `maxNewTokens` from the first facade pass** (`00_code_plans/05`); `03_code_plans/02` locks parity/mapping, it does not rename.
- **Registries replace hardcoded dispatch.** PEFT methods, model architectures, and merger variants are declared in data-driven registries; no `if x == "lora"/elif "mars"` or `architectures[0] == "..."` chains in business logic. Adding a method/architecture/merger is a registry entry + an enum member, not a new `elif`. The same pattern also owns **task-type, execution-provider, document-loader (RAG), and export-front-door** registries, added as their consumers land. Owned by `00_code_plans/09_typed_models_enums_and_registries.md`.
- **Pydantic v2 is the typed config contract.** Every tunable config object is a Pydantic model with camelCase aliases; cross-boundary JSON is `model_dump(by_alias=True)` + a generated `schemas/*.schema.json` kept as a **CI parity artifact** (the Kotlin parser and C++ loader enforce the contract via typed fail-closed parsing, not runtime schema validation — see the typed-parsing decision above). Secrets stay in `Settings` (`00_code_plans/02`). `pydantic>=2` is a core dependency (`00_code_plans/03`). Owned by `00_code_plans/09`.
- **Enums own every closed string set**, mirrored Python (`mobiletransformers/config/constants.py`) ↔ Kotlin (`.../constants/*.kt`): sampling method, scheduler type, execution provider, core/memory config id, search type, quantization type, PEFT method, task type, handoff mode, merger variant. Owned by `00_code_plans/09`.
- **Full merger unification.** A single parameterized `build_merger_model(MergerSpec)` replaces `create_lora_merger_model{,_2}` / `create_mars_merger_model{,_2}` (the "merger 1/2" duplication); C++ `get_merger_type`/`run_merger_model` resolve the spec from the handoff map + merger registry instead of branching on `"lora_q"/"mars_q"`. Owned by `00_code_plans/09`, wired by `01_code_plans/01`.
- **Schema versioning is uniform and forward-compatible.** Every JSON contract (`mobiletransformers_manifest.json`, `weight_handoff_map.json`, `model_support_matrix.json`, `FederatedAdapterRecord`, generated `schemas/*.schema.json`) carries `schemaVersion` (`MAJOR.MINOR`) + `minReaderVersion`. Readers **preserve unknown fields** (additive minor bumps are non-breaking) and **fail closed with a "needs newer SDK" message** only when `major` exceeds support or `minReaderVersion` is unmet. One `check_compat()` helper mirrored Python↔Kotlin. Owned by `00_code_plans/07`; consumed by `00/06`, `02/02`, `02/03`, `04_code_plans/03`.
- **Cross-language parity is CI-enforced, not manual.** Pydantic models (`config/models.py`) + enums (`config/constants.py`) are the single source of truth; `schemas/*.schema.json` and a golden `enums.json` are generated from them, and a CI `parity` job fails if the checked-in Kotlin/C++ mirrors drift. Owned by `00_code_plans/09`, gated by `05_code_plans/02`.
- **The Python library API is a declared, SemVer-governed surface** (`mobiletransformers.__all__`), peer to the Kotlin facade and the CLI — not just the CLI. Owned by `00_code_plans/10`, documented by `05_code_plans/04`, versioned by `05_code_plans/05`.
- **Tests follow a fixed taxonomy in every plan**: **unit** (automated; `pytest` / Android JVM, incl. a compile/assemble check) · **integration** (automated; run → expected output) · **manual** (user-run; long/intensive or device-specific) · a **Definition of done**. End-to-end **workflow** tests live only on feature-complete checkpoint plans (#15, #17, #19, #21, #27, #30, #32, #34, #35, #37). Android device/emulator behaviour is always manual (user-run); CI runs only compile + `assembleDebug`.

## How to execute a plan (implementer protocol — read this first)

Every code plan is written to be executed cold by an agent. Follow this protocol:

1. **Read in this order:** the [Canonical decisions](#canonical-decisions-every-plan-inherits) above → the plan file itself → every plan named in its `Interactions` section (skim their Data-contracts sections). A plan's `Prerequisites` header tells you what is already merged when you start — you may rely on those contracts existing.
2. **Plan anatomy is uniform:** `Purpose` (why) → `Touched / new files` (the complete file inventory) → `Data contracts / interfaces` (the shapes you must produce — these are normative, not illustrative, unless marked "sketch"/"illustrative") → `Implementation steps` (ordered; follow the order) → `Interactions` (who owns what) → `Tests & acceptance` (unit/integration/manual taxonomy + Definition of done).
3. **Ownership wins on conflict.** If two plans describe the same contract differently, the plan marked as OWNER (or named in the canonical-decisions block) is right and the other is stale — fix the stale reference, do not implement both. Never re-declare a type/schema another plan owns; import or reference it.
4. **Line numbers are hints, not addresses.** Cited `file.py:123` locations were verified against the tree at planning time and **will have drifted** by execution time. Re-locate by symbol/function name and surrounding code, not by line. If a cited symbol no longer exists, stop and check whether an earlier-ordered plan renamed/removed it (its Interactions section will say so).
5. **File inventories are snapshots.** "Touched / new files" lists reflect the tree at planning time; earlier-ordered plans add/delete files. Re-derive the real set (`git ls-files`, grep) before bulk operations like the #16 rename.
6. **Fail closed is the default.** Any parse/validate/lookup that cannot satisfy a contract raises a typed error naming the offending entity *before* side effects (session creation, file writes, downloads). Never silently fall back to legacy behavior unless the plan explicitly defines the fallback.
7. **Wire names are camelCase** in every cross-boundary JSON; closed-set values go through the shared enums; every new cross-boundary JSON gets the `schemaVersion`/`minReaderVersion` block and a `check_compat()` gate (algorithm pinned in `00_code_plans/07`).
8. **Do not widen scope.** If you discover an adjacent problem the plan does not cover, note it for the plan that owns that area (check IMPLEMENTATION_ORDER) instead of fixing it inline.
9. **Done = Definition of done + self-check.** Flip the `Done` box only per the Tracking rule at the top of this file; run the plan's tests in the taxonomy order (unit → integration → manual/user-run).

## Global order

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [x] | 1 | `00_code_plans/01_python_package_and_uv_scaffolding.md` | Unblocks all Python work | — |
| [x] | 2 | `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md` | Lock the toolchain before building on it (incl. `pydantic>=2` core) | 1 |
| [x] | 3 | `01_code_plans/06_source_built_ort_training_pipeline.md` | Prove the train toolchain is alive (`generate_artifacts`) | 2 |
| [x] | 4 | `00_code_plans/02_config_layering_settings_constants.md` | Three config layers + secrets; needed by export/CLI | 1 |
| [x] | 5 | `00_code_plans/10_python_code_quality_and_module_health.md` | **Conventions, typing, lint, shared logging/exceptions + the monolith-decomposition strategy** — established before registries/merger/builders so they inherit it | 1, 2, 4 |
| [x] | 6 | `00_code_plans/09_typed_models_enums_and_registries.md` | **Typed config + enums + PEFT/arch/merger registries** — kills hardcoded dispatch before builders are written | 2, 4, 5 | *(owned contract layer done; legacy-dispatch consumption + merger graph-collapse deferred to #7/#9 — see notes)* |
| [x] | 7 | `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md` | Inference-export front door (arch via registry) | 2, 4, 6 |
| [x] | 8 | `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` | Data contract every later piece reads (codec consumes registries) | 4, 6 | *(Python owner layer done: schema + codec + `check_compat`; C++/Kotlin consumers ride with #9/#23)* |
| [ ] | 9 | `01_code_plans/01_unified_merger_and_external_data_export.md` | **Dual-engine core** + full merger unification | 6, 7, 8 | *(code-complete 2026-07-14: Python A/B/C tested, C++ D/E compile+link-verified on arm64-v8a; box stays open pending the manual on-device parity/atomic/load-smoke tests — see #9 self-check)* |
| [ ] | 10 | `01_code_plans/02_genai_external_data_swap_spike.md` | Feeds Gate 0.1 | 9 |
| [ ] | 11 | `01_code_plans/03_inference_engine_abstraction_native_and_genai.md` | Engine selection | 10 |
| [ ] | 12 | `01_code_plans/04_memory_mapping_experiments.md` | Optimizes 9–11 (non-blocking) | 9 |
| [x] | 13 | `00_code_plans/06_manifest_first_package_and_cache_bridge.md` | Package contract | 8, 9 | *(done 2026-07-14: Python `artifacts/manifest.py` + Kotlin `packages/` cache-bridge (6 classes), JVM-tested + `compileDebugKotlin`; only the on-device generate smoke deferred)* |
| [x] | 14 | `02_code_plans/03_hub_model_package_format.md` | Hub repo shape | 13 | *(done 2026-07-14: `hub/package_format.py` — `sanitize_repo_id`, `build_manifest`, tiny_package fixture; no device)* |
| [x] | 15 | `02_code_plans/05_one_command_export_cli.md` | Wraps 7–9 + 13 *(checkpoint: export E2E)* | 7, 9, 13 | *(done 2026-07-14: `export/pipeline.py` + `cli export/push`; dry-run + assemble→validate checkpoint automated; real full-model export env-gated, not device)* |
| [x] | 16 | `00_code_plans/04_android_gradle_rename_migration.md` | Isolated, verified rename | — | *(done 2026-07-14: **full removal / option B** — Kotlin+native+JNI all renamed off `ortmobile`; supersedes the doc's option-A. Compile+link-verified arm64-v8a, `compileDebugKotlin` + `make parity` green.)* |
| [ ] | 17 | `00_code_plans/05_android_facade_foundation.md` | Public SDK facade *(checkpoint: load→generate)* | 13, 16 |
| [ ] | 18 | `00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md` | Job/progress/checkpoint API | 17 |
| [ ] | 19 | `02_code_plans/01_hf_style_kotlin_facade.md` | `fromPretrained`/train/merge/generate *(checkpoint: train→merge→generate)* | 11, 17 |
| [x] | 20 | `02_code_plans/02_optimum_support_matrix.md` | Reporting layer | 7 | *(done 2026-07-14: `support/` package (statuses/models/matrix) + `support-matrix` CLI; inherited statuses, probe ingestion, filtered docs; detection injectable/mocked in CI, ready-statuses read device probes when present)* |
| [x] | 21 | `02_code_plans/04_hub_pull_and_cache_flow.md` | Python pull first, Android downloader next *(checkpoint: pull→load)* | 13, 14 | *(done 2026-07-14: `hub/{variant_select,pull}.py` + `cli pull`/`install-package`; pull→install→sha256 automated over the fixture; Android downloader + device load deferred)* |
| [x] | 22 | `02_code_plans/06_adapter_pushback.md` | Last Tier-1 piece | 9, 14 | *(done 2026-07-14: `adapter/{export,convert,model_card}.py` + `cli push-adapter`; PEFT/native gate + card + dry-run automated; safetensors materialization + on-device upload env/device-gated)* |

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
| [ ] | 34 | `04_code_plans/02_training_scheduler_workmanager.md` | Charging-cycle training; closes the **`LinearLRScheduler`** state-persistence gap (`ORTScheduler.kt:156-161` `TODO` — Cosine already implements + is wired via `training_state.json`) *(checkpoint: scheduled train→resume)* | 18, 17 |
| [ ] | 35 | `04_code_plans/03_federated_codec_and_python_simulation.md` | Python Flower sim (Option A first) *(checkpoint: N-client sim)* | 8, 9, 13 |
| [ ] | 36 | `04_code_plans/04_federated_android_gateway.md` | Android client + gateway (Option B) | 35, 34, 18 |
| [ ] | 37 | `04_code_plans/05_functiongemma_architecture_gate_and_intents.md` | Gemma-3 inference-graph as arch-registry entry + intents *(checkpoint: train→tool-call→intent)* | 11, 7, 9 |

## Per-plan completion self-checks

Reflective "is it really done?" questions, complementary to each plan's `## Tests & acceptance` Definition of done. Answer **all yes** before flipping the `Done` box.

### #1 — Python package & uv scaffolding (`00_code_plans/01`)
- [x] Does `uv sync` succeed and expose the `mobiletransformers` console script?
- [x] Does `python -c "import mobiletransformers"` import the whole package tree cleanly?
- [x] Are generated artifacts (`build/`, `models/`, caches) gitignored and outside the package?

> Done 2026-07-13. `uv sync --group dev` resolves on Python 3.10.12 (core is platform-neutral: no torch/onnxruntime); `mobiletransformers`/`python -m mobiletransformers.cli.main` both `--help` exit 0; `uv build` emits a wheel containing only `src/mobiletransformers`; full `pkgutil.walk_packages` import clean; `test_import_compat.py` green.

### #2 — Dependency profiles & ORT-training wheel (`00_code_plans/03`)
- [x] Do the four `onnxruntime`-bearing profiles resolve in mutually-exclusive uv groups that never co-install?
- [x] Is the torch ABI for the source-built ORT-training wheel pinned and recorded?
- [x] Does `third_party/onnxruntime/manifest.json` capture ORT SHA + build flags + wheel checksum?

> Done 2026-07-13. `[tool.uv] conflicts` isolates `ort-training-local`✕`export`✕`genai-smoke`; `uv sync --extra export --group ort-training-local` errors as designed. The **source-built wheel is proven alive**: `uv sync --python 3.12 --group ort-training-local` then `import onnxruntime, torch; from onnxruntime.training import artifacts; from onnxruntime.training.api import CheckpointState, Module, Optimizer` → ORT 1.23.0+cpu · torch 2.7.1 · numpy 1.26.4 · `generate_artifacts` callable. Manifest records ORT SHA `9b25b6a838…`, build flags, `torch_version=2.7.1`, wheel sha256 `87e6f3c6…`. Lockfiles + SBOM generated deterministically.
> **Deviations from the plan doc** (all validated): (a) the wheel is **cp312-only** → `ort-training-local` syncs under Python 3.12, marked `python_version=='3.12'` in pyproject; (b) real torch ABI is **2.7.1** (doc guessed 2.5.1) — the "one unknown" was already resolved by the existing build; (c) the wheel's C-extension needs **numpy<2** (built against 1.26.4), pinned for the 3.12 fork; (d) the `export` profile transitively needs **Python ≥3.11** (onnxruntime≥1.24 dropped cp310 wheels) — core/dev stay 3.10-clean; (e) **`export-rocm` is deferred/empty** (ROCm wheels need a dedicated AMD index, out of this foundation pass) — so the fourth onnxruntime provider and the `export`✕`export-rocm` conflict are not yet wired.

### #3 — Source-built ORT-training pipeline (`01_code_plans/06`)
- [x] Can `generate_artifacts` run from the source-built wheel on a tiny fixture (toolchain proven alive)?
- [x] Is the rebuild reproducible from `BUILD.md` with recorded checksums?
- [x] Does the wheel CI smoke pass without ever pulling a public `onnxruntime-training`?

> Done 2026-07-13. `tests/fixtures/tiny_trainable.onnx` (252 B, generated by `make_tiny_trainable.py`) + `training_config.json` carry the exact fields `gen_artifacts` reads. `tests/integration/test_ort_training_smoke.py` (skips unless `onnxruntime.training` imports) **passes under the 3.12 training profile**: `generate_artifacts(..., optimizer=AdamW)` produces `training_model.onnx`/`eval_model.onnx`/`optimizer_model.onnx`/`checkpoint`, and a one-step `Module`/`Optimizer` train yields a finite loss. Fixture well-formedness (`tests/fixtures/test_tiny_trainable.py`) runs onnx-only in the core env. Build scripts + `manifest.json` (SHA256 `87e6f3c6…`) + `BUILD.md` were authored under #2.
> **Runtime ABI constraints discovered & pinned** (into the `ort-training-local` group, recorded in the manifest): `numpy<2` (wheel built against the numpy 1.26 ABI) and `onnx<1.19` (ORT 1.23 runtime caps ONNX IR at 11; onnx≥1.19 emits IR 13 and the generated optimizer graph fails to load). Both forked to the 3.12 split so the export profile keeps numpy/onnx 2.x.
> **Deferred to #29 (CI staged pipeline):** `.github/workflows/ort-training-smoke.yml` exists but is `workflow_dispatch`-only and self-skips when the wheel is absent — the source-built wheel is git-ignored and not on PyPI, so provisioning it to a hosted runner (rebuild-in-CI vs. private index/artifact) is a #29 decision. The **local** smoke is the Gate 0.3 evidence and passes. The Android AAR (`build_ort_training_android.sh`) remains a #16/#04 concern (manifest Android fields still null).

### #4 — Config layering (`00_code_plans/02`)
- [x] Does precedence resolve CLI > env > YAML > package default, with secrets only in `Settings`?
- [x] Do the deprecation shims for `config.py` / `tools/parser_config.py` still import and warn?
- [x] Are non-secret constants in `constants.py` and user knobs in `config.yml`, with **no** secrets in YAML?

> Done 2026-07-13. `mobiletransformers.config.resolve()` implements CLI>env>YAML>default (unit-tested); secrets live only in `Settings`/`get_settings()` and the CI grep guard (`os.environ[...HF_TOKEN|HF_CACHE|GEMINI_API_KEY|AZURE_...]` over `src/`) is empty. Root `config.py` + `tools/parser_config.py` are deprecation shims (import + `DeprecationWarning` tested); `config/config.yml` copied verbatim (4 sections, YAML-only dir); `utils/yaml.load_config_from_file` added. `test_settings_precedence.py` green.
> **In-scope deferral:** the six in-module `load_config_from_file` copies are intentionally NOT ripped out this pass (business-module migration is a later plan). `trainer/builder.py`'s variant pre-indexes `config[TRAIN_CONFIG]`, so its call sites must migrate together with that swap — the shared util is provided for new code only.

### #5 — Python code quality & module health (`00_code_plans/10`)
- [x] Are `ruff` + `mypy` green in CI at the agreed ratchet, wired into `make lint` / `make typecheck`?
- [x] Does new code use `get_logger()` + the `exceptions.py` hierarchy (no `print`, no bare `raise Exception`)?
- [x] Is `mobiletransformers.__all__` declared, and does the enums/schema parity test pass?
- [x] Does each remaining monolith carry a decomposition note tying its split to the registry/merger work?

> Done 2026-07-13. `[tool.ruff]` (E/F/I/W/UP/B, format) + `[tool.mypy]` (lenient global, `disallow_untyped_defs` for `mobiletransformers.*`) gate `src/`+`tests/` only (legacy root + `research/` excluded until they migrate); `ruff check`/`ruff format --check`/`mypy` all clean (23 files). `exceptions.py` (`MobileTransformersError` → `ConfigValidation/Export/Manifest/Handoff/Merge/UnsupportedModel/Hub`, mirroring the Kotlin names), `utils/logging.py` (`get_logger`/`configure_logging`, NullHandler, no `print` in library code), `_typing.py` + `py.typed`. `mobiletransformers.__all__` declared with a `public_api.txt` golden (guarded by `test_public_api.py`). All 7 monoliths carry `# DECOMPOSE(#N):` notes. Minimal `Makefile` (`lint`/`typecheck`/`test`/`test-train`/`check`) — full set owned by #28. `research/` + legacy root are already out of the wheel (`[tool.hatch.build] packages=["src/mobiletransformers"]`). `make check` green (33 tests).
> **Cross-plan:** the **enums/schema parity test** (`tests/parity/`) is delivered by #6 (it owns the enum/Pydantic source of truth) and CI-wired by #29 — it is implemented immediately after this in #6. The per-module **decomposition splits** ride with their owning plans (#6 per-arch inference builders, #9 merger name-resolution) per the plan's scope boundary; #5 owns the strategy/notes, not the splits.

### #6 — Typed models, enums & registries (`00_code_plans/09`)
- [x] Can a new PEFT / architecture / merger be added with **only** a registry entry + enum member (zero new `if/elif`; grep for survivors)? *(true for `src/`; legacy `trainer/`,`inference/`,`artifact/` still branch — their rewrites ride with #7/#9)*
- [x] Is every closed string set an enum mirrored Python↔Kotlin, proven by the CI parity test?
- [x] Does `model_dump(by_alias=True)` round-trip through the generated schema that Kotlin/C++ validate?
- [x] Did `build_merger_model(MergerSpec)` replace the `create_*_merger_model{,_2}` duplication? *(wired by #9 2026-07-14: single builder + golden-equivalence test vs the legacy `*_2` factories; the four factories are deleted)*

> Done 2026-07-13 (owned contract layer). **A2 enums** — 11 `str,Enum` classes in `config/constants.py` (`SamplingMethod`,`SchedulerType`,`ExecutionProvider`,`CoreConfigId`,`MemoryConfigId`,`SearchType`,`QuantizationType`,`PEFTMethod`,`TaskType`,`HandoffMode`,`MergerVariant`) + `ENUM_REGISTRY`; `SUPPORTED_PEFT_METHODS` now derives from `PEFTMethod`. **A1 Pydantic v2** — `config/models.py` (`SamplingConfig`,`DeviceOptions`,`Linear/CosineScheduler` discriminated union,`QuantizationOptions`,`GenerationConfig`,`TrainingConfig`,`RagConfig`), camelCase aliases, `extra="ignore"` (unknown fields tolerated / unknown enum values fail closed), `schemaVersion`+`minReaderVersion` block. **A3/A4/A5 registries** — `config/registry/{peft,architecture,merger}.py` with lazy dotted-path class binding (core-importable, no optimum/torch at load); `resolve_architecture` covers all 8 legacy training arches, `get_peft_spec` all 5 methods, `resolve_merger` all variants. **Parity** — `python -m mobiletransformers.codegen.enums` generates checked-in `schemas/*.schema.json` + golden `schemas/enums.json`; `--check` (the `make parity` gate) diffs them + the 11 hand-mirrored Kotlin `constants/*.kt` `fromWire` enums (under the pre-rename path) and fails on drift. Tests: `test_config_models`/`test_registries`/`test_enum_parity` + an `src/`-scoped dispatch grep guard. `make check` (lint+typecheck+parity+71 tests) green.
> **Deferred to owning plans** (per the plan's own Interactions/ownership): rewiring the legacy `trainer/builder.py` / `inference/builder.py` / `artifact/onnx_builder.py` dispatch to the registries rides with the training-export migration (#7) and is gated for the inference builder by the Optimum/GenAI decision (restructure master plan says don't rewrite it yet); the single `build_merger_model` ONNX-graph collapse of the four `create_*_merger_model{,_2}` factories + the C++ `weight_merger.cpp` rewrite are wired by #9 (`01_code_plans/01`), which owns the on-disk merge contract + golden-equivalence test. Kotlin enum **mirror files** exist and pass parity, but swapping closed-set `String` fields to enums in `ORTGenerationConfig/ORTTrainingConfig/ORTRagConfig.kt` + `FileUtil.kt fromWire` wiring rides with the Android facade plans (#17/#19) and Android rename (#16). Broadening top-level `__all__` to export the config models/enums/registries is deferred until their consumers (#7/#9) exercise them and the SemVer surface is finalized (#32).

### #7 — Optimum ONNX export & TasksManager (`01_code_plans/05`)
- [x] Does TasksManager-driven discovery + task auto-select work with no per-architecture `if/elif`?
- [x] Is the Optimum-v2 / `optimum-onnx` migration risk covered by the symbol-check spike with a documented fallback?
- [x] Is the export front door behind `EXPORT_FRONTEND_REGISTRY` (F3)?

> Done 2026-07-13. **Migration spike (decisive):** under **optimum 2.1.0 / optimum-onnx 0.1.0 / transformers
> 4.46.2** (`spikes/optimum_migration/check_symbols.py`), `main_export`, `TasksManager`, the `*OnnxConfig`
> model_configs, and lower-level `export()` all **survive**; **`OnnxConfigWithLoss` was REMOVED with no
> replacement**. Two discovery gotchas found & handled: TasksManager's ONNX task map is empty until
> `optimum.exporters.onnx.model_configs` is imported (decorator registration), and lookups need
> `library_name="transformers"`.
> **Deviation from the plan's fallback menu (better than Fallback A/B):** because `export()` survives, the
> training-graph path stays on it with a **vendored `OnnxConfigWithLoss`** (`export/onnx_config_with_loss.py`,
> adapted from optimum v1.24.0, deps `OnnxConfig`/`OnnxConfigWithPast`/`DummyLabelsGenerator`/
> `DEFAULT_DUMMY_SHAPES` all present in 2.1) rather than reconstructing the graph via `torch.onnx` (Fallback A)
> or pinning legacy optimum (Fallback B). The `torch.onnx` frontend row remains **declared + fail-closed**
> (`export/torch_frontend.py`), reserved for the day `export()` also disappears.
> **Delivered:** `export/registry.py` (`discover_tasks`/`choose_task`/`is_supported` + `EXPORT_FRONTEND_REGISTRY`
> keyed by the new Python-only `ExportFrontend` enum — deliberately **not** in `ENUM_REGISTRY`, no Kotlin
> mirror), `export/inference_export.py` (`export_inference` orchestration over `main_export`, captures
> optimum/optimum-onnx/transformers versions), `export/normalize.py` (verifies canonical IO/KV names, fails
> closed on missing `logits`/`present.*`, consolidates external data to one `model.onnx_data`),
> `export/support_matrix.py` (idempotent merge; sets `optimum_exportable` + `mobile_package_exportable`,
> seeds the four deferred statuses None and preserves later-plan values). `trainer/builder.py` ladder
> (`architectures[0] ==`) replaced by `resolve_architecture(config).load_onnx_config_class()` + `choose_task`
> (also fixes the old unbound-`ocl` bug on unknown archs). **Proven end-to-end with real exports:** inference
> export of `hf-internal-testing/tiny-random-LlamaForCausalLM` → flat package (`model.onnx` + single
> `model.onnx_data`, canonical `past_key_values.N`/`present.N` KV names matching `inference/builder.py`
> `make_genai_config`, tokenizer + `generation_config.json`); training-graph export via the vendored wrapper
> (`labels` in, `loss` out). Tests: `tests/export/{test_registry,test_support_matrix,test_onnx_config_with_loss}.py`
> — 11 run in core/dev, 6 skip (need the export profile) and pass under `uv run --extra export`. `make check`
> green (82 passed, 6 skipped); `uv lock --check` + `uv build --wheel` clean.
> **Deferred to owning plans:** real full-size model export smoke (SmolLM2-135M) is a user-run manual test;
> `inference/builder.py` dispatch rewrite still gated by the Optimum-vs-GenAI decision (not #7); the
> `torch.onnx` frontend body only if a future optimum removes `export()`.

### #8 — Weight handoff map & tensor codec (`00_code_plans/07`)
- [x] Is `weight_handoff_map.json` the sole source of tensor identity (no hard-coded `replace_prefix` on the load side)? *(true for the Python contract; the C++ load-side rewrite of `session_cache.h`/`weight_merger.cpp:904` rides with #23/#9)*
- [x] Does the codec resolve names/dtype/shape/order deterministically from the map + registries?
- [x] Are `schemaVersion` + `minReaderVersion` present and is an unsupported `handoffMode` fail-closed (F1/F7)?
- [x] Is the quantized triple (weight/scale/zero_point) naming inconsistency resolved and regression-tested?

> Done 2026-07-13 (Python owner layer). `src/mobiletransformers/artifacts/handoff_map.py` owns the
> `weight_handoff_map.json` schema (v1.0): `TensorSpec`, `ObservedInit`, `HandoffEntry`, `HandoffMap`,
> `TrainableTensorCodec`, all camelCase wire names per the canonical `entries[]` shape. `to_json()` is
> **byte-deterministic** (entries sorted by canonical weight name + `sort_keys`), so #13 can checksum it.
> `validate()` fails closed on: `mergedTensorNames != inferenceInitializerNames` (external_initializer
> invariant), the **documented quantized scale-naming bug** (quantization role names must equal the
> observed inference initializers, never `base_layer_name`), duplicate `externalDataLocation` /
> `inferenceInitializerNames`, and `model_input`/`adapter` modes (fail-closed stubs, F7).
> `TrainableTensorCodec.canonical_inference_name` is the **single** Python impl of the
> `weight_merger.cpp:904` rewrite, driven by #6's `ArchitectureSpec.attention_module_name` (data, not
> literals); `from_peft_mapping` joins `peft_mapping` + `requires_grad` + observed inference inits using
> #6's `PEFTMethodSpec.component_schema` role vocabulary, and **raises on train/infer naming drift**
> (build-time, not runtime). The canonical `check_compat` (F1) lives in
> `artifacts/versioning.py` (`SchemaVersionError`) with a shared cross-language fixture
> `tests/fixtures/check_compat_cases.json`. Tests: `tests/unit/test_handoff_map.py` +
> `test_tensor_codec.py` (26 tests). `make check` green (108 passed, 6 skipped).
> **Deferred to owning plans** (per the plan's Interactions): the build-side **emit** wiring —
> accumulating `observed_inference_inits` in the inference-graph builder and feeding `peft_mapping` at
> export time — needs the inference-builder migration (gated by the Optimum/GenAI decision, as in #7);
> the C++ `weight_merger.cpp`/`session_cache.h` map-driven save/load rewrite rides with **#9** (merge/save)
> and **#23** (native load); the Kotlin JNI thread-through rides with **#18/#19**. The
> cross-language golden + C++ smoke integration tests land with those consumers.

### #9 — Unified merger & external-data export (`01_code_plans/01`)
- [x] Do offline and on-device merge emit **identical** external-initializer filenames keyed by the map? *(code-level yes: both sides read `externalDataLocation[role]` from `weight_handoff_map.json`; byte-for-byte device-vs-offline parity is the outstanding **manual** device test)*
- [x] Is the frozen base one immutable blob + per-tensor trainables (no graph rewrite)? *(export splits base→`frozen_base.onnx.data`, trainables→per-tensor `<name>.bin`; `external_initializer` mode only, `model_input`/`adapter` fail closed)*
- [x] Are atomic rename + checksum enforced on both the Python and C++ sides? *(Python `os.replace` + `.sha256`; C++ `write_raw_tensor_atomic` temp→rename→`.sha256`, SHA-256 host-verified against known vectors)*

**Done 2026-07-14 (code-complete + compile-verified; manual device tests outstanding).** Scope this session was full #9 including the C++ half (user-confirmed).
- **Merger graph collapse (A):** `config/registry/merger.py::build_merger_model` now emits the LoRA/MARS graphs (independent `quant_in`/`quant_out`), replacing the four `artifact/merger.py` factories. Golden-equivalence test `tests/unit/test_merger_builder.py` pins it byte-for-byte to committed goldens generated from the legacy `*_2` factories (`tests/fixtures/gen_merger_golden.py` + `merger_golden/`); structural check runs in the core env, numerical (ORT) under the export profile. Closes #6's open box below.
- **Export orchestrator (B):** new `inference/export_inference_package.py` — splits frozen base vs per-tensor trainable externals, emits `weight_handoff_map.json` via #8's `TrainableTensorCodec`/`HandoffMap`, populates `mergerModels`, writes per-tensor `.sha256`, augments `genai_config.json` with the `session.model_external_initializers_file_folder_path` entry. Role classification reconciles the QDQ and GPTQ quantized vocabularies (closes Tier-0 finding #10). `model_input`/`adapter` fail closed. Unit-tested (`tests/unit/test_export_inference_package.py`, onnx-only).
- **Offline emit driver (C):** `artifact/merger.py` reduced to `emit_merger_models` (registry-driven, descriptive filenames); the four factories + the `onnx_builder.py:628-641` `peft_method == "lora"/"mars"` dispatch are gone.
- **Device save (D):** `weight_merger.cpp` rewritten — `load_handoff_map` + a C++ `check_compat` mirror; `load_merger_models` resolves filenames from `mergerModels`; `save_merged_parameters` writes raw tensor bytes to `externalDataLocation[role]` (atomic temp→rename + `.sha256`), deleting the `inference_name` string-rewrite. **Compiles + links on arm64-v8a** (`libortmobile.so` built); x86_64 link is blocked only by an incomplete vendored `jniLibs/x86_64` in the source repo, not by code.
- **Kotlin caller (E):** `ORTTrainerNative.mergeExportSessionWeights` points merge at the unified `inference/` dir (retires `inference/merged`); `compileDebugKotlin` passes.
- **Outstanding (manual/device + #23):** on-device atomic-overwrite-under-kill, offline-vs-device byte-identical `.bin` parity, native load-and-generate smoke. The native **load** side (`ORTGeneratorNative.loadMergedWeights` / `session_cache.h`) still probes `inference/merged` — its migration to the handoff map is **#23**, flagged with `DECOMPOSE(#23)` at both sites.
- **Env note:** the native build needs the untracked vendored deps (`cpp/includes/google` protobuf headers, `jniLibs/`, `aarLibs/`) — all `.gitignore`d (aarLibs added this session); they were provisioned locally from the sibling `../ORTTransformer` checkout for the compile-check.

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
**Done 2026-07-14 (Python + Kotlin, no device).** Python `artifacts/manifest.py` (validator + `select_variant`, reusing `versioning.check_compat` w/ `MANIFEST_READER_VERSION`); Kotlin `packages/` — `MobileTransformersManifest`, `ManifestValidator`, `VariantSelector`, `ChecksumVerifier`, `ModelPackageInstaller` (atomic `renameTo`), `CacheIndex`; `LLMRepository` untouched. 10 JVM tests + `compileDebugKotlin` green.
- [x] Does the installer materialize the Hub layout into the `LLMRepository` cache shape **atomically**? *(stage `.staging/<id>` → `renameTo`; JVM-tested; on-device generate smoke deferred)*
- [x] Does variant selection (ABI / memory / feature) pick deterministically and tie-break stably? *(Python + Kotlin `select_variant`, identical tie-break; both tested)*
- [x] Are `schemaVersion` + `minReaderVersion` honored with unknown-field tolerance (F1)? *(one `check_compat`, mirrored Kotlin, pinned by shared `check_compat_cases.json`; Gson ignores unknown fields)*

### #14 — Hub model package format (`02_code_plans/03`)
**Done 2026-07-14 (Python, no device).** `hub/package_format.py` + committed `tests/fixtures/tiny_package` fixture + `sanitize_repo_id_cases.json`.
- [x] Is it **one** shared package (variant declares engines), not separate per-engine packages? *(one tree; `variants[].supportedEngines`; dual-engine sanity tested)*
- [x] Do `sanitize_repo_id` + the feature-group download plan match the cache-bridge contract? *(`sanitize_repo_id` parity Python↔Kotlin via shared oracle; `downloadPlan` keyed by `FEATURE_GROUPS`)*
- [x] Are checksums + per-file sizes present and schema-versioned? *(`build_manifest` stream-hashes `sha256`/`fileSizes`; per-variant `checksums.json`; `schemaVersion`/`minReaderVersion`)*

### #15 — One-command export CLI (`02_code_plans/05`) · checkpoint
**Done 2026-07-14 (Python; automated checkpoint leg, no device).** `export/pipeline.py` (`plan_export`/`export_package`/`assemble_package`) + `export/model_card.py` + `cli/export.py` + `cli/push.py` wired into the dispatcher.
- [x] Does one command go HF model → validated device-ready package? *(dry-run plans it; `assemble_package` reshapes stage outputs → #14 tree → validates against #13. The real full-model export (`create_model`/`gen_artifacts`) is **env-gated** (optimum + ORT-training profiles), not run in CI — this is env-gated, not device-gated)*
- [x] Does it delegate to existing modules (no reimplementation), with CLI > env > YAML > default overlay? *(reuses #7 discovery, #9 `export_inference_package`, `build_merger_model`, `build_manifest`)*
- [x] Does the **export E2E workflow** test pass on a tiny model end to end? *(automated leg over the fixture stub → validates against #13; real-tiny-model run is the env-gated manual leg)*

### #16 — Android Gradle rename (`00_code_plans/04`)
**Done 2026-07-14 — full removal (option B), user-confirmed, superseding the doc's isolate-only option A.**
Workspace `ORTransformer`→`MobileTransformersApp`, SDK module `ORTransformersMobile`→`MobileTransformers`
(`:MobileTransformers`), SDK package `com.martinkorelic.ortmobile`→`com.martinkorelic.mobiletransformers`,
app package `com.martinkorelic.orttransformer`→`com.martinkorelic.mobiletransformers.app` (+ matching
`applicationId`). **Native fully renamed too:** `libmobiletransformers.so`, CMake `project("mobiletransformers")`,
`loadLibrary("mobiletransformers")`, all 22 JNI symbols (SDK classes→`Java_com_martinkorelic_mobiletransformers_*`,
MainActivity→`..._mobiletransformers_app_MainActivity_*`). Python couplings updated in lockstep:
`codegen/enums.py::KOTLIN_CONSTANTS_RELPATH`, tokenizer file `mobiletransformers_tokenizer_config.json`
(writer+reader), and `ORTransformerGenerator`→`MobileTransformerGenerator`. Zero residual
`ortmobile`/`orttransformer`/`ORT(T)ransformer` in the Android tree + live code/docs.
- [x] Do `:MobileTransformers` + `:MobileTransformersApp` assemble after the rename? *(arm64-v8a native links, `:MobileTransformers:compileDebugKotlin` + `:app:compileDebugKotlin` green; full `assembleDebug`/device install is manual)*
- [x] Is ObjectBox generated code regenerated under the new namespace without breakage? *(objectbox codegen regenerates under `com.martinkorelic.mobiletransformers.entity`; verified via compileDebugKotlin)*
- [x] Are the JNI symbol decisions (rename vs alias) documented and does the app still link? *(chose rename-all, not alias; each symbol mangled from its class's new package; `libmobiletransformers.so` links on arm64-v8a)*

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
**Done 2026-07-14 (Python, no device).** `support/{statuses,models,matrix}.py` + `cli support-matrix`. Detection injectable (mocked in CI); the three ready-statuses read a device/CI probe file and degrade to `false`+blocker when absent.
- [x] Is `model_support_matrix.json` generated truth with inherited statuses + earliest-blocker attribution? *(`apply_inheritance` + `first_blocked`; list-shaped envelope owns the schema)*
- [x] Does the matrix **feed** (not duplicate) the compatibility doc (F6)? *(`filtered_docs_dict` renders the user-facing subset FROM the generated matrix)*

### #21 — Hub pull & cache flow (`02_code_plans/04`) · checkpoint
**Done 2026-07-14 (Python-first, no device).** `hub/variant_select.py` (`Constraints` + `select_variant`: soft quant preference, download-size tie-break, 0.9× storage budget, over #13's hard filter) + `hub/pull.py` (`pull_package` manifest-first + sha256-verify; `install_package` = Python cache-bridge, tokenizer-flatten, atomic `.partial`→`os.replace`) + `cli pull`/`install-package`.
- [x] Does Python `pull_package` / `install_package` produce the exact cache layout the SDK loads? *(install smoke asserts `train/`,`inference/`,`embedding/`,`tokenizer/` + flattened tokenizer)*
- [x] Is variant selection identical Python↔Kotlin (deterministic)? *(Python done + unit-tabled; the Kotlin `VariantSelector` parity is the device leg — deferred)*
- [x] Does the **pull → materialize → load** workflow pass? *(pull→install→sha256 automated over the fixture; the on-device load leg is Manual — deferred)*

### #22 — Adapter push-back (`02_code_plans/06`)
**Done 2026-07-14 (Python-first, no device).** `adapter/{export,convert,model_card}.py` + `cli push-adapter`. Gate is pure metadata; safetensors materialization is `torch`/`peft` env-gated.
- [x] Does export produce a PEFT-compatible layout when clean, else a documented native fallback? *(`to_peft_layout` → Mode-1 `adapter_config.json` for clean LoRA; MARS / factor-less LoRA → Mode-2 native subtree + `mobiletransformers_adapter.json`; `--peft-only` errors)*
- [x] Is Android upload gated / disabled by default with a privacy warning? *(card carries the bold privacy warning, asserted before upload; on-device `AdapterUploader.kt` is the deferred, gated device leg — default path is device→desktop→`push-adapter`)*

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
- [ ] Is the code relicensed to Apache-2.0 with SPDX headers on **first-party source only** (vendored Microsoft/tokenizers/proto code untouched, enumerated in `THIRD_PARTY_NOTICES.md`), the `pyproject.toml` license expression set, and all rights-holders' agreement?
- [ ] Do **all version sites agree** (pyproject == `__version__` == Gradle `-Pversion` == `CITATION.cff` == tag), and are CHANGELOG non-goals listed?
- [ ] Does the **full release gate** (CI green + AAR + consumer smoke + docs + tag) pass?

### #33 — Encoder-model support (`04_code_plans/01`)
- [ ] Did the spike prove export + a train step + Android smoke + a metric for one small encoder?
- [ ] Is encoder support a `TASK_REGISTRY`/architecture-registry entry (no new `if/elif`, no KV-cache) (F3)?
- [ ] Was MARS-transfer-to-encoder-linear-layers **verified**, not assumed?

### #34 — Charging-cycle training scheduler (`04_code_plans/02`) · checkpoint
- [ ] Is `LinearLRScheduler.stateDict()`/`loadFromState()` implemented (`ORTScheduler.kt:156-161` `TODO` closed — `CosineLRScheduler` already implements both at `:77`/`:111` and is wired via `training_state.json`), and is the restore path verified to survive WorkManager chunk boundaries and process death?
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

# Implementation Order — Code Plans Index

This index orders every feature code-plan under `agent_docs/00_code_plans/` … `agent_docs/05_code_plans/` from **most foundational first**. Implement in this order; each plan file repeats its own Prerequisites / Blocks header.

The high-level "what & why" lives in the six tier docs (`00_repository_restructure_plan.md`, `01_tier0_foundation_decisions.md`, `02_tier1_hf_integrated_core.md`, `03_tier2_inference_and_rag.md`, `04_tier3_reach_extensions.md`, `05_cross_cutting_release_modernization.md`). These code plans are the "how" — concrete enough for another agent to implement.

> **Status as of 2026-08-07.** A six-agent audit (`agent_docs/audits/`) found several `[x]` marks
> overstated; a remediation pass then closed most of what it found. The `Done` column and the
> self-checks below have been corrected against the **verified** state — where a box now reads `[x]`
> it is because the DoD holds today, not because a session claimed it. Boxes still `[ ]` are honest
> remaining work, and each says what is missing.
>
> The tree **is** committed (the "nothing is committed" premise repeated in older session logs is
> false). Gate status is recorded at the end of `HANDOFF.md`.

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

> **Device acceptance 2026-08-08** (Galaxy S21 FE, Android 15, arm64-v8a). #11/#17/#23/#24/#25/#26/#27
> ticked: their instrumented legs ran green over a real exported package (`make device-package` →
> `make device-test`, 7 pass / 1 skip). #9/#18/#19 stay open — they ride on `TrainMergeGenerateTest`,
> which still skips until a `TRAIN=1` (ort-training-local) package exists. #12 stays open: the RSS
> probe and the mmap system-property toggle now exist, but the four-point Gate 0.2 table is unwritten.
> Ten export↔runtime defects were found and fixed getting there — see HANDOFF.md.


| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [x] | 1 | `00_code_plans/01_python_package_and_uv_scaffolding.md` | Unblocks all Python work | — |
| [x] | 2 | `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md` | Lock the toolchain before building on it (incl. `pydantic>=2` core) | 1 |
| [x] | 3 | `01_code_plans/06_source_built_ort_training_pipeline.md` | Prove the train toolchain is alive (`generate_artifacts`) | 2 |
| [x] | 4 | `00_code_plans/02_config_layering_settings_constants.md` | Three config layers + secrets; needed by export/CLI | 1 |
| [x] | 5 | `00_code_plans/10_python_code_quality_and_module_health.md` | **Conventions, typing, lint, shared logging/exceptions + the monolith-decomposition strategy** — established before registries/merger/builders so they inherit it | 1, 2, 4 |
| [x] | 6 | `00_code_plans/09_typed_models_enums_and_registries.md` | **Typed config + enums + PEFT/arch/merger registries** — kills hardcoded dispatch before builders are written | 2, 4, 5 | *(contract layer + consumption both landed 2026-08-07; `inference/builder.py`'s arch ladder is the one tracked remainder — allow-listed with an owner)* |
| [x] | 7 | `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md` | Inference-export front door (arch via registry) | 2, 4, 6 |
| [x] | 8 | `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` | Data contract every later piece reads (codec consumes registries) | 4, 6 | *(Python owner layer done: schema + codec + `check_compat`; C++/Kotlin consumers ride with #9/#23)* |
| [ ] | 9 | `01_code_plans/01_unified_merger_and_external_data_export.md` | **Dual-engine core** + full merger unification | 6, 7, 8 | *(code-complete 2026-07-14: Python A/B/C tested, C++ D/E compile+link-verified on arm64-v8a; box stays open pending the manual on-device parity/atomic/load-smoke tests — see #9 self-check)* |
| [x] | 10 | `01_code_plans/02_genai_external_data_swap_spike.md` | Feeds Gate 0.1 | 9 | *(2026-07-15: **Gate 0.1 = ADOPT GenAI.** F2 validated — external-data swap changes GenAI output on device (token/fp differ) + desktop; symbol/fork-only confirmed; RSS measured (mmap). The one blocker — ORT-runtime coexistence (genai needs stock ORT ≥1.26, Native needs training ORT 1.23) — **resolved** via `libort_gen.so` distinct-soname separation, verified both coexist on device. Cross-engine #1/#4 (same package under BOTH engines) ride with #11's dual-engine smoke + a real #9 package. See `spikes/genai_external_swap/README.md`.)* |
| [x] | 11 | `01_code_plans/03_inference_engine_abstraction_native_and_genai.md` | Engine selection | 10 | *(code-complete 2026-07-15: `runtime/ModelRuntime.kt` (interface + `EngineCapabilities` + `EXECUTION_PROVIDER_REGISTRY` F3 + `GenAiSupport` + `ModelRuntimeFactory` pure-select + device-create w/ transparent Native fallback); `ORTGeneratorGenAI.kt` + `cpp/genai_runtime.cpp` (streaming, callback parity); `ORTGeneratorNative` adapted to `ModelRuntime`; `engine` field in `ORTGenerationConfig`; `LLMRepository` wired via factory; dead `ORTGenAINative.kt`+`onnx-genai.cpp` deleted. Compiles+links arm64, **48 JVM tests**, device build loads. Box open pending the dual-engine same-folder device smoke + streaming-parity harness — need a real #9 package.)* |
| [ ] | 12 | `01_code_plans/04_memory_mapping_experiments.md` | Optimizes 9–11 (non-blocking) | 9 | *(code-complete 2026-07-15: `cpp/mem_probe.h` (RSS) + `cpp/mmap_tensor.h` (RAII) + a default-off `MTF_MMAP_WEIGHTS` zero-copy branch in `session_cache.h` (copy path stays the shipping default, #23 unaffected); `spikes/mmap/{measure_rss(re-export),base_blob_mmap_spike}.py` (desktop byte-identical correctness invariant). arm64 links. Device 4-point RSS table = the manual Gate 0.2 leg.)* |
| [x] | 13 | `00_code_plans/06_manifest_first_package_and_cache_bridge.md` | Package contract | 8, 9 | *(done 2026-07-14: Python `artifacts/manifest.py` + Kotlin `packages/` cache-bridge (6 classes), JVM-tested + `compileDebugKotlin`; only the on-device generate smoke deferred)* |
| [x] | 14 | `02_code_plans/03_hub_model_package_format.md` | Hub repo shape | 13 | *(done 2026-07-14: `hub/package_format.py` — `sanitize_repo_id`, `build_manifest`, tiny_package fixture; no device)* |
| [x] | 15 | `02_code_plans/05_one_command_export_cli.md` | Wraps 7–9 + 13 *(checkpoint: export E2E)* | 7, 9, 13 | *(done 2026-07-14; 2026-07-15: `_full_export` real inference+GenAI leg implemented (stage-gated) + on-box verified — SmolLM2-135M → #13-valid package, GenAI loads it; training stage staged for the ort-training-local run — see #15 self-check)* |
| [x] | 16 | `00_code_plans/04_android_gradle_rename_migration.md` | Isolated, verified rename | — | *(done 2026-07-14: **full removal / option B** — Kotlin+native+JNI all renamed off `ortmobile`; supersedes the doc's option-A. Compile+link-verified arm64-v8a, `compileDebugKotlin` + `make parity` green.)* |
| [x] | 17 | `00_code_plans/05_android_facade_foundation.md` | Public SDK facade *(checkpoint: load→generate)* | 13, 16 | *(code-complete 2026-07-15: `MobileTransformers.fromPretrained`/`MobileTransformerModel`, public `config/*` + `runtime/{ModelSession,RuntimeCapabilities,Results}`, `ConfigMappers`, `RepositoryBackedModelSession`, `ModelFeature`, exception stubs; 13 JVM tests (config round-trip, feature/engine, variant-select, facade delegation). Box open pending the device load→generate leg. `InferenceEngine` placeholder retired by #11.)* |
| [ ] | 18 | `00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md` | Job/progress/checkpoint API | 17 | *(code-complete 2026-07-15: `training/` `TrainingJob`/`TrainingStatus`/`TrainingEvent`/`CheckpointInfo`/`TrainingJobManager`(+`TrainingJobSpec`) + `TrainingEventAdapter`; `ORTTrainerNative` cooperative `cancelRequested` (no format change); `TrainingResult` enriched. 5 JVM tests (event mapping, checkpoint round-trip). Box open pending device resume/summary/train→merge→generate legs.)* |
| [ ] | 19 | `02_code_plans/01_hf_style_kotlin_facade.md` | `fromPretrained`/train/merge/generate *(checkpoint: train→merge→generate)* | 11, 17 | *(code-complete 2026-07-15: DELTA over #17 — `applyPeft`/`pushAdapter` + public callbacks + full sealed exception hierarchy + sealed `PeftConfig`/`PeftSupport` + engine/merge-driven `ConfigMappers` + construction-time feature/GenAI gates; 87 SDK JVM tests + sample-app compile green; box open pending the device train→merge→generate workflow — see #19 self-check)* |
| [x] | 20 | `02_code_plans/02_optimum_support_matrix.md` | Reporting layer | 7 | *(done 2026-07-14: `support/` package (statuses/models/matrix) + `support-matrix` CLI; inherited statuses, probe ingestion, filtered docs; detection injectable/mocked in CI, ready-statuses read device probes when present)* |
| [x] | 21 | `02_code_plans/04_hub_pull_and_cache_flow.md` | Python pull first, Android downloader next *(checkpoint: pull→load)* | 13, 14 | *(done 2026-07-14: Python `hub/{variant_select,pull}.py`. 2026-07-15: **Android downloader** — `hub/{HubResolver,DownloadPlanner,PackageDownloader,HubDownloader,PackageDownloadWorker}` (OkHttp + WorkManager, reuses `packages/` verify/select/install), `fromPretrained` triggers pull-then-load; JVM/MockWebServer tests. WorkManager scheduling + real network = device leg.)* |
| [x] | 22 | `02_code_plans/06_adapter_pushback.md` | Last Tier-1 piece | 9, 14 | *(done 2026-07-14; 2026-07-15 Python `materialize_peft_weights`. 2026-07-15: **Android `hub/AdapterUploader`** — cache→AdapterPackage + Mode-1/2 gate + privacy-gated card (fail-closed sections), default-off `BuildConfig.ADAPTER_UPLOAD_ENABLED`; fills the `pushAdapter` stub. JVM tests. Real authenticated upload + checkpoint factor read = device leg.)* |

### Tier 2 — Inference & RAG (Phase 7)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [x] | 23 | `03_code_plans/01_inference_handoff_alignment_and_native_hardening.md` | Wire Native into `ModelRuntime`, retire `inference/merged/` | 11, 9, 8 | *(code-complete 2026-07-15: fail-closed map-driven load — Kotlin `HandoffPrecondition` + shared C++ `handoff_io.h`/`session_cache.h`; conversation-reset fix; host JVM tests + arm64 link green; box open pending device load/generate/reset smokes — see #23 self-check)* |
| [x] | 24 | `03_code_plans/02_sampling_and_streaming_public_config.md` | HF-aligned generation config (enum-typed) + callback parity | 23, 19 | *(code-complete 2026-07-15: `SamplingMethod.nativeOrdinal` + `methodMap` retired + `maxNewTokens` lock; host JVM tests + enum parity green; box open pending cross-engine callback-parity device leg)* |
| [x] | 25 | `03_code_plans/03_vector_store_boundary_and_inmemory.md` | Testable `VectorStore`; `SearchType` enum; dynamic dimension registry | 17 | *(done 2026-07-14: `rag/` VectorStore boundary + ObjectBoxVectorStore + test-only InMemoryVectorStore + DimensionRegistry + VectorStoreRegistry (F4); `ORTRetriever` routes through it; `RagResult`→`RagMatch`; 23 JVM tests + `compileDebugKotlin` (both modules) green. #17 prereq nominal — wraps existing classes, no facade code. No device.)*
| [x] | 26 | `03_code_plans/04_rag_ingestion_and_chunking.md` | Implement `ingestData()` | 25, 23 | *(code-complete 2026-07-15: `rag/DocumentChunker` + `DocumentSource`/`DOCUMENT_LOADER_REGISTRY` (F3, txt/md/jsonl; PDF/Word rejected) + `IngestionProgress` + pure `IngestionPipeline` seam; `ORTRetriever.ingestData` binds the real embedder; `RagRepository.ingest` + facade `ingest`. JVM tests (chunker/loader/pipeline over InMemoryVectorStore). Device ingest smoke = `RagDeviceTest`.)* |
| [x] | 27 | `03_code_plans/05_rag_config_and_grounded_generation.md` | Public `RagConfig` + grounded flow *(checkpoint: ingest→retrieve→generate)* | 26, 24, 21 | *(code-complete 2026-07-15: `rag/PromptAssembler` + `GroundedResult` + facade `generateWithRag` (retrieve→assemble→generate, inspectable prompt); `RagConfig`/`ORTRagConfig` +`minScore`/`indexingMode`, new `IndexingMode` enum (Python+Kotlin+parity), F7 dynamic fail-closed; fixed `makeOrtRag`/`prepareRetriever` override + `minScore` threading. JVM tests (mapper/prompt/grounded flow). Device checkpoint = `RagDeviceTest`.)* |

### Cross-cutting — Release modernization (Phase 8; runs alongside, gated)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [x] | 28 | `05_code_plans/01_makefile_and_cli_entrypoints.md` | One-command path | 1, 15, 2 | *(done 2026-07-14: real Makefile — thin wrappers over the `mobiletransformers` CLI + Gradle, `make help` self-documents, profile-isolated `setup*`, non-destructive `clean-generated`; `scripts/{android_build_aar,publish_local_maven,run_smoke}.sh` stubs (bodies owned by #30))*|
| [x] | 29 | `05_code_plans/02_ci_staged_pipeline.md` | Standing "it works" proof (incl. lint/typecheck/parity gates) | 28, 1, 3 | *(done 2026-07-14: `.github/workflows/ci.yml` (fast → export-smoke → android-assemble, `fail-fast:false`, per-job `timeout-minutes`) + `device.yml` (dispatch + nightly). fast+export-smoke run in CI; android-assemble self-skips without the git-ignored vendored native deps, exactly like `ort-training-smoke.yml`. No PR job downloads a large model.)*|
| [ ] | 30 | `05_code_plans/03_aar_maven_publication.md` | Portable Android consumption *(checkpoint: consumer app builds)* | 16, 28 |
| [ ] | 31 | `05_code_plans/04_docs_set_and_compatibility_matrix.md` | Public docs + registry-driven matrix as contracts stabilize | 23-27, 19, 13 | *(partial 2026-07-14: `docs/EXPORT.md`, `docs/RAG.md` (#25 scope), `docs/PUBLIC_API.md`, generated `docs/COMPATIBILITY_MATRIX.md` + `support/render.py` renderer + `support-matrix --md` + drift test; `CHANGELOG.md` + `docs/RELEASE_CHECKLIST.md` skeletons. `docs/MODEL_FORMAT.md` + `docs/CONFIGURATION.md` added 2026-07-15 (contracts locked). Remaining pages (ARCHITECTURE/ANDROID_SDK) await #23/#24/#30; box stays open.)*|
| [ ] | 32 | `05_code_plans/05_versioning_license_release.md` | v1.0 release gate *(checkpoint: full release)* | 29, 30, 31 |

### Tier 3 — Reach extensions (Phase 9; each spike-gated, never blocks v1.0)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [ ] | 33 | `04_code_plans/01_encoder_model_support.md` | Cheapest reach; arch-registry entry + codec/manifest reuse | 13, 7, 9 |
| [ ] | 34 | `04_code_plans/02_training_scheduler_workmanager.md` | Charging-cycle training; WorkManager scheduling (the `LinearLRScheduler` state-persistence gap it was written against was closed in the 2026-08-07 remediation pass; `ORTScheduler.kt:161-162` records the fix) *(checkpoint: scheduled train→resume)* | 18, 17 |
| [ ] | 35 | `04_code_plans/03_federated_codec_and_python_simulation.md` | Python Flower sim (Option A first) *(checkpoint: N-client sim)* | 8, 9, 13 | *(code-complete 2026-07-15: `federated/` `FederatedAdapterRecord` (codec-derived, pinned byte serialization = #36 golden), `flower_sim.py` pure `federated_average` + `cli federated simulate`; `flower_client.py` lazy-flwr. Tests `tests/federated/` (roundtrip/format-version/comm-size/fedavg/dropout + byte golden) run in core env. flwr kept OUT of the universal lock (out-of-band, like the ORT wheel). Box open pending the manual N-client ORT-`fit` sim + aggregated-adapter logits smoke — runnable under the ORT-training profile.)* |
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
- [x] Can a new PEFT / architecture / merger be added with **only** a registry entry + enum member (zero new `if/elif`; grep for survivors)? *(2026-08-07: the consumption half landed. `trainer/builder.py` parses `PEFTMethod` once at the boundary; `weight_merger.cpp` dispatches on a typed `MergerVariant` (`cpp/constants/merger_variant.h`, googletest-covered); the duplicated MARS/ABLATION target tables collapsed onto one registry source; `build_adapter_mapping` — the normative A3 symbol that previously existed nowhere — is the single mapping entry point. `make guard` now greps `src/`, the legacy roots AND `cpp/`. The ONE remainder is `inference/builder.py`'s 14-branch `architectures[0]` ladder, which needs 7 new registry rows and is unimportable under every declared profile: allow-listed with a named owner in `tests/unit/test_guards.py`.)*
- [x] Is every closed string set an enum mirrored Python↔Kotlin, proven by the CI parity test?
- [x] Does `model_dump(by_alias=True)` round-trip through the generated schema that Kotlin/C++ validate?
- [x] Did `build_merger_model(MergerSpec)` replace the `create_*_merger_model{,_2}` duplication? *(wired by #9 2026-07-14: single builder + golden-equivalence test vs the legacy `*_2` factories; the four factories are deleted)*

> Done 2026-07-13 (owned contract layer). **A2 enums** — 11 `str,Enum` classes in `config/constants.py` (`SamplingMethod`,`SchedulerType`,`ExecutionProvider`,`CoreConfigId`,`MemoryConfigId`,`SearchType`,`QuantizationType`,`PEFTMethod`,`TaskType`,`HandoffMode`,`MergerVariant`) + `ENUM_REGISTRY`; `SUPPORTED_PEFT_METHODS` now derives from `PEFTMethod`. **A1 Pydantic v2** — `config/models.py` (`SamplingConfig`,`DeviceOptions`,`Linear/CosineScheduler` discriminated union,`QuantizationOptions`,`GenerationConfig`,`TrainingConfig`,`RagConfig`), camelCase aliases, `extra="ignore"` (unknown fields tolerated / unknown enum values fail closed), `schemaVersion`+`minReaderVersion` block. **A3/A4/A5 registries** — `config/registry/{peft,architecture,merger}.py` with lazy dotted-path class binding (core-importable, no optimum/torch at load); `resolve_architecture` covers all 8 legacy training arches, `get_peft_spec` all 5 methods, `resolve_merger` all variants. **Parity** — `python -m mobiletransformers.codegen.enums` generates checked-in `schemas/*.schema.json` + golden `schemas/enums.json`; `--check` (the `make parity` gate) diffs them + the 11 hand-mirrored Kotlin `constants/*.kt` `fromWire` enums (under the pre-rename path) and fails on drift. Tests: `test_config_models`/`test_registries`/`test_enum_parity` + an `src/`-scoped dispatch grep guard. `make check` (lint+typecheck+parity+71 tests) green.
> **Consumption half landed 2026-08-07.** The audit found this deferral had gone ownerless: #7 and #9 both closed without doing it. What landed: (a) **C++** — `merger_type == "lora"` string dispatch at five sites replaced by a typed `MergerVariant` mirrored in `cpp/constants/merger_variant.h`, with `merger_sessions_` keyed by the enum so an unknown tag in the handoff map fails at *load* time; (b) **`trainer/builder.py`** — the `train_method ==` chain and its mapping sub-ladder replaced by one boundary parse plus `build_adapter_mapping`; (c) **the duplicated target tables** — `peft_models/{mars,ablation}/utils.py` were byte-identical copies; both now re-export one `PEFT_TARGET_MODULES_BY_MODEL_TYPE`. NOTE they were deliberately NOT folded onto `ArchitectureSpec.target_modules`: the two are keyed differently (`model_type` vs `architectures[0]`) and the PEFT table is far wider, so folding would have silently dropped 13 model types; (d) **Kotlin** — `FileUtil.kt`'s silent `println`-and-default scheduler parse is gone, and every closed-set field is validated through its enum's `fromWire` at the parse boundary (field *types* stay `String` because they cross JNI as strings); `ORTRetriever` dispatches on `SearchType`. Remaining: `inference/builder.py`'s ladder (allow-listed, shrink-only ratchet) and broadening top-level `__all__` (still #32's call).

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
> **Deferral expired.** The Optimum-vs-GenAI decision was made 2026-07-15 (Gate 0.1 = ADOPT), which
> ungated this; the audit found it had become ownerless. As of 2026-08-07 the ladder is the single
> tracked #6 remainder, allow-listed with a named owner + a shrink-only ratchet in
> `tests/unit/test_guards.py`, and the file itself moved out of scope for the wheel (it is not on the
> export path). Original note follows.
>
> `inference/builder.py` dispatch rewrite still gated by the Optimum-vs-GenAI decision (not #7); the
> `torch.onnx` frontend body only if a future optimum removes `export()`.

### #8 — Weight handoff map & tensor codec (`00_code_plans/07`)
- [x] Is `weight_handoff_map.json` the sole source of tensor identity (no hard-coded `replace_prefix` on the load side)? *(Python contract + C++ both sides: #23 landed the load-side rewrite — `session_cache.h` now derives names from `inferenceInitializerNames[role]` via the shared `handoff_io.h` reader, no `<dirname>.<filestem>` reconstruction)*
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
- **Outstanding (manual/device):** on-device atomic-overwrite-under-kill, offline-vs-device byte-identical `.bin` parity, native load-and-generate smoke. The native **load** side migration is **DONE (#23, 2026-07-15)** — `ORTGeneratorNative` + `session_cache.h` now consume the handoff map fail-closed (map-driven flat `<name>.bin`, no `inference/merged` probe); a device build now sees #9's in-place merges.
- **Env note:** the native build needs the untracked vendored deps (`cpp/includes/google` protobuf headers, `jniLibs/`, `aarLibs/`) — all `.gitignore`d (aarLibs added this session); they were provisioned locally from the sibling `../ORTTransformer` checkout for the compile-check.

### #10 — GenAI external-data swap spike (`01_code_plans/02`)
- [x] Does the spike show a `.bin` swap is observable in GenAI output (or a clear FAIL)? *(YES — device: token 28→6156, fp 1.518e8→9.82e7 on a fresh `OgaCreateModel`; desktop `|ΔL|=39.6`)*
- [x] Is RSS measured and the `OgaCreateModelWithInitializers` symbol presence verified on the Android lib? *(RSS measured — mmap/lazy, not 2× copy; `check_symbols.sh`: `OgaCreateModelWithInitializers` ABSENT, `OgaCreateModel` present)*
- [x] Is the Gate 0.1 PASS/FAIL decision recorded with evidence? *(**ADOPT GenAI** — `spikes/genai_external_swap/README.md`)*

**Done 2026-07-15 (device-verified).** Gate 0.1 = ADOPT. F2 validated on device + desktop. The blocker
(genai needs stock ORT ≥1.26, Native needs training ORT 1.23; both share soname `libonnxruntime.so`) was
**resolved** with engine separation: ship the genai-paired stock ORT 1.27 as `libort_gen.so` (distinct
soname, raw-patched — patchelf corrupts verneed) + raw-patch the genai `.so`'s dlopen target; training ORT
stays `libonnxruntime.so`. Only ~3 exported symbols per ORT (hidden visibility) + dlsym-on-handle → no
interposition. Verified both ORTs coexist in one process on device. Reproducible via
`spikes/genai_external_swap/setup_ort_separation.sh`. Cross-engine #1/#4 (same package under BOTH engines)
ride with #11's dual-engine smoke + a real #9 package.

### #11 — Inference engine abstraction (`01_code_plans/03`)
- [x] Is there **one** `ModelRuntime` with Native (guaranteed) + GenAI (opt-in) over the **same** package? *(`ModelRuntime` interface; `ORTGeneratorNative` (NATIVE) + `ORTGeneratorGenAI` (GENAI); `ModelRuntimeFactory.create` over one `inference/` dir)*
- [x] Do both engines emit an identical callback sequence (parity lock)? *(**PROVEN**, S21 FE / Android 15 / arm64, 2026-08-08: `DualEngineParityTest.bothEnginesEmitTheSameOrderedCallbackSequence` records the ORDERED event names for the same package under both engines and asserts the lists are equal. It also checks the contract itself on Native first — opens with `onStartGeneration`, closes with `onCompletion`, both exactly once, with partials between — so two engines that were identically wrong could not pass as parity, and re-asserts `capabilities.engine` so a fallback cannot fake it.)*
- [x] Does engine selection fall back to Native transparently when GenAI is unavailable? *(`ModelRuntimeFactory` pure `selectEngine` + device `create` catch→Native; `RuntimeSelectionTest` covers the matrix; GenAI failure never reaches the caller)*

**Code-complete 2026-07-15.** 48 SDK JVM tests, compiles+links arm64, device build loads. `EXECUTION_PROVIDER_REGISTRY` (F3) data-driven; dead `ORTGenAINative.kt`+`onnx-genai.cpp` deleted. Box open pending the streaming-parity harness + dual-engine same-folder device smoke — both need a real #9 package (the builder model is GenAI-format only).

### #12 — Memory-mapping experiments (`01_code_plans/04`)
- [x] Do the experiments produce RSS numbers feeding the Gate 0.2 decision? *(full 2x2 table collected via `make device-rss`, S21 FE / Android 15 / arm64, 2026-08-08. **Gate 0.2 FAILS as specified**: 6.3% peak reduction on Native, -0.1% on GenAI, against 15% required — mmap covers only the 8.2% of weight bytes in the trainable split, and GenAI never routes through `WeightSessionCache` at all. See `01_tier0_foundation_decisions.md` Gate 0.2 RESULT.)*
- [x] Is mmap kept non-blocking (an optimization, not a v1 requirement)? *(yes — default-off behind `debug.mtf.mmap_weights`; Gate 0.2's measured FAIL therefore does not block v1.)*

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
**Done 2026-07-14 (checkpoint leg); real `_full_export` inference leg landed 2026-07-15.** `export/pipeline.py` (`plan_export`/`export_package`/`assemble_package`) + `export/model_card.py` + `cli/export.py` + `cli/push.py` wired into the dispatcher.
- **`_full_export` (2026-07-15):** now a **stage-gated orchestrator** (inference | training | embedding), injectable builders, effective features/engines computed from what's on disk (never claims a missing subtree). The **inference stage is real** (`export` profile): optimum `export_inference` → normalized `model.onnx`/`model.onnx_data` + tokenizer + `generation_config.json`, an empty (all-frozen) `weight_handoff_map.json`, and a self-contained `genai_config.json` built from HF `AutoConfig` + the fixed canonical IO scheme (the vendored `inference.builder` can't import under `export` — it needs an onnxruntime symbol absent there). **Verified on-box:** `mobiletransformers export --model HuggingFaceTB/SmolLM2-135M-Instruct --output build/pkg --genai` → `#13` manifest validates (`features=[core,inference,genai]`), and GenAI `OgaCreateModel` loads the same `inference/` dir and produces logits (`desktop_spike`, genai-smoke). CI: `tests/export/test_full_export_orchestration.py` (orchestration over injected fake builders). The **training stage** (`gen_artifacts` + `export_inference_package` trainable split/handoff map) was **implemented 2026-07-15** (was a seam): it runs under a separate `ort-training-local` invocation into the same `--output`, writes `train/` + overwrites the empty handoff map with the real trainable split; `_effective_features` unions on-disk state so the training-only re-assembly keeps `inference`/`genai`. Env-gated `tests/integration/test_training_stage_smoke.py` covers the `gen_artifacts` leg; the full `optimum_hf_export` real-model run is the `make device-package TRAIN=1` leg. This unblocks #19 train→merge→generate.
- [x] Does one command go HF model → validated device-ready package? *(dry-run plans it; `assemble_package` reshapes stage outputs → #14 tree → validates against #13. Real inference+GenAI export on-box verified; the training stage is implemented (env-gated smoke) + produced via the two-profile `make device-package TRAIN=1` run)*
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
- [x] Does `MobileTransformers.fromPretrained(...)` return a working `MobileTransformerModel` wrapping the existing repositories (no JNI rewrite)? *(`RepositoryBackedModelSession` wraps `LLMRepository` + 3 sub-repos; no `ORT*`/`Job` in public signatures; delegation covered by `FacadeDelegationTest`. Actual on-device load→generate is the outstanding leg)*
- [x] Are the engine selector + exception hierarchy in place? *(`InferenceEngine`/`RuntimeCapabilities` + `ModelFeature` engine-selector semantics tested; `MobileTransformersException` base + `ModelNotInstalled`/`MissingArtifact` stubs — full hierarchy is #19)*
- [x] Does the facade workflow (load → generate one token, device-manual) pass? *(`FacadeLoadGenerateTest` PASS, S21 FE / Android 15 / arm64, 2026-08-08)*

**Code-complete 2026-07-15.** 13 JVM tests (config round-trip == `ORT*Config` defaults, feature/engine semantics, manifest variant-select, facade→session delegation via a hand-written fake). Box open pending the device load→generate checkpoint.

### #18 — Training lifecycle & checkpoint contracts (`00_code_plans/08`)
- [x] Does `TrainingJob` expose status/events/checkpoint without hiding the native lifecycle? *(`TrainingJob` + `TrainingStatus`/`TrainingEvent` (StateFlow/SharedFlow) + `CheckpointInfo`; `TrainingEventAdapter` maps the `TrainingCallback` 1:1 — `TrainingEventAdapterTest` asserts order/transitions)*
- [x] Is the callback→event adapter complete and the checkpoint file format preserved? *(adapter complete; `CheckpointInfo` is a read-only Gson projection of `training_state.json` — `CheckpointInfoTest` asserts the on-disk JSON is unchanged after read)*
- [x] Are the session lock + cooperative cancellation defined for reuse by the scheduler (#34)? *(cooperative cancel: `ORTTrainerNative.@Volatile cancelRequested` checked at the step/epoch loop tops, no format change. **The session lock did not exist when this was first ticked** — the audit found zero mutual exclusion anywhere in the library. Added 2026-08-07: `LLMRepository.sessionLock` (a `Mutex`) guards all four native session create/teardown paths, held INSIDE the launched coroutine (wrapping the `Job`-returning function would release it before any handle was touched) and deliberately not across a full training run, so a long job never blocks `release`. `llmState` is now `@Volatile`.)*

**Code-complete 2026-07-15.** 5 JVM tests. `TrainingResult` (the #17 type) enriched with `checkpoint`/`summary`. Box open pending the device resume/summary/train→merge→generate manual legs.

### #19 — HF-style Kotlin facade (`02_code_plans/01`) · checkpoint
**Code-complete 2026-07-15 (host-verified; box open pending the device workflow).** DELTA over #17:
`MobileTransformerModel` gained `applyPeft`/`pushAdapter` + public `callback:` params; the full sealed
exception hierarchy landed in `MobileTransformersException.kt` (`PeftMismatch`/`FeatureNotInstalled`/
`EngineUnavailable`/`NotImplementedFeature` + the #17 base/`ModelNotInstalled`/`MissingArtifact`); flat
`PeftConfig` → sealed `config/PeftConfig.kt` (`Lora`/`MarsOpt0/1`/`MarsQuantized`) with pure
`internal/config/PeftSupport.kt` taxonomy+validation; public `TrainCallback`/`GenerateCallback`/
`RetrieveCallback` + payloads mapped 1:1 from the repository callbacks (streaming = callback, tokens
forwarded while accumulated); `ConfigMappers` drives `type` from the engine + `loadMergedWeights` from
merge state and adds `DatasetConfig.toOrt()`; `fromPretrained` adds construction-time feature-gate +
GenAI-config gate; `pushAdapter` is a `NotImplementedFeatureException` stub (rides #22). `PeftMappingTest`
/`ConfigMappingDeltaTest`/`ExceptionMessageTest`/updated `FacadeDelegationTest` (host JVM) green; sample
app still compiles.
- [x] Do `applyPeft`/`train`/`merge`/`generate`/`retrieve` map cleanly onto existing repositories? *(RepositoryBackedModelSession delegates all five; applyPeft validates via PeftSupport)*
- [x] Are public config names HF-aligned and mapped to internal config (mapping table present)? *(ConfigMappers + PeftSupport taxonomy; delta tests assert engine→type, merge→loadMergedWeights, dataset, sampling)*
- [x] Does the **train→merge→generate** workflow (device-manual) pass end to end? *(`TrainMergeGenerateTest` PASS, S21 FE / Android 15 / arm64, 2026-08-08: 60/60 trainable `.bin` files rewritten by the merge. NOTE the evidence is that the merge HAPPENED — byte-fingerprint before/after — not that it is numerically sane; one LoRA step on an 8-row fixture yields `,,,,,,,,`.)*
> Notes: manifest carries no `peftMethods` — PEFT support is derived from `train/training_config.json`
> alone (Gson, testable); GenAI availability is a `fromPretrained` file check (no new `LLMRepository`
> flag). `compat/LegacyAliases.kt` skipped: #16 fully retired the `ortmobile` brand, so there are no
> consumers to alias. Sample-app ViewModel migration to the facade handle deferred (app builds unchanged;
> facade is additive).

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
**Done 2026-07-14 (Python-first, no device); CORRECTED 2026-08-07.** `adapter/{export,convert,model_card}.py` + `cli push-adapter`. Gate is pure metadata; safetensors materialization is `torch`/`peft` env-gated. **The audit found this was not actually done:** `cli/push_adapter.py` never called `materialize_peft_weights`, so a Mode-1 push published `adapter_config.json` with **no weights** — a repo `PeftModel.from_pretrained` cannot load — and the path skipped in CI so nothing caught it. Fixed: the materialization is called, and outside the `train` profile the upload now REFUSES rather than publishing an unusable adapter (`--dry-run` stays core-runnable and reports what is missing). `_read_checkpoint_factors` also normalizes ORT errors to `ExportError` so a corrupt checkpoint cannot escape the fail-closed handler.
- [x] Does export produce a PEFT-compatible layout when clean, else a documented native fallback? *(`to_peft_layout` → Mode-1 `adapter_config.json` for clean LoRA; MARS / factor-less LoRA → Mode-2 native subtree + `mobiletransformers_adapter.json`; `--peft-only` errors)*
- [x] Is Android upload gated / disabled by default with a privacy warning? *(card carries the bold privacy warning, asserted before upload; on-device `AdapterUploader.kt` is the deferred, gated device leg — default path is device→desktop→`push-adapter`)*

### #23 — Inference handoff alignment & native hardening (`03_code_plans/01`)
**Code-complete 2026-07-15 (Kotlin + C++ host-verified; box open pending device legs).** New
`internal/runtime/HandoffPrecondition.kt` (fail-closed, map-driven merged-weight gate over the #9
on-disk contract — `weight_handoff_map.json` + flat `<name>.bin` + map-`sha256`/sibling-`.sha256`) +
`packages/WeightHandoffMap.kt` (Gson read model) wired into `ORTGeneratorNative.createInferenceModel`
(retired the `inference/merged` probe) and `EngineCapabilities.supportsLoadMergedWeights`. C++: new shared
`cpp/handoff_io.h` (ONE reader — `HandoffEntry` + `load_handoff_entries` + `check_compat`, used by both
`weight_merger.cpp` and `session_cache.h`); `WeightSessionCache::init` rewritten to load flat `<name>.bin`
keyed by `inferenceInitializerNames[role]` (no `<dirname>.<filestem>` reconstruction) with dtype/shape
fail-closed validation before `AddExternalInitializers`. Conversation-reset prepend bug fixed in
`ORTConversationState.addAssistantMessage` (advance by rendered content offset, not decoded length);
`resetConversation()` now runs on `load()`. `HandoffPreconditionTest` + `ORTConversationStateTest` +
`NativeLoadRegressionTest` (host JVM) green; arm64 AAR links.
- [x] Does Native implement `ModelRuntime` with map-driven, **fail-closed** external-initializer load? *(HandoffPrecondition gate + C++ `session_cache.h` map-driven load; both fail closed naming the tensor)*
- [x] Are there zero `inference/merged/` references left, and the dead GenAI stubs deleted? *(grep regression test `NativeLoadRegressionTest`; `/merged` gone from both load sides; `ORTGenAINative.kt`/`onnx-genai.cpp` stay deleted)*
- [x] Is the conversation-state prepend bug fixed with a reset test? *(rendered-offset fix + `resetConversation()` on load; `ORTConversationStateTest` covers reset — the two-prompt device smoke is the deferred leg)*
- [x] **Device:** map-driven load-and-generate over a real #9 package (`FacadeLoadGenerateTest`); two-prompt no-leak smoke (`ConversationResetTest`); train→merge→generate reflecting merged weights (`TrainMergeGenerateTest`). All PASS, S21 FE / Android 15 / arm64, 2026-08-08.

### #24 — Sampling & streaming public config (`03_code_plans/02`)
**Code-complete 2026-07-15 (host-verified; box open pending the cross-engine parity device leg).**
`SamplingMethod` gained `nativeOrdinal` (as a `when`, not a constructor arg — keeps enum parity intact)
matching the C++ `sampling.h` enum; `ORTGeneratorNative.updateSamplingOptions` dropped the `methodMap`
magic for `SamplingMethod.fromWire(...).nativeOrdinal` (fail-closed on unknown, no silent greedy);
`maxNewTokens → maxSequenceLength` locked (+ generation-loop comment); `DECOMPOSE(#24)` retired.
`SamplingMappingTest` (host JVM) + enum parity green.
- [x] Are public sampling names HF-aligned (`SamplingMethod` enum) and mapped to internal with exact defaults? *(nativeOrdinal 0/1/2; `SamplingConfig`/`GenerationConfig` map via `toOrt`; defaults round-trip in ConfigMapperTest)*
- [x] Is the callback sequence identical across engines (parity)? *(**PROVEN**, S21 FE / Android 15 / arm64, 2026-08-08 — same test as #11 above; the ordered-event assertion that was outstanding is written and passing.)*

### #25 — Vector store boundary & in-memory test (`03_code_plans/03`)
**Done 2026-07-14 (Kotlin, JVM-tested, no device).** New `com.martinkorelic.mobiletransformers.rag`: `VectorStore` (+ `RagDocument`/`RagMatch`), `ObjectBoxVectorStore` (wraps `ORTVectorDatabase`, preserves COSINE / `1 - distance` / `minScore` / embedding-strip / text path), test-only `InMemoryVectorStore`, `DimensionRegistry` (single declared source; `ORTVectorDatabase.SUPPORTED_DIMENSIONS` delegates to it), `VectorStoreRegistry` (F4, `objectbox` default). `ORTRetriever.query` routes through the boundary; `RagResult.documents` migrated `List<Pair<VectorEntityInterface,Double>>` → `List<RagMatch>` (consumer `InferenceViewModel` updated). `compileDebugKotlin` (`:MobileTransformers` + `:app`) + 23 JVM tests green.
- [x] Does `InMemoryVectorStore` let RAG logic be unit-tested on the JVM with no ObjectBox/device? *(pure-Kotlin cosine store in the test source set; insert/search/count/topK/minScore tested)*
- [x] Is the `1 - score` distance→similarity conversion covered by an explicit test? *(`minScoreFiltersOnSimilarity` + `searchOrdersByCosineSimilarity` assert hand-computed similarities; identical→1.0, orthogonal→0.0)*
- [x] Are unsupported embedding dimensions rejected fail-closed, and backends pluggable via the registry (F4)? *(`DimensionRegistry.requireSupported` throws on dim 300; `register(301)` then accepted; `VectorStoreRegistry.create` throws on unknown key)*
- Deferred (device/later): the ObjectBox parity smoke (Android, supported dims — manual). ~~`SearchType` String→enum swap rides with #17/#19~~ — **that deferral expired** when both landed 2026-07-15 and the audit found it ownerless; done 2026-08-07 (`FileUtil` validates `searchType` through `SearchType.fromWire` at the parse boundary and `ORTRetriever.query` dispatches on the enum; the field stays `String` because it crosses JNI as one). ~~`docs/RAG.md` is #31~~ — written, and de-drifted 2026-08-07 (it had claimed #26/#27 were unimplemented long after they landed).

### #26 — RAG ingestion & chunking (`03_code_plans/04`)
- [x] Does `ingestData()` chunk + embed + store `.txt`/`.md`/`.jsonl` with progress, replacing the TODO? *(device-verified 2026-08-08: `RagDeviceTest` ingests a real `.txt` through the real embedder)*
- [x] Are loaders behind `DOCUMENT_LOADER_REGISTRY` so PDF/HTML slot in later (F3)? *(`DocumentSourceTest`)*
- [x] Is PDF/Word explicitly out of v1 scope and documented? *(rejected fail-closed by the registry; `docs/RAG.md`)*

### #27 — RAG config & grounded generation (`03_code_plans/05`) · checkpoint
- [x] Is the grounded flow inspectable (retrieve → assemble → generate, prompt visible)? *(`GroundedResult.prompt`; asserted on device 2026-08-08)*
- [x] Is `RagConfig` public and the default template overridable? *(`PromptAssembler` + `PromptStrategy`)*
- [x] Does the **ingest → retrieve → grounded-generate** workflow pass? *(device-verified 2026-08-08, `RagDeviceTest` green on a real package)*

### #28 — Makefile & CLI entrypoints (`05_code_plans/01`)
**Done 2026-07-14.** Real root `Makefile` replaces the #5 stub; `scripts/{android_build_aar,publish_local_maven,run_smoke}.sh` created (fail-closed stubs; #30 owns the AAR/Maven bodies). Console-script `mobiletransformers = ...cli.main:main` confirmed.
- [x] Are all targets thin wrappers over the CLI/Gradle (no logic), respecting profile isolation? *(`export-model`/`package-model` wrap `mobiletransformers …`; `android-build` wraps Gradle; `setup`/`setup-export`/`setup-train`/`setup-genai` each sync their own env — never the conflicting pairs)*
- [x] Does `clean-generated` never touch `cache_dir/`? *(removes only `build/`,`dist/`,`onnx_models/`,`*.egg-info`, repo-local `__pycache__`; no cache/HF paths)*
- [x] Are `lint` / `typecheck` targets present (for #5)? *(kept `lint`/`format`/`typecheck`/`parity`/`test`/`check`; `make help` self-documents every target — grep-checkable)*

### #29 — Staged CI pipeline (`05_code_plans/02`)
**Done 2026-07-14 (no device).** `.github/workflows/ci.yml` + `device.yml`. YAML parse-verified. The android-assemble job cannot be *proven* green on a hosted runner (git-ignored vendored native deps absent — same story as the ORT wheel), so it **self-skips** with a warning rather than failing spuriously; fast + export-smoke are genuinely CI-runnable.
- [x] Is CI staged cheapest-first (fast → export-smoke → android-assemble), with zoo/device nightly only? *(`needs:` chains fast→{export-smoke,android}; `device.yml` on `workflow_dispatch` + nightly `schedule`; `fail-fast:false`, per-job `timeout-minutes`)*
- [x] Does the Android assemble job work **before and after** the rename? *(targets the post-rename `:MobileTransformers`/`:app`; gated on presence of the vendored native deps so it never spuriously fails)*
- [x] Are the lint / typecheck / parity gates wired (F2)? *(fast job runs `make lint` + `make typecheck` + `make parity`; no PR job downloads a large model)*
- Deferred: how the vendored native deps + ORT-training wheel reach a hosted runner (rebuild vs. cached artifact vs. private storage) — the standing open CI-provisioning question, tied to #30.

### #30 — AAR & local-Maven publication (`05_code_plans/03`) · checkpoint
- [x] Is the missing `aarLibs/` / `libs` native input resolved first? *(resolved for **arm64-v8a**, the only shipped ABI: `make publish-local` builds and publishes `mobiletransformers-android-0.1.0.aar` + POM + sources. x86_64 remains unbuildable — no ORT/tokenizers x86_64 inputs exist — and `abiFilters` advertises arm64-v8a only rather than an ABI that would fail at `System.loadLibrary`.)*
- [x] Does the AAR publish to mavenLocal and an **external consumer app build** against it (workflow)? *(**PROVEN 2026-08-08**: `make publish-local && make consumer-app`. `examples/consumer-app` is a separate Gradle build with `RepositoriesMode.FAIL_ON_PROJECT_REPOS`, so it can only resolve the coordinates from mavenLocal. It had **no Gradle wrapper**, which is why this had never run; one was added. Result: `app-debug.apk`, 105 MB, carrying all 7 native libs out of the AAR — `libonnxruntime.so` 59.9 MB, `libort_gen.so` 28.0 MB, `libmobiletransformers.so` 6.4 MB, `libonnxruntime-genai.so` 5.6 MB, `libobjectbox-jni.so` 2.5 MB + 2 JNI shims, all `lib/arm64-v8a/`. Compilation alone would not have shown the native payload transfers.)*
- [x] Are third-party AARs (`onnxruntime-genai`) handled (vendored `.so` or explicit consumer dep)? *(**vendored**, verified by the consumer APK above: `libonnxruntime-genai.so` and `libonnxruntime-genai-jni.so` are packaged from our AAR, so a consumer declares no extra dependency. The ORT engine separation holds through publication too — GenAI dlopens the distinct-soname `libort_gen.so`, confirmed in device logcat.)*

### #31 — Docs set & compatibility matrix (`05_code_plans/04`)
**Partial 2026-07-14 (contract-locked pages only, no device).** Wrote the pages whose contracts are
locked: `docs/EXPORT.md` (#15/#2), `docs/RAG.md` (#25 boundary — ingestion/grounded-gen marked
not-yet), `docs/PUBLIC_API.md` (Python `__all__` + CLI; Kotlin facade pending #17/#19), generated
`docs/COMPATIBILITY_MATRIX.md` (via new `support/render.py` + `support-matrix --md`, drift-guarded by
`tests/support/test_render.py` + `tests/fixtures/gen_compat_matrix_doc.py`), and `CHANGELOG.md` +
`docs/RELEASE_CHECKLIST.md` skeletons (finalized by #32). Box stays open.
- [x] Is the compatibility matrix **rendered from** `model_support_matrix.json` (F6)? *(rendered by `support/render.py`; committed doc is a network-free representative sample, regenerable live under the export profile; re-render drift test guards it)*
- [~] Is each doc written only when its contract locks (no drift)? *(only locked pages written; ARCHITECTURE/MODEL_FORMAT/CONFIGURATION/ANDROID_SDK deferred until #23/#9-#13/#6/#30 lock or are documented)*
- [~] Is the Python public API (F5) documented in `PUBLIC_API.md`? *(Python `__all__` + CLI done; Kotlin facade section pending #17/#19)*
- Deferred: the markdown link-check wired into CI (#29 owns), and the remaining pages.

### #32 — Versioning, license & v1.0 release (`05_code_plans/05`) · checkpoint
- [ ] Is the code relicensed to Apache-2.0 with SPDX headers on **first-party source only** (vendored Microsoft/tokenizers/proto code untouched, enumerated in `THIRD_PARTY_NOTICES.md`), the `pyproject.toml` license expression set, and all rights-holders' agreement?
- [ ] Do **all version sites agree** (pyproject == `__version__` == Gradle `-Pversion` == `CITATION.cff` == tag), and are CHANGELOG non-goals listed?
- [ ] Does the **full release gate** (CI green + AAR + consumer smoke + docs + tag) pass?

### #33 — Encoder-model support (`04_code_plans/01`)
- [ ] Did the spike prove export + a train step + Android smoke + a metric for one small encoder?
- [ ] Is encoder support a `TASK_REGISTRY`/architecture-registry entry (no new `if/elif`, no KV-cache) (F3)?
- [ ] Was MARS-transfer-to-encoder-linear-layers **verified**, not assumed?

### #34 — Charging-cycle training scheduler (`04_code_plans/02`) · checkpoint
- [x] Is `LinearLRScheduler.stateDict()`/`loadFromState()` implemented (closed in the 2026-08-07 remediation pass — `CosineLRScheduler` already implements both at `:77`/`:111` and is wired via `training_state.json`), and is the restore path verified to survive WorkManager chunk boundaries and process death?
- [ ] Does a charging-constrained foreground `CoroutineWorker` checkpoint + resume cleanly across Doze?
- [ ] Does the **multi-chunk scheduled-train → resume** workflow pass (thermal/energy logged)?

### #35 — Federated codec & Python Flower simulation (`04_code_plans/03`) · checkpoint
- [x] Does `FederatedAdapterRecord` derive ordering from `TrainableTensorCodec` (no new ordering) (F8)? *(order == `HandoffMap._sorted_entries`/`tensor_specs`; `test_codec_roundtrip` pins it; byte golden `federated_record.golden.bin` freezes the #36 serialization)*
- [ ] Does an N-client Flower simulation aggregate adapter tensors and improve the metric? *(pure `federated_average` FedAvg + dropout tested; the N-client ORT-`fit` run that shows metric-improvement is the manual leg — runnable under the ORT-training profile)*
- [x] Is `adapterFormatVersion` linked to `weight_handoff_map.schemaVersion` (F1/F8)? *(`check_format` fails closed on mismatch via shared `check_compat`; `test_format_version`)*

**Code-complete 2026-07-15 (Python, no device).** `federated/` package + `cli federated simulate` + `tests/federated/` (all core-env). flwr kept OUT of the universal lock (out-of-band, like the ORT wheel — it downgrades protobuf/rich/typer + bumps mypy). Box open pending the manual N-client sim + aggregated-adapter logits smoke.

### #36 — Federated Android client & gateway (`04_code_plans/04`)
- [ ] Are `exportTrainableTensors`/`importTrainableTensors` JNI added, and is the codec byte-identical to Python (golden test)?
- [ ] Are the privacy/security gates (consent, TLS, auth, clipping/DP) addressed before any real-user run?
- [ ] Is it hard-gated on #35 passing first?

### #37 — FunctionGemma architecture gate & intents (`04_code_plans/05`) · checkpoint
- [ ] Did the architecture gate pass — Gemma-3 **inference**-graph export added as a registry entry?
- [ ] Does it **never** execute raw model output (allowlist + dry-run + validated tool calls)?
- [ ] Does the train → validated-tool-call → dry-run-intent demo show ≥2 differentiators?

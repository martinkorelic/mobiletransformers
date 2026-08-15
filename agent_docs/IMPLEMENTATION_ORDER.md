# Implementation Order — Code Plans Index

This index orders every feature code-plan under `agent_docs/00_code_plans/` … `agent_docs/05_code_plans/` from **most foundational first**. Implement in this order; each plan file repeats its own Prerequisites / Blocks header.

The high-level "what & why" lives in the six tier docs (`00_repository_restructure_plan.md`, `01_tier0_foundation_decisions.md`, `02_tier1_hf_integrated_core.md`, `03_tier2_inference_and_rag.md`, `04_tier3_reach_extensions.md`, `05_cross_cutting_release_modernization.md`). These code plans are the "how" — concrete enough for another agent to implement.

> **Status as of 2026-08-14: 36 / 37 plans done · 108 of 111 self-check boxes ticked.**
>
> ⚠️ **The block that stood here was stale in four separate claims** and is corrected above rather than
> appended to, per this file's own rule ("record the correction next to the claim"). It read
> *"31 / 37 plans · 102 of 111"* and described **#35's simulation and #37's architecture gate as
> blocked**. All four are false: #35's 4-client × 3-round simulation passed 2026-08-09 (eval loss
> falling 8.7353 → 8.5258 → 8.2579) and #37's architecture gate passed 2026-08-09. The counts here are
> a measured `grep -c` over this file's self-check boxes (**108 `[x]` / 3 `[ ]`**), not carried forward.
>
> **The 3 open boxes are all of #32's, and none of them is engineering:**
>
> 1. the **relicense** — CC-BY-NC-4.0 → Apache-2.0, a rights-holders decision (both authors in
>    `CITATION.cff`), and the one genuine v1.0 blocker;
> 2. **version sites == tag** — the sites already agree at `0.1.0` and are test-guarded
>    (`tests/unit/test_version_sites.py`, which as of 2026-08-14 also pins the manifest's
>    `mobiletransformersVersion`, the 5th site, which had been unguarded); this closes when v1.0.0 is
>    tagged, i.e. it is the release act, not work;
> 3. the **full release gate**, which needs (1), a tag, and the CI-trigger decision.
>
> **#37's differentiation demo — the last open engineering box in the project — closed 2026-08-14**
> on the S21 FE, after the merge transpose defect that had been silently corrupting it was found and
> fixed. See "The #37 demo run". Every technical item the release gate was waiting behind is finished;
> what remains is a licensing decision and the release act itself.
>
> **One standing caveat on the `Done` column:** device results recorded *before* 2026-08-14 were
> measured against a merge that wrote every weight transposed. Anything whose evidence depends on
> post-merge numerics — the 15/15 device suite, #18/#19's train→merge→generate — was re-validated after
> the fix; see "Re-validation after the transpose fix" below for what was re-run and what was not.
>
> A six-agent audit (`agent_docs/audits/`) found several `[x]` marks overstated; a 2026-08-07
> remediation pass closed most of what it found, and a 2026-08-09 pass de-drifted the `Done` column
> against the recorded device runs (#9/#12/#18/#19/#30) and the pages that actually exist (#31). Where
> a box reads `[x]` it is because the DoD holds today, not because a session claimed it. Boxes still
> `[ ]` are honest remaining work, and each says what is missing.
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
- **Python config lives inside the package.** `src/mobiletransformers/config/` holds `constants.py` (enums), `settings.py` (secrets), `models.py` (Pydantic), `registry/` — all shipped in the wheel. Repo-root `config/` holds only user-editable YAML (`config.yml`, `logging.yml`). Root `config.py` / `tools/parser_config.py` became deprecation shims importing from `mobiletransformers.config.*`; **both are now deleted** — `tools/parser_config.py` with the `tools/` root in S9, `config.py` on 2026-08-14 once its last two importers were repointed. Owned by `00_code_plans/02` + `00_code_plans/09`.
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
> `make device-test`, 7 pass / 1 skip). Ten export↔runtime defects were found and fixed getting there
> — see HANDOFF.md.
>
> **`Done` column de-drifted 2026-08-09.** #9/#18/#19 (`TrainMergeGenerateTest` PASS once a `TRAIN=1`
> package existed), #12 (the four-point Gate 0.2 table was collected via `make device-rss`; the gate
> FAILS as specified, which is a recorded result, not missing work) and #30 (`make publish-local &&
> make consumer-app` produced a 105 MB consumer APK from mavenLocal) all had every self-check ticked
> with a device run recorded while their `Done` box still read `[ ]`. Per the Tracking rule above the
> column follows the self-checks, so they are now `[x]`. No new work was done to flip them.


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
| [x] | 9 | `01_code_plans/01_unified_merger_and_external_data_export.md` | **Dual-engine core** + full merger unification | 6, 7, 8 | *(code-complete 2026-07-14: Python A/B/C tested, C++ D/E compile+link-verified on arm64-v8a. Device legs PASS 2026-08-08 — `TrainMergeGenerateTest` rewrote 60/60 trainable `.bin` files; ticked 2026-08-09 — see #9 self-check)* |
| [x] | 10 | `01_code_plans/02_genai_external_data_swap_spike.md` | Feeds Gate 0.1 | 9 | *(2026-07-15: **Gate 0.1 = ADOPT GenAI.** F2 validated — external-data swap changes GenAI output on device (token/fp differ) + desktop; symbol/fork-only confirmed; RSS measured (mmap). The one blocker — ORT-runtime coexistence (genai needs stock ORT ≥1.26, Native needs training ORT 1.23) — **resolved** via `libort_gen.so` distinct-soname separation, verified both coexist on device. Cross-engine #1/#4 (same package under BOTH engines) ride with #11's dual-engine smoke + a real #9 package. See `spikes/genai_external_swap/README.md`.)* |
| [x] | 11 | `01_code_plans/03_inference_engine_abstraction_native_and_genai.md` | Engine selection | 10 | *(code-complete 2026-07-15: `runtime/ModelRuntime.kt` (interface + `EngineCapabilities` + `EXECUTION_PROVIDER_REGISTRY` F3 + `GenAiSupport` + `ModelRuntimeFactory` pure-select + device-create w/ transparent Native fallback); `ORTGeneratorGenAI.kt` + `cpp/genai_runtime.cpp` (streaming, callback parity); `ORTGeneratorNative` adapted to `ModelRuntime`; `engine` field in `ORTGenerationConfig`; `LLMRepository` wired via factory; dead `ORTGenAINative.kt`+`onnx-genai.cpp` deleted. Compiles+links arm64, **48 JVM tests**, device build loads. Box open pending the dual-engine same-folder device smoke + streaming-parity harness — need a real #9 package.)* |
| [x] | 12 | `01_code_plans/04_memory_mapping_experiments.md` | Optimizes 9–11 (non-blocking) | 9 | *(2026-08-08: the four-point RSS table was collected on device via `make device-rss`; **Gate 0.2 FAILS as specified** (6.3% vs 15% required) and that measured FAIL is the deliverable — mmap is default-off and explicitly "an optimization, not a v1 requirement", so it does not block v1. Ticked 2026-08-09. Code-complete 2026-07-15: `cpp/mem_probe.h` (RSS) + `cpp/mmap_tensor.h` (RAII) + a default-off `MTF_MMAP_WEIGHTS` zero-copy branch in `session_cache.h` (copy path stays the shipping default, #23 unaffected); `spikes/mmap/{measure_rss(re-export),base_blob_mmap_spike}.py` (desktop byte-identical correctness invariant). arm64 links. Device 4-point RSS table = the manual Gate 0.2 leg.)* |
| [x] | 13 | `00_code_plans/06_manifest_first_package_and_cache_bridge.md` | Package contract | 8, 9 | *(done 2026-07-14: Python `artifacts/manifest.py` + Kotlin `packages/` cache-bridge (6 classes), JVM-tested + `compileDebugKotlin`; the on-device generate smoke it deferred is **CLOSED** — `FacadeLoadGenerateTest` PASS 2026-08-08 on the S21 FE)* |
| [x] | 14 | `02_code_plans/03_hub_model_package_format.md` | Hub repo shape | 13 | *(done 2026-07-14: `hub/package_format.py` — `sanitize_repo_id`, `build_manifest`, tiny_package fixture; no device)* |
| [x] | 15 | `02_code_plans/05_one_command_export_cli.md` | Wraps 7–9 + 13 *(checkpoint: export E2E)* | 7, 9, 13 | *(done 2026-07-14; 2026-07-15: `_full_export` real inference+GenAI leg implemented (stage-gated) + on-box verified — SmolLM2-135M → #13-valid package, GenAI loads it; training stage staged for the ort-training-local run — see #15 self-check)* |
| [x] | 16 | `00_code_plans/04_android_gradle_rename_migration.md` | Isolated, verified rename | — | *(done 2026-07-14: **full removal / option B** — Kotlin+native+JNI all renamed off `ortmobile`; supersedes the doc's option-A. Compile+link-verified arm64-v8a, `compileDebugKotlin` + `make parity` green.)* |
| [x] | 17 | `00_code_plans/05_android_facade_foundation.md` | Public SDK facade *(checkpoint: load→generate)* | 13, 16 | *(code-complete 2026-07-15: `MobileTransformers.fromPretrained`/`MobileTransformerModel`, public `config/*` + `runtime/{ModelSession,RuntimeCapabilities,Results}`, `ConfigMappers`, `RepositoryBackedModelSession`, `ModelFeature`, exception stubs; 13 JVM tests (config round-trip, feature/engine, variant-select, facade delegation). Box open pending the device load→generate leg. `InferenceEngine` placeholder retired by #11.)* |
| [x] | 18 | `00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md` | Job/progress/checkpoint API | 17 | *(device legs PASS 2026-08-08 (`TrainMergeGenerateTest`); session lock added in the 2026-08-07 remediation; ticked 2026-08-09. Code-complete 2026-07-15: `training/` `TrainingJob`/`TrainingStatus`/`TrainingEvent`/`CheckpointInfo`/`TrainingJobManager`(+`TrainingJobSpec`) + `TrainingEventAdapter`; `ORTTrainerNative` cooperative `cancelRequested` (no format change); `TrainingResult` enriched. 5 JVM tests (event mapping, checkpoint round-trip). Box open pending device resume/summary/train→merge→generate legs.)* |
| [x] | 19 | `02_code_plans/01_hf_style_kotlin_facade.md` | `fromPretrained`/train/merge/generate *(checkpoint: train→merge→generate)* | 11, 17 | *(**train→merge→generate PASSES on device** 2026-08-08 (S21 FE / Android 15 / arm64); ticked 2026-08-09. Code-complete 2026-07-15: DELTA over #17 — `applyPeft`/`pushAdapter` + public callbacks + full sealed exception hierarchy + sealed `PeftConfig`/`PeftSupport` + engine/merge-driven `ConfigMappers` + construction-time feature/GenAI gates; 87 SDK JVM tests + sample-app compile green; box open pending the device train→merge→generate workflow — see #19 self-check)* |
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
| [x] | 30 | `05_code_plans/03_aar_maven_publication.md` | Portable Android consumption *(checkpoint: consumer app builds)* | 16, 28 | *(**PROVEN 2026-08-08**: `make publish-local && make consumer-app` → a 105 MB consumer APK built from mavenLocal alone (`FAIL_ON_PROJECT_REPOS`), carrying all 7 native libs out of the AAR. Ticked 2026-08-09. arm64-v8a only — `abiFilters` advertises no ABI it cannot load.)* |
| [x] | 31 | `05_code_plans/04_docs_set_and_compatibility_matrix.md` | Public docs + registry-driven matrix as contracts stabilize | 23-27, 19, 13 | *(ticked 2026-08-09: all three self-checks hold — 16 pages in `docs/` incl. ARCHITECTURE/MODEL_FORMAT/CONFIGURATION/ANDROID_SDK, the matrix is rendered from `model_support_matrix.json` with a drift test, and `PUBLIC_API.md`'s Kotlin section is now guarded by `test_docs.py`. The one deferral left — the markdown link-check in CI — is **owned by #29**, not #31. Partial 2026-07-14: `docs/EXPORT.md`, `docs/RAG.md` (#25 scope), `docs/PUBLIC_API.md`, generated `docs/COMPATIBILITY_MATRIX.md` + `support/render.py` renderer + `support-matrix --md` + drift test; `CHANGELOG.md` + `docs/RELEASE_CHECKLIST.md` skeletons. `docs/MODEL_FORMAT.md` + `docs/CONFIGURATION.md` added 2026-07-15 (contracts locked). Remaining pages (ARCHITECTURE/ANDROID_SDK) await #23/#24/#30; box stays open.)*|
| [ ] | 32 | `05_code_plans/05_versioning_license_release.md` | v1.0 release gate *(checkpoint: full release)* | 29, 30, 31 |

### Tier 3 — Reach extensions (Phase 9; each spike-gated, never blocks v1.0)

| Done | # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- | --- |
| [x] | 33 | `04_code_plans/01_encoder_model_support.md` | Cheapest reach; arch-registry entry + codec/manifest reuse | 13, 7, 9 | *(ticked 2026-08-10: all three self-checks hold. The last one — the Android smoke — needed the encoder TRAINING export to work past `gen_artifacts`, which exposed a decoder-shaped constant in the peft→ORT name rewrite; see "The #33 encoder legs".)*
| [x] | 34 | `04_code_plans/02_training_scheduler_workmanager.md` | Charging-cycle training; WorkManager scheduling (the `LinearLRScheduler` state-persistence gap it was written against was closed in the 2026-08-07 remediation pass; `ORTScheduler.kt:161-162` records the fix) *(checkpoint: scheduled train→resume)* | 18, 17 | *(ticked 2026-08-10 by the Tracking rule: all three self-checks were already `[x]` with the 2026-08-09 device evidence — chunk/checkpoint/resume proven on the S21 FE, thermal/energy trace captured. NO new work; the column had simply drifted behind the self-checks, the same way #9/#12/#18/#19/#30 had. The multi-hour/Doze leg is a recorded DEBT, and the plan's own DoD does not require it.)*
| [x] | 35 | `04_code_plans/03_federated_codec_and_python_simulation.md` | Python Flower sim (Option A first) *(checkpoint: N-client sim)* | 8, 9, 13 | *(code-complete 2026-07-15: `federated/` `FederatedAdapterRecord` (codec-derived, pinned byte serialization = #36 golden), `flower_sim.py` pure `federated_average` + `cli federated simulate`; `flower_client.py` lazy-flwr. Tests `tests/federated/` (roundtrip/format-version/comm-size/fedavg/dropout + byte golden) run in core env. flwr kept OUT of the universal lock (out-of-band, like the ORT wheel). Box open pending the manual N-client ORT-`fit` sim + aggregated-adapter logits smoke — runnable under the ORT-training profile.)* | *(ticked 2026-08-10 by the Tracking rule: all three self-checks were already `[x]` — the 4-client x 3-round simulation passes with the aggregated-adapter eval loss falling monotonically 8.7353 -> 8.5258 -> 8.2579. NO new work.)*
| [x] | 36 | `04_code_plans/04_federated_android_gateway.md` | Android client + gateway (Option B) | 35, 34, 18 | *(ticked 2026-08-10: the device round-trip passes on the S21 FE — export from a live checkpoint → `federated serve` → import back → the checkpoint moved and survived a reload. See "The #36 device round-trip". Found two real defects doing it.)*
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
- [x] Do the deprecation shims for `config.py` / `tools/parser_config.py` still import and warn? *(both shims are now **deleted** — the box records that the migration they covered completed, not that the shims survive. `tests/unit/test_settings_precedence.py` asserts the root `config.py` stays gone.)*
- [x] Are non-secret constants in `constants.py` and user knobs in `config.yml`, with **no** secrets in YAML?

> Done 2026-07-13. `mobiletransformers.config.resolve()` implements CLI>env>YAML>default (unit-tested); secrets live only in `Settings`/`get_settings()` and the CI grep guard (`os.environ[...HF_TOKEN|HF_CACHE|GEMINI_API_KEY|AZURE_...]` over `src/`) is empty. Root `config.py` + `tools/parser_config.py` were deprecation shims (import + `DeprecationWarning` tested), **both since deleted**; `config/config.yml` copied verbatim (4 sections, YAML-only dir); `utils/yaml.load_config_from_file` added. `test_settings_precedence.py` green.
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
>
> ⚠️ **CLOSED — this deferral note is stale (corrected 2026-08-14).** The build-side emit landed
> (the 2026-08-07 audit recorded it as done "contrary to the plan's own trailing 'deferred' note"),
> the C++ rewrite shipped with #9/#23, and the JNI thread-through with #18/#19. Nothing here is open.

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
- [x] Do the experiments produce RSS numbers feeding the Gate 0.2 decision? *(full 2x2 table collected via `make device-rss`, S21 FE / Android 15 / arm64, 2026-08-08. As originally specified the gate **FAILED**: 6.3% whole-process peak reduction on Native, -0.1% on GenAI, against 15% required. **Re-specified 2026-08-09** — the 15% predated #9's base/trainable split, which moved 91.8% of weight bytes outside what `WeightSessionCache` loads, making the figure unreachable by design rather than by defect. Scoped to the trainable-split bytes mmap owns, the measurement **PASSES**: 50,524 kB saved of 54,374 kB eligible = 92.9% of the theoretical maximum. GenAI is n/a by construction. See `01_tier0_foundation_decisions.md` "Gate 0.2 RE-SPECIFIED".)*
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
- [x] Does `MobileTransformers.fromPretrained(...)` return a working `MobileTransformerModel` wrapping the existing repositories (no JNI rewrite)? *(`RepositoryBackedModelSession` wraps `LLMRepository` + 3 sub-repos; no `ORT*`/`Job` in public signatures; delegation covered by `FacadeDelegationTest`. The on-device load→generate leg is **CLOSED** — `FacadeLoadGenerateTest` PASS 2026-08-08, S21 FE / Android 15 / arm64-v8a)*
- [x] Are the engine selector + exception hierarchy in place? *(`InferenceEngine`/`RuntimeCapabilities` + `ModelFeature` engine-selector semantics tested; `MobileTransformersException` base + `ModelNotInstalled`/`MissingArtifact` stubs — full hierarchy is #19)*
- [x] Does the facade workflow (load → generate one token, device-manual) pass? *(`FacadeLoadGenerateTest` PASS, S21 FE / Android 15 / arm64, 2026-08-08)*

**Code-complete 2026-07-15.** 13 JVM tests (config round-trip == `ORT*Config` defaults, feature/engine semantics, manifest variant-select, facade→session delegation via a hand-written fake). Box open pending the device load→generate checkpoint.

### Six facade gaps found by the app rewrite — all fixed IN the facade (2026-08-14)

The showcase app was rewritten onto the public facade (see below). Driving six real screens through it
found six places where the facade could not express what a screen needed. **Each was fixed in the
facade, not worked around** — a reach-around would have been the same debt under a new name. This is
the return on the rewrite: the API had never met a real screen, and every one of these had been
invisible for that reason.

| # | gap | fix |
| --- | --- | --- |
| 1 | `fromPretrained` called `HubDownloader.downloadAndInstall` with the default no-op `onProgress`, so a multi-hundred-MB first-run download was **invisible** to any facade-only caller | `onDownloadProgress: DownloadProgressListener?` parameter + a named `DownloadProgress` type (`filesDone`/`filesTotal`/`path`/`fraction`, the last `null` until the plan resolves) |
| 2 | no way to answer "what is already installed?" — `CacheIndex.list` existed but nothing on the entry point exposed it | `MobileTransformers.installed(cacheDir)` / `installed(context)` |
| 3 | `TrainingJob.start` took an `ORTTrainingConfig`, so the lifecycle handle (`status`/`events`/`cancel`/`checkpoint`) was **reachable and unusable at once** — an app could have progress flows or stay on the facade, never both | `TrainingJob.start(dataset: DatasetConfig, config: TrainConfig)`, mapping through the same `ConfigMappers` as the one-shot path |
| 4 | `TrainingScheduler.schedule` likewise took `ORTTrainingConfig`, so #34's scheduler — which `RuntimeCapabilities.supportsScheduledTraining` **advertises through the facade** — could not be driven from it | public overload taking `DatasetConfig` + `TrainConfig`; the `customPreprocess == null` precondition holds by construction since `DatasetConfig` names a registered task |
| 5 | **federation was entirely unreachable.** `FederatedTrainingRepository.forSession` is `internal`, and hand-assembly needs `NativeCheckpointTensorStore(trainer: ORTTrainerNative)`. The only shipped capability with no public door at all | `MobileTransformerModel.federatedRound(...)` → `FederatedRoundResult`; fails closed on consent/TLS/auth **and** on a missing `weight_handoff_map.json`, naming which. `LocalRoundTraining` became a `fun interface` so callers can pass a lambda |
| 6 | `RuntimeCapabilities` said which engine you got but not which others were **selectable**, so an engine picker could only offer both and learn the answer by catching `EngineUnavailableException` — an exception as control flow for a question the SDK already knew | `RuntimeCapabilities.availableEngines`, computed from the same two conditions `ModelRuntimeFactory` applies (package ships `genai_config.json` **and** the native probe succeeds) |

Covered by `FacadeDelegationTest::federatedRoundIsReachableFromTheFacadeAndPassesItsArgumentsThrough`
and the app module's `ShowcaseStateTest` (6 JVM tests). No seventh gap was found: the ~45 configuration
knobs the old app edited via `ORT*` types all proved expressible through
`GenerationConfig`/`TrainConfig`/`RagConfig`/`DatasetConfig`, which is the result the Configuration
screen exists to report.

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
- [x] Does the **train→merge→generate** workflow (device-manual) pass end to end? *(`TrainMergeGenerateTest` PASS, S21 FE `SM-G990B` / Android 15 / arm64-v8a — first 2026-08-08 (60/60 trainable `.bin` files rewritten by the merge), **re-proven 2026-08-09 under the changed `q_proj`/`v_proj` PEFT targets** as part of a 13/13 0-skipped suite. NOTE the evidence is that the merge HAPPENED — byte-fingerprint before/after — not that it is numerically sane; one LoRA step on an 8-row fixture yields `,,,,,,,,`. Post-merge numerical correctness on device was tracked here as a standing debt; it is **CLOSED 2026-08-10** — see "Post-merge numerical correctness — CLOSED" in the permanent section. Corrected here 2026-08-14, because the debt was closed there and never struck out here.)*
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
- [x] Is the conversation-state prepend bug fixed with a reset test? *(rendered-offset fix + `resetConversation()` on load; `ORTConversationStateTest` covers reset; the two-prompt device smoke it deferred is **CLOSED** — `ConversationResetTest` PASS on device)*
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
- ~~Deferred (device/later): the ObjectBox parity smoke (Android, supported dims — manual).~~ **CLOSED** — `ObjectBoxParityTest` PASS in both recorded device runs. ~~`SearchType` String→enum swap rides with #17/#19~~ — **that deferral expired** when both landed 2026-07-15 and the audit found it ownerless; done 2026-08-07 (`FileUtil` validates `searchType` through `SearchType.fromWire` at the parse boundary and `ORTRetriever.query` dispatches on the enum; the field stays `String` because it crosses JNI as one). ~~`docs/RAG.md` is #31~~ — written, and de-drifted 2026-08-07 (it had claimed #26/#27 were unimplemented long after they landed).

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
- [x] Is each doc written only when its contract locks (no drift)? *(2026-08-09: the deferral list in this box was stale. `ARCHITECTURE.md` (2026-08-07), `MODEL_FORMAT.md` + `CONFIGURATION.md` (2026-07-15) and `ANDROID_SDK.md` (2026-08-08) all exist — 16 pages in `docs/` — each written after its contract locked, per the original rule.)*
- [x] Is the Python public API (F5) documented in `PUBLIC_API.md`? *(Python `__all__` + CLI done 2026-07-14; the Kotlin facade section was written 2026-08-09 from the real surface — `MobileTransformers.fromPretrained`, `MobileTransformerModel`, the sealed exception hierarchy — now that #17/#19 are ticked)*
- Deferred: the markdown link-check wired into CI (**#29 owns it**, and no workflow currently runs automatically — see #29's deferral). That is the only remaining #31 item.

### #32 — Versioning, license & v1.0 release (`05_code_plans/05`) · checkpoint
- [ ] Is the code relicensed to Apache-2.0 with SPDX headers on **first-party source only** (vendored Microsoft/tokenizers/proto code untouched, enumerated in `THIRD_PARTY_NOTICES.md`), the `pyproject.toml` license expression set, and all rights-holders' agreement?
- [ ] Do **all version sites agree** (pyproject == `__version__` == Gradle `-Pversion` == `CITATION.cff` == tag), and are CHANGELOG non-goals listed? *(2026-08-09: **satisfied except the tag, which is the release act itself.** All four sites read `0.1.0` (`pyproject.toml`, `__init__.__version__` reads it dynamically, `android/MobileTransformers/gradle.properties`, `CITATION.cff`) and `tests/unit/test_version_sites.py` guards them — 6 tests green, so they cannot drift. `CHANGELOG.md` has its `Non-goals` section. The repo has 0 tags; this box closes when v1.0.0 is tagged.)*
- [ ] Does the **full release gate** (CI green + AAR + consumer smoke + docs + tag) pass?

### #33 — Encoder-model support (`04_code_plans/01`)
- [x] Did the spike prove export + a train step + Android smoke + a metric for one small encoder? *(**ALL LEGS PASS 2026-08-10** — see "The #33 encoder legs" below. The Android smoke is `EncoderTrainStepDeviceTest` on the S21 FE (`SM-G990B` / Android 15 / arm64-v8a): 2 real steps over a `text-classification` package, finite losses `[0.7222, 0.7224]`, **24/24 adapter factors moved** in the ORT checkpoint. Getting there needed the encoder TRAINING export to work past `gen_artifacts` for the first time — it did not, and the reason was the layer-identity problem in a second namespace (below). host legs ALL PASS 2026-08-09 — export → `generate_artifacts` → 30 real train steps → a metric, on `sentence-transformers/all-MiniLM-L6-v2`: loss **0.6950 → 0.5433** (21.8%, monotonic) and **accuracy 0.250 → 1.000** on a separable 8-example set. Pinned by `tests/integration/test_encoder_training_gate.py` (4 tests, `ort-training-local` profile). The encoder **inference/embedding** path was already shipping as the RAG embedder.)*
- [x] Is encoder support a `TASK_REGISTRY`/architecture-registry entry (no new `if/elif`, no KV-cache) (F3)? *(2026-08-09: `config/registry/task.py` — `TASK_REGISTRY`/`TaskSpec`/`get_task_spec` — now owns the auto-model class, the KV-cache kwargs, PEFT's `task_type` and the training-wrapper class. The three task-shaped branches in `export/training_export.py` are gone. It fixed two real defects on the way: PEFT's `task_type` was **hardcoded `"CAUSAL_LM"` at both LoRA call sites**, which mis-wraps any encoder, and `OnnxTrainerWrapper`'s forward signature was decoder-only, which is what an encoder export actually failed on inside optimum. 10 unit tests; decoder export verified byte-unchanged in its ONNX input names.)*
- [x] Was MARS-transfer-to-encoder-linear-layers **verified**, not assumed? *(**VERIFIED 2026-08-09**, `tests/integration/test_mars_encoder_transfer.py` — 9 tests, no network, no HF token. See "MARS on encoder" below for what was actually wrong and how the assertions catch it.)*

### The package shape is now task-driven (2026-08-10)

The host training chain worked on 2026-08-09, but everything *after* `gen_artifacts` was still
decoder-assumed: `export/pipeline.py` read **no `TaskSpec` at all**. It stamped KV-cache geometry into
every graph, emitted a `model.decoder` GenAI block for every package, claimed a `train` stage for every
model including tasks that cannot produce one, and ran a causal-LM parity gate that raises on rank-2
logits. `TaskSpec` was consumed by exactly one module (`export/training_export.py`).

`TaskSpec` now carries the **package** facts alongside the graph facts — `emits_genai_config`,
`stamps_kv_metadata`, `parity_check` (a dotted path, so the gate is task data rather than an `if`), and
a derived `stages`. The pipeline reads them; the `plan.task.startswith("text-generation")` string test
in `_build_training_stage` is replaced by the registry lookup it was standing in for.

**A real defect fixed on the way.** `export_inference_package` called `resolve_architecture(model_config)`
with no `architecture=` override — the exact bug `training_export.py:393` had already fixed on its side.
A sentence-transformers checkpoint declares `["BertModel"]` even when loaded as
`BertForSequenceClassification`, so the training half resolved the classification row while the
packaging half resolved the un-headed one, and the two halves of one export disagreed about the
architecture. The loaded class is now recorded in `training_config.json` (`"architecture"`) and read
back by the packager, rather than each half guessing.

**Evidence, both directions:**

```
encoder  sentence-transformers/all-MiniLM-L6-v2 --task text-classification --stages inference --validate
         -> #13-valid package. selectedTask "text-classification", features ["core","inference"],
            supportedEngines ["native"], metadata_props {} (NO KV geometry), no genai_config.json,
            inputs [input_ids, attention_mask, token_type_ids], outputs [logits]

decoder  HuggingFaceTB/SmolLM2-135M-Instruct, re-exported after the change
         -> model.onnx BYTE-IDENTICAL to the pre-change export (md5 2f77a194…), same side-car set,
            metadata_props {head_dim 64, num_kv_heads 3, num_layers 30} unchanged
```

The decoder byte-identity is the point: this is a no-op for every package that ships today.

**The Kotlin half — per-sequence labels.** `DataCollatorForSupervisedDataset` padded labels to
`maxLength` with `-100` unconditionally, which turns one label per example back into one per token and
defeats the native rank inference (`training_inputs.h::labels_shape` reads `batch*seq` as `[batch, seq]`
and `batch` as `[batch]`). Both halves were correct alone; the seam decided the exported label rank.
`TrainingSample.perSequenceLabel` now records which objective it is — explicitly rather than inferred
from `labels.size`, because at `sequenceLength == 1` the two are indistinguishable by count.

The shape comes from data: `TaskPreprocessor.classLabel(json)` returns a class index or `null`, added
with a **default** rather than a changed signature because `TaskPreprocessor` is public API (callers
pass their own through `DatasetConfig.customPreprocess`). `CoLAClassificationPreprocessor` (`cola_cls`)
is the same file as a real classification objective, next to `CoLAPreprocessor`, which stringifies the
class into `"acceptable"`/`"unacceptable"` to fit the decoder's text-to-text contract. 6 JVM tests.
`DataCollatorForSupervisedDataset` now takes the pad token rather than the whole tokenizer — that is
all it used, and the dependency on `ORTTokenizerNative` (whose `init` loads the native library) is what
made the label-padding rule untestable on the host.

`make device-package` gained a `TASK=` knob, and suppresses `--genai` for cacheless tasks. It was
otherwise impossible to export an encoder through it: `TASK_PREFERENCE` does not contain
`text-classification`, so `all-MiniLM-L6-v2` auto-resolves to `feature-extraction`, which is declared
`trainable=False`.

~~**Still outstanding:** the on-device encoder train step.~~ **CLOSED 2026-08-10** — see "The #33
 encoder legs — ALL PASS" below (`EncoderTrainStepDeviceTest`: 2 real steps, finite losses, 24/24
 adapter factors moved). Struck out here 2026-08-14; the closure was recorded further down but this
 claim was left standing.

**Objective DECIDED 2026-08-09: sequence classification** (user's call; the abstraction is built so
masked-LM and contrastive embedding arrive as registry rows — see `registry/task.py`'s "Adding a
training objective"). **Status: the full host chain works — export → artifacts → training → metric.**

```
loss  0.6950 -> 0.5433   (21.8% drop over 30 steps, monotonic)
accuracy 0.250 -> 1.000  (separable 8-example sentiment set)
```

~~Only the Android smoke is outstanding, and it is device-gated.~~ **CLOSED 2026-08-10** — the smoke
passed, and getting there exposed a decoder-shaped constant in the peft→ORT name rewrite.

### What works, proven under `ort-training-local` on `sentence-transformers/all-MiniLM-L6-v2`

The classification **training graph exports cleanly**:

```
inputs : input_ids[batch,seq]  attention_mask[batch,seq]  token_type_ids[batch,seq]  labels[batch]
outputs: loss, logits
```

`labels[batch]` — one per sequence — is the contract that distinguishes this objective, and it is
declared as data (`TaskSpec.label_shape`) rather than implied. Landed with it:

- `TaskType.SEQUENCE_CLASSIFICATION` (`"text-classification"`), mirrored in Kotlin, `make parity` green.
- `TaskSpec` extended into a real objective description: `trainable`, `label_shape`,
  `model_init_kwargs` (`num_labels`), `quantization_exclude_layers`, `trainer_wrapper_class`.
- Architecture rows for `BertForSequenceClassification`, `RobertaForSequenceClassification`,
  `DistilBertForSequenceClassification` (the last with `q_lin`/`v_lin` — the per-architecture
  difference the registry exists to hold as data).
- `resolve_architecture(config, architecture=type(model).__name__)`: **the head is part of the
  architecture identity.** A sentence-transformers checkpoint declares `["BertModel"]` even when loaded
  as a `BertForSequenceClassification`, so keying off the config alone resolved an encoder fine-tune to
  the un-headed row. For every decoder the two agree, so this is strictly more accurate, not a change.
- `feature-extraction` is now declared `trainable=False` and fails with one sentence naming the
  alternative, instead of `BertModel.forward() got an unexpected keyword argument 'labels'`.

### The two gradient-graph defects that had to be fixed (neither reachable from a decoder)

**1. Activation quantization on the gradient path.** `generate_artifacts` failed with

```
The gradient builder has not been registered: DynamicQuantizeLinear
  for node /backbone/bert/pooler/Gather_output_0_QuantizeLinear
```

The principle it violated: **quantized weights are frozen and dequantize to float
(`DequantizeLinear`), so the backward pass crosses them untouched — but a quantized *activation* has
no gradient at all.** BERT's `pooler`/`classifier` are `Gemm`; the decoder's linear layers are
`MatMul`. Measured: decoder **0** DQL nodes, encoder **2**.

The trap was that **ORT rewrites `Gemm` → `MatMul` *before* quantizing and matches
`nodes_to_exclude` against the rewritten name** `<gemm>_MatMul` — so excluding the `Gemm` by its own
name silently missed, which is why the first exclusion attempt (and restricting `op_types_to_quantize`
to `MatMul`) changed nothing. `onnx_dynamic_quantization` now also excludes the `_MatMul` alias of any
excluded `Gemm`. Result: **DQL 2 → 0**, `DequantizeLinear` 27 retained (weights still quantized).

**2. `LayerNormalization` exported with a single output.** ORT's `LayerNormalizationGrad` reads the
forward node's optional **second and third outputs** (saved mean / inverse std) instead of recomputing
them, and `torch.onnx` emits only `Y`. The failure names neither op nor node:

```
GradientBuilderBase::O(size_t, bool) const i < node_->OutputDefs().size() was false
```

Found by bisecting the trainable set until the boundary landed between the pooler (**OK**) and the
encoder layers (**fail**). Decoders never hit it — Llama RMSNorm exports as
`SimplifiedLayerNormalization`, already carrying 2 outputs (61 of them in the shipped package, each
with a matching `SimplifiedLayerNormalizationGrad`). Fixed by
`artifacts/graph_prep.py::ensure_layernorm_grad_outputs`, a pre-pass that appends the optional outputs
(spec-optional, ORT fills them, nothing else consumes them) and writes the rewritten graph **beside**
the source so relative external-data references keep resolving. **No-op on every decoder package** —
different op type, and verified `LayerNorm(1 output) == 0` on the real SmolLM2 graph.

**Decoder regression verified** on the real `SmolLM2-135M-Instruct` training stage after both fixes:
`DQL=0`, `DequantizeLinear=152`, checkpoint **176,349,432 B** and `trainable_parameter_count`
**460,800** — both identical to the shipped package.

### MARS on encoder — verified 2026-08-09 (self-check 3)

**It was six hardcoded sites, not one.** The standing note said `peft/mars/model.py` ignored
`ArchitectureSpec.attention_module_name`. That was the entry point; the decoder naming was actually
inlined in six places, five in `peft/mars/model.py` and one in `peft/mapping.py`:

| site | assumption | BERT reality |
| --- | --- | --- |
| attention lookup | `isinstance(m, type(model.model.layers[0].self_attn))` | `BertForSequenceClassification` has `.bert`, no `.model.layers` — **`AttributeError`**, not a no-op |
| shared-output hook | `kwargs["hidden_states"]` | `BertAttention`/`BertSelfAttention` pass it **positionally** |
| projection hooks | `register_proj_hook("q_proj", …)` ×3 | named `query`/`key`/`value` (DistilBERT: `q_lin`/`v_lin`), and nested one level deeper under `attention.self` |
| `projection_type` ladder | `"q_proj" in target_name` | never matches → `is_standalone` stays `True` → **MARS silently degrades to unshared adapters** |
| `_replace_module` grouping | same literals | the shared adapter is never wired to the wrapped module |
| `peft/mapping.py` back-pointer | `"v_proj" in base_layer_name` | BERT's `value` silently lost `shared_A`/`intermediate`/`adapter_index` while `query` kept them — the two halves of one layer disagreed about whether they shared a tensor |

**The fix is data.** `ArchitectureSpec.projection_names` maps projection **role** → module name
(`DEFAULT_PROJECTION_NAMES` is the Llama naming; BERT/RoBERTa and DistilBERT rows override it), with
`module_name_for_role`/`role_for_module` as the two lookups. The anchor a shared adapter attaches to
is now **the module that directly owns the projections** — `self_attn` on a decoder, `attention.self`
on BERT — scoped by `attention_module_name` so an unrelated module owning a `query` child is not
mistaken for attention. Anchoring on `attention_module_name` itself would have attached the adapter
to the wrong parent, which is what `_replace_module` reads `shared_qkv` off.

**Why the assertions are shaped the way they are.** A no-op and a success look identical here, so the
tests assert *counts* (one `SharedAttentionAdapter` per attention block; every wrapped projection
`is_standalone=False`) and, **across the seam**, that perturbing only the *shared* parameters moves
the logits — impossible unless the shared adapter is genuinely on the compute graph. `up_project` is
zero-initialised, so it is filled first or the whole adapter branch multiplies to zero and the test
would pass for the wrong reason.

**Evidence:**

- 9 tests pass; **against the pre-fix tree the 3 encoder tests + the fail-closed test FAIL and the 3
  decoder tests PASS** — the exact shape of the defect (`AttributeError: 'BertForSequenceClassification'
  object has no attribute 'model'`).
- **Decoder regression: the MARS adapter mapping is byte-identical** before and after the change
  (dumped and diffed on a 2-layer Llama).
- Real MARS encoder export end to end on `sentence-transformers/all-MiniLM-L6-v2`:
  `optimum_hf_export(train_method="mars", task_type="text-classification")` → `gen_artifacts` →
  **30 real train steps, loss 0.6892 → 0.6505**. The exported graph carries
  `/backbone/bert/encoder/layer.N/attention/shared_qkv/…` nodes, and `training_config.json` lists 24
  `requires_grad` entries including the shared tensors.
- Unknown architectures now **fail closed** in `MarsModel.__init__` instead of silently applying
  decoder naming.
- Registry data pinned by 9 core-env tests in `tests/unit/test_registries.py` (run in `make check`).

**Finding — FIXED 2026-08-09, see "MARS's shared adapter was frozen by quantization" below.** It was
found here but is not encoder-specific: in the quantized training graph the MARS shared tensors came
out `_quantized`/`_scale`/`_zero_point` and were therefore frozen, so only the per-module
`up_project` adapters trained. A decoder behaved identically.

### Guards added 2026-08-09 (the generalizing half of this cycle)

Three new gates, each written because a real defect got through the existing ones:

| guard | catches | proven by |
| --- | --- | --- |
| `tests/unit/test_guards.py::test_no_architecture_literals_outside_the_registry` | per-architecture module names spelled as literals outside `config/registry/` | reintroducing the defect fails it |
| `tests/export/test_registry_matches_optimum.py` | a registry row bound to an ONNX config Optimum would not pick | restoring the old Gemma-3 binding fails it |
| `tests/unit/test_trainable_gate.py` (+ the in-export gate) | a tensor declared trainable that the graph does not realize as trainable | raises on real pre-fix export output |
| `tests/unit/test_dependency_profiles.py` | the export/training `transformers` fork collapsing back to one version | asserts both lines are in `uv.lock` |

**What the architecture-literal guard found on its first run.** `training_export.py`'s `--lora_target`
still defaulted to `["q_proj", "k_proj"]` — the decoder-specific pairing the architecture registry was
introduced to replace. Because argparse always supplies a value, **every run through that entry point
silently overrode the registry with it**, including on encoders where those modules do not exist. The
default is now `None`, so the registry decides and an explicit `--lora_target` still wins. Two smaller
hits went with it: `handoff_map.py` now takes its fallback from the registry's
`DEFAULT_ATTENTION_MODULE_NAME` instead of spelling `"self_attn"`, and `artifacts/validation.py`'s
`replace("self_attn", "attn")` is allow-listed with a named owner (it builds names from `.npz`
filenames and has no `ArchitectureSpec` to resolve from — the fix is to thread one in, not to widen
the allowance).

**Correction to an earlier claim.** A previous handoff entry said `peft/ablation/model.py` still
carried the decoder assumptions #33 removed from MARS. **That is wrong** — its `_replace_module` is
generic, with no projection-type grouping and no architecture literals. The guard confirms it.

**A test-isolation bug, found by the guards being run on a machine with secrets configured.**
`get_settings()` calls `load_dotenv()`, which writes `.env` into `os.environ` — so
`monkeypatch.delenv("HF_TOKEN")` was silently undone and `test_settings_precedence.py` measured the
developer's machine instead of the precedence rules it is named for. Since `.env` is the **documented**
place to put `HF_TOKEN` (settings' own error message says so), the suite went red for anyone following
the documentation. The module now neutralizes `load_dotenv` for its own tests.

### MARS's shared adapter was frozen by quantization — fixed 2026-08-09

**MARS was not training the thing that makes it MARS**, on every architecture, in every quantized
export. Measured requested-vs-realized trainable tensors before the fix:

| case | requested | realized | lost |
| --- | --- | --- | --- |
| decoder LoRA (`tiny-random-LlamaForCausalLM`) | 8 | 8 | 0 |
| **decoder MARS** | 8 | **4** | **4 — every `shared_qkv.*`** |
| encoder LoRA (`all-MiniLM-L6-v2`, classification) | 24 | 24 | 0 |
| **encoder MARS** | 24 | **12** | **12 — every `shared_qkv.*`** |

**Root cause.** `onnx_dynamic_quantization`'s `exclude_weights` holds the PEFT **target module** names
(`q_proj`/`v_proj`, `query`/`value`) and matches them as substrings of **node** names. That covers LoRA,
whose `lora_A`/`lora_B` live inside the target module's own subtree. MARS's shared adapter does not:
it is attached to the attention block (`.../attention/shared_qkv/...`), **outside every target
module's path — which is the entire point of sharing it across projections**. So it was quantized,
and a quantized tensor is a frozen tensor: `gen_artifacts` correctly refuses to ask for the gradient
of an int tensor and routes it to `frozen_params`.

Nothing failed. Training ran, the loss fell (the per-module factors were still trainable), and every
structural assertion passed. The method was quietly degraded to LoRA with a frozen, randomly
initialised down-projection.

**The fix** is the principled rule rather than a longer exclusion list: *a tensor the export declares
trainable must never be quantized*. `onnx_dynamic_quantization` takes
`exclude_trainable_initializers` — the exact `requires_grad` names, matched **exactly** against node
**inputs** (they are full initializer names, not fragments) — and the training export passes
`grad_layers` straight in.

**Evidence:**

- All four cases above now realize **every** requested tensor (8/8, 8/8, 24/24, 24/24).
- **LoRA graphs are byte-identical** before and after — same initializer set, same op-type histogram,
  same graph IO. The shipping SmolLM2 package is LoRA, so the 2026-08-09 device evidence stands.
- On MARS only the shared tensors moved: each `_quantized`/`_scale`/`_zero_point` triple replaced by
  one float weight (encoder: +12 float, −36 companions; `DequantizeLinear` 39 → 27). Nothing else.
- **Across the seam:** after a real 30-step encoder run, **12 of 12 shared tensors move under the
  optimizer** (they moved 0 of 12 before), and the loss falls **17.4%** where the frozen-shared
  version managed **5.6%** — so this is a quality improvement, not only a correctness one.

**The gate that would have caught it, and now will.**
`artifacts/trainable_gate.py::assert_every_requested_tensor_is_trainable` compares what
`training_config.json` declared trainable against what the graph actually realizes, and fails closed
naming the lost tensors **and the reason** (quantized away vs absent entirely). It is a **set**
comparison, not a count — a count can coincide while the wrong tensors are frozen. It is deliberately
in its own ORT-free module so it is testable in the core env (`builder.py` imports
`onnxruntime.training` at module scope); 6 tests run in `make check`. Verified against real
pre-fix export output: it raises *"4 of 8 tensors declared trainable … 4 were QUANTIZED"*.

This is the recurring failure shape again — `training_config.json` was right, the graph was right, and
nobody compared them.

> Unrelated pre-existing issue noticed while testing: `gen_artifacts` on
> `hf-internal-testing/tiny-random-LlamaForCausalLM` fails inside ORT with *"Duplicate definition of
> name (…post_attention_layernorm.weight_grad)"*. It fails identically on pre-fix exports, so it is a
> property of that degenerate fixture, not of this change. Real decoders are unaffected.

### Training-input binding — the graph decides, not a fixed list (#33 B2, 2026-08-09)

`train.cpp::train_step` built its input vector positionally with fixed shapes:

```cpp
std::vector<Ort::Value> user_inputs; // {input_ids, attention_mask, position_ids, labels}
const std::vector<int64_t> labels_shape({batch_size, sequence_length});
```

That is one architecture's answer hardcoded. The encoder classification training graph — the one
`tests/integration/test_encoder_training_gate.py` already pins — declares
`{input_ids, attention_mask, token_type_ids, labels[batch]}`: a different input **set**, a different
**order**, and a different label **rank**. Feeding it through the positional binder either throws
inside ORT or, worse, binds the wrong tensor to the wrong input.

**Now the graph decides.** `Ort::TrainingSession::InputNames(true)` gives the names in the order
`TrainStep` wants them; `training_inputs.h::plan_training_inputs` turns that list plus the batch
geometry into an ordered plan, and `train.cpp` executes it. `position_ids`/`token_type_ids` are
synthesized only when the graph asks for them. The label rank is derived from **how many label
elements the caller actually supplied** (`env->GetArrayLength`) rather than from a declared constant
that can drift from the data; `batch*seq` → per-token `[batch, seq]`, `batch` → per-sequence
`[batch]`, anything else fails closed naming both counts. When `seq == 1` the two are
indistinguishable by count and `[batch, seq]` wins, so the decoder keeps its exact shipped shape.

The decision is deliberately ORT-free so it is host-testable — the same reason `layer_name.h`,
`handoff_io.h` and `constants/merger_variant.h` are shaped that way. **8 new googletest cases**
(`tests/test_training_inputs.cpp`, `make test-cpp` now 30) pin the decoder plan as a regression, the
encoder plan, that the plan follows the graph's declared order rather than a canonical one, and that
both failure paths name the offending entity. `ModelFeature.Training` was deliberately NOT split:
variant/feature ids are a wire contract and the download group is genuinely identical. Compiles and
links arm64-v8a.

~~**Outstanding:** no encoder package has been exported or pushed, so the on-device encoder train step
has not run.~~ **CLOSED 2026-08-10** — the encoder package was exported and pushed (19 tests, 0
failures, 16 skipped: the generation suites correctly decline a graph with no token loop). The text
below describes what it took. That needs the encoder package shape end to end (classification inference graph, handoff
map, merger models, and per-sequence label plumbing through `ORTDataCurator`), which is materially
more than the binder.

### `lora_target` — fixed, and it changes decoder behaviour

The export's `lora_target` default (`["q_proj","k_proj"]`) disagreed with the architecture registry
(`("q_proj","v_proj")` on every decoder row) and nothing ever passed the argument, so the registry's
value was dead data and an encoder could not be targeted at all. **The registry is now the source of
truth**, with `--peft-target` (CLI) and `peft_target:` (export YAML) as the override.

⚠️ **This changes which modules the decoder trains — `q_proj`/`k_proj` → `q_proj`/`v_proj`.** The
registry matches the LoRA convention (adapt Wq and Wv) and is the better default.

✅ **RE-PROVEN ON DEVICE 2026-08-09** (S21 FE `SM-G990B` / Android 15 / arm64-v8a): a `TRAIN=1 RAG=1`
package exported under the new pairing runs the full instrumented suite **13/13, 0 skipped**
(`TrainMergeGenerate` + both `TrainConvergence` legs included). The staged package's
`train/trainable_parameters.json` was checked to list `q_proj`/`v_proj` pairs before the run, so the
suite genuinely exercised the changed targets rather than a stale export. The stale-evidence caveat
that stood here is therefore closed — see "Device suite" in the permanent section for the table.

### #34 — Charging-cycle training scheduler (`04_code_plans/02`) · checkpoint
- [x] Is `LinearLRScheduler.stateDict()`/`loadFromState()` implemented (closed in the 2026-08-07 remediation pass — `CosineLRScheduler` already implements both at `:77`/`:111` and is wired via `training_state.json`), and is the restore path verified to survive WorkManager chunk boundaries and process death?
- [x] Does a charging-constrained foreground `CoroutineWorker` checkpoint + resume cleanly across Doze? *(**chunk/checkpoint/resume PROVEN ON DEVICE 2026-08-09** — S21 FE `SM-G990B` / Android 15 / arm64-v8a, `ScheduledTrainingDeviceTest`: chunk 1 `globalStep 0 -> 2`, chunk 2 `2 -> 4`, scheduler step `2 -> 4`. **Doze deferral itself is NOT proven** and is not this library's behaviour — see the caveat below.)*
- [x] Does the **multi-chunk scheduled-train → resume** workflow pass (thermal/energy logged)? *(**YES, short run, 2026-08-09.** Two real chunks on the S21 FE with the trace emitted: `35.0C -> 36.1C` battery, thermal status 0 (NONE), charge counter flat at 4,076,000 uAh. **The multi-hour run under Android 16's tightened FGS quotas remains unproven** — recorded, not ticked into.)*

**Landed 2026-08-09.** `scheduler/{TrainingScheduleConfig,TrainingScheduler,TrainingWorker,ThermalGuard}.kt`;
foreground `dataSync` `CoroutineWorker` with a mandatory notification + cancel action, `Result.retry()`
at `THERMAL_STATUS_SEVERE`, self-chaining chunks whose constraints WorkManager re-evaluates each time.
The WorkManager shape is copied from `hub/PackageDownloadWorker.kt` per the plan, and no WorkManager
dependency entered the training path — the scheduler lives outside `ORTTrainerNative`.
`RuntimeCapabilities.supportsScheduledTraining` tracks `repo.isTrainingAvailable`; the LIBRARY manifest
declares `FOREGROUND_SERVICE{,_DATA_SYNC}`/`POST_NOTIFICATIONS`/`WAKE_LOCK` + the `dataSync` type.

### The two defects the device leg found (neither visible on the host)

**1. `maxSteps` is a CUMULATIVE target, not a per-chunk budget.** `ORTTrainerNative` computes
`totalSteps = maxSteps ?: epochs*stepsPerEpoch` and loops `while (globalStep < totalSteps)` **after**
restoring `globalStep`. `applyTo` passed the chunk size straight in, so chunk 2 restored `globalStep=2`,
logged *"Training for 2 steps"*, found `2 < 2` false, and exited **having trained nothing — while
reporting success**. Every chunk after the first was a silent no-op. Now `maxSteps = resumedGlobalStep
+ maxStepsPerChunk`, pinned by `TrainingScheduleConfigTest` at several resume points.

The host tests could not have caught this: they exercise the LR arithmetic across boundaries, which is
correct in isolation. Only running two real chunks against a real checkpoint shows the loop bound.

**2. `job.checkpoint()` returns null until the native trainer exists.** The worker read `stepsBefore`
through it *before* `start()`, so it was always 0 — which broke the chunk budget above and made the
`stalled` guard meaningless (anything `> 0` looked like progress). The worker now reads
`training_state.json` directly via `CheckpointInfo.read`, which needs no trainer.

Also found: **a scheduled job could not say what to train on.** The worker fabricated a bare
`ORTTrainingConfig`, so the first device run died with *"Unsupported task: none"*. `TrainingJobSpec` is
documented as "a reconstructable description of a training job", and a worker rebuilt after process
death has nothing but its input `Data`. `TrainingJobCodec` now carries the job (task, dataset, batch,
epochs, scheduler type + LR) as **data**, and `TrainingScheduler.schedule` **rejects a
`customPreprocess` lambda up front** — a lambda cannot survive process death, so a scheduled job must
name a registered task. Failing at schedule time beats failing hours later in a rebuilt worker.

### The device-suite mutation hazard — fixed systemically 2026-08-14, after it recurred

The training suites **mutate the package in place**: training rewrites `train/checkpoint` and
`training_state.json`, and the merge rewrites the per-tensor weight blobs under `inference/`. Tests
that assert about a clean package then read whatever an earlier test left behind.

**This bit twice, with the same two symptoms both times.**

1. *First occurrence.* `ScheduledTrainingDeviceTest` sorts before `TrainConvergenceTest` and
   `TrainMergeGenerateTest`, and its first version turned both red — **a NaN initial loss and a no-op
   merge**. Fixed by having that one class stash and restore `train/checkpoint` +
   `training_state.json`, with a note telling future authors to do likewise.
2. *Second occurrence, 2026-08-14.* Three new mutating `PostMergeNumericsTest` cases landed and the
   full suite came back **3 failures / 23 tests** — `TrainConvergenceTest` ×2 (**NaN initial losses**)
   and `TrainMergeGenerateTest` (**"merge wrote no new weights"**). The same two symptoms, from the
   same cause, in the same two victims. All three pass against a freshly pushed package.

**The lesson is about the shape of the first fix, not about the tests.** A per-class convention that
new authors must remember is not a fix for a suite-wide invariant; it postpones the recurrence until
someone adds a class without reading the note. The second fix is structural: `PristinePackageRule`, a
JUnit `TestRule` applied to every class that trains or merges, which captures the mutable artifacts
before each test and restores them afterwards **including on failure** — so one genuine failure no
longer cascades into misleading ones. It also recovers from an orphaned stash left by a test that died
before its `finally`, which would otherwise make the rule preserve corruption rather than prevent it.

Note what the first fix left uncovered even for its own class: it restored the checkpoint but not the
merged weights, because stashing ~60 tensors per test looked more expensive than re-pushing. That
trade was wrong — it cost a full suite run plus the investigation to re-derive a known cause.

**And the replacement got it wrong too, in a way worth recording.** `PristinePackageRule`'s first
version only copied stashed files *back*, skipping anything with no saved copy. But
`training_state.json` **does not exist in a freshly exported package** — training CREATES it, and its
presence makes the next training run RESUME instead of starting fresh. So the file survived every
restore, and `TrainConvergenceTest` kept reporting `NaN` initial losses while `TrainMergeGenerateTest`
(whose weight blobs pre-exist) was fixed. **Restoring state means restoring absence too**: a file that
was absent before a test and present after is restored by *deleting* it.

That is the same fail-open-by-omission shape as the transpose defect itself — `ObservedInit.transposed`
was honoured though never produced; here absence was state and was not treated as state. The pattern to
watch for is a restore/compare/validate that iterates over *what is there* and has no branch for *what
should not be*.

**The verification of that fix went wrong twice before it went right, and that is the more useful
lesson.** Two runs pairing a mutating class with `TrainConvergenceTest` came back green and were
reported as confirmation. Both were vacuous: the rule logs `captured N / restored M`, and both read
`181/181`, meaning the delete path never executed — the chosen "mutating" class did not write
`training_state.json`, so the failure could not occur. Guessing the writer twice cost two 10-minute
runs; reading it out of the previous suite's logcat (`ToolCallDeviceTest` writes at 18:54,
`TrainConvergenceTest` loads at 18:57) took one command. The real run reads **`captured 181 / restored
182`** — the 182nd artifact is the deletion — and the consumer logs "No existing training state found"
where it previously logged "Loading training state from:". **A fixture rule needs a counter that
distinguishes "did nothing" from "did the right thing", or its green is unreadable.**

**A new device test that trains or merges gets `@get:Rule val pristinePackage = PristinePackageRule()`.
Do not re-invent a per-test stash.** The precondition that survives: push a fresh package before a
suite run. What no longer holds is that the package degrades during one.

### What the device leg deliberately does NOT prove

Chunks are driven through `TestListenableWorkerBuilder`, not by waiting on real charging/idle
constraints, because **constraint evaluation and Doze deferral are Android's behaviour, not this
library's** — gating an automated test on someone plugging in a cable makes it a test of the room.
Unproven and recorded as such: Doze deferral, the notification's appearance, and multi-hour behaviour
under Android 16's FGS quotas.

### #35 — Federated codec & Python Flower simulation### #35 — Federated codec & Python Flower simulation (`04_code_plans/03`) · checkpoint
- [x] Does `FederatedAdapterRecord` derive ordering from `TrainableTensorCodec` (no new ordering) (F8)? *(order == `HandoffMap._sorted_entries`/`tensor_specs`; `test_codec_roundtrip` pins it; byte golden `federated_record.golden.bin` freezes the #36 serialization)*
- [x] Does an N-client Flower simulation aggregate adapter tensors and improve the metric? *(**YES — 2026-08-09, flwr 1.33 + the ORT-training profile, 4 clients x 3 rounds over the real `TRAIN=1` SmolLM2-135M package. Held-out eval loss on the AGGREGATED adapter falls monotonically: 8.7353 -> 8.5258 -> 8.2579.** Per-round payload 1,868,908 B — the 460,800 rank-r floats, against ~53 MB had it stayed merged-weight-shaped. Six defects were fixed to get here plus the vocabulary decision; see "The #35 simulation" below.)*
- [x] Is `adapterFormatVersion` linked to `weight_handoff_map.schemaVersion` (F1/F8)? *(`check_format` fails closed on mismatch via shared `check_compat`; `test_format_version`)*

**Code-complete 2026-07-15 (Python, no device).** `federated/` package + `cli federated simulate` + `tests/federated/` (all core-env). flwr kept OUT of the universal lock (out-of-band, like the ORT wheel — it downgrades protobuf/rich/typer + bumps mypy). Box open pending the manual N-client sim + aggregated-adapter logits smoke.

### The #35 simulation — PASSES 2026-08-09 (rank-r vocabulary)

`pip install "flwr[simulation]"` out-of-band (flwr **1.33.0**), `uv sync --python 3.12 --group
ort-training-local`, then `mobiletransformers federated simulate --package build/pkg --clients 4
--rounds 3 --local-max-steps 2` against the real `TRAIN=1` SmolLM2-135M package:

```
round 1: global adapter eval loss 8.735328
round 2: global adapter eval loss 8.525835
round 3: global adapter eval loss 8.257865      payload 1,868,908 B / round
```

The metric is measured **server-side on the AGGREGATED tensors** over a fixed held-out batch, and the
acceptance is **relative** (last round must beat the first) — a client-side training loss falls
whenever each client fits its own shard, which is true even when aggregation is broken.

#### The vocabulary decision (the thing that was actually blocking it)

The record carried 60 **merged inference initializers**; an ORT client produces 120 **rank-r factors**.
Two different objects, not an off-by-one. **Decided 2026-08-09: carry the rank-r factors.**

* per-round traffic is `d_in*d_out / (r*(d_in+d_out))` smaller — **measured 28.8x** on this model
  (1.87 MB vs ~53 MB), and the ratio grows as `r` shrinks;
* it matches the tier doc's "do not aggregate merged base weights", which v1's
  `merged_base_plus_adapter` role contradicted;
* a client no longer has to merge locally before it can send anything.

The cost, taken deliberately and now paid: **`tests/fixtures/federated_record.golden.bin` was
regenerated**, and `SUPPORTED_ROLES` gained the adapter roles. The merged roles stay **accepted on
read** — a peer may hold an older record and rejecting it as "unknown role" is worse than accepting it
— but nothing produces them.

#### Prerequisite: the handoff map had to DESCRIBE the factors, not just name them

`checkpoint_names` already named `adapter_A`/`adapter_B`; `tensorDtypes`/`tensorShapes` covered only
the merged `weight` role. So the "single source of tensor identity" could not describe the tensors
federation exchanges, and a consumer would have had to infer shapes from the rank — the re-derivation
that causes layer-identity defects. **Schema 1.1** adds `adapterDtypes`/`adapterShapes` per entry,
populated in `gen_artifacts` from the training graph's initializers (the one place that graph is
already open). Purely **additive**: `minReaderVersion` stays 1.0, a 1.0 reader ignores the new fields,
and 1.0 maps still load — they simply cannot describe their factors, which `codec_tensor_specs`
reports as a **fail-closed error naming the re-export**, never a silent fallback to merged.

#### The six defects fixed along the way

1. **The ServerApp asked for node ids before any supernode registered** — `get_node_ids()` read once,
   so round 1 died with "no client nodes available" on every run. Now waits, bounded, fail-closed.
2. **`ctx.node_config["package_dir"]` was unreachable.** `flwr.simulation.run_simulation` has **no
   `node_config` parameter**, so nothing could ever populate that key. Package location and shard
   index now travel in the round's `ConfigRecord`.
3. **The client looked for `<package>/train/`, the on-device CACHE layout.** A hub package puts the
   stage at `variants/<variantId>/train`, declared in the manifest's `paths.train`. Symptom: ORT's
   `INVALID_ARGUMENT : Invalid fd was supplied: -1`, naming no file. The CLI now resolves it through
   `manifest.select_variant(...)`.
4. **The client fetched its tokenizer from the Hub** — fails inside a Ray actor ("HF_TOKEN is not
   set"; actors do not inherit the driver's settings) and is the wrong shape regardless: a federated
   client should not phone a remote service to do local work. It now loads the package's tokenizer.
5. **Every client trained the same two hardcoded sentences**, so FedAvg averaged identical updates and
   "aggregation improves the metric" was unprovable in principle. Per-client shards added.
6. **Tensors were matched by checkpoint ITERATION order, not by name.** Codec order is (entries by
   canonical weight name) x adapter role; `state.parameters` yields whatever ORT stored. A positional
   mismatch would write one layer's `lora_A` over another's — mostly caught by differing shapes, and
   "mostly" is not a guarantee. Both the import and export paths now match by name and fail closed on
   a missing factor.

**#36 is now ungated by the decision** — the wire format is settled and the golden it must mirror is
final.

### The #37 architecture gate — PASSES 2026-08-09

Gate 1 is "run a real Gemma-3 export and see whether the graph is right". It does, and it is. Two
things had to be fixed to get there, and the second was a genuine defect.

**1. The toolchain fork.** `transformers 4.46.2` has no Gemma-3 at all — no `Gemma3ForCausalLM`, no
`Gemma3Config`, no `gemma3` row in `CONFIG_MAPPING_NAMES` — so the model could not be loaded and the
graph question was unreachable. That pin lives **only** in `ort-training-local`, as part of the
paired stack reproducing the source-built ORT wheel's build environment; the `export` extra separately
declared `>=4.45,<4.58`, which already admitted Gemma-3.

`[tool.uv] conflicts` already declared the group and the extra mutually exclusive, but that alone does
**not** force different versions — uv prefers one version whenever a single one satisfies both, and
4.46.2 satisfied `>=4.45`. Raising the export floor to **`>=4.50`** makes the two unsatisfiable
together, so they resolve separately, exactly as `numpy` and `onnx` already do for the 3.12 split. The
lock now carries `transformers` **4.46.2 and 4.57.6** side by side, and nothing else moved.

The training profile is deliberately untouched: floating a pin in that block has broken
`get_peft_model` before. ~~**Consequence, recorded:** Gemma-3 *inference* export works; on-device
FunctionGemma *training* is still blocked behind the same pin.~~

> ### ✅ CORRECTION 2026-08-15: the pin was NOT the blocker. FunctionGemma exports train-capable.
>
> The struck sentence stood for six days and was wrong in both directions. The experiment that settles
> it: a throwaway **sibling** group reproducing the paired stack exactly — torch 2.7.1, peft 0.13.2,
> numpy<2, onnx<1.19, the same source-built ORT wheel — with `transformers` as the single moved
> variable (4.46.2 → 4.57.6).
>
> **The sibling is gone as of the same day.** Once a third control (BERT encoder, `all-MiniLM-L6-v2`
> text-classification: 73,728 trainable, unchanged) joined the other two, the bumped range moved into
> `ort-training-local` itself and the experimental group was deleted. There is ONE training profile
> again. `paired_stack` in `third_party/onnxruntime/manifest.json` still records
> `transformers: 4.46.2` — that is a record of the wheel's *build* environment, not a runtime
> constraint, and manifest.json's own notes name only `numpy<2` and `onnx<1.19` as ABI constraints.
> Treating every paired_stack entry as load-bearing is precisely what kept this blocked.
>
> **`get_peft_model` did not break.** That specific fear is what kept the pin, and it did not reproduce.
>
> **The real blocker was an input-set mismatch, unrelated to any dependency.** `LlamaOnnxConfig.inputs`
> is `[input_ids, attention_mask, position_ids]`; `Gemma3TextOnnxConfig.inputs` is
> `[input_ids, attention_mask]` — Gemma-3 declares no `position_ids`. Optimum passes the generated
> dummy inputs to the traced module **positionally**, so `OnnxTrainerWrapper`'s four-parameter forward
> got `labels` bound to `position_ids` and nothing bound to `labels`, surfacing as a bare
> `TypeError: … missing 1 required positional argument: 'labels'` raised from `torch/jit/_trace.py`,
> several frames below anything this repo owns. Two further layers had the same decoder-shaped
> assumption baked in: `train_inference_parity.py` fed a hardcoded 4-tuple to `Module.__call__`
> (`Train input name index out of range. Expected in range [0-3). Actual: 3`).
>
> Fixes, all registry/observation-driven rather than special-cased:
> `OnnxDecoderNoPositionIdsTrainerWrapper`; `ArchitectureSpec.trainer_wrapper_class` (the architecture
> owns the input set because its `onnx_config_class` decides it, while the task still owns the label
> contract); `_check_wrapper_matches_config_inputs()`, which fails closed naming **both** lists so this
> class of bug can never again present as a `torch.jit` TypeError; and a parity probe that feeds by the
> names `Module.input_names()` reports.
>
> **The control that makes this trustworthy:** `SmolLM2-135M-Instruct` through the *same* bumped group
> exports fine, parity delta **0.0111 nats** — byte-identical to what its 2026-08-10 export recorded
> under 4.46.2. Raising the pin moved nothing for a model that already worked.
>
> **Published:** `mobiletransformers/functiongemma-270m-it` from `google/functiongemma-270m-it`
> (the real FunctionGemma, `Gemma3ForCausalLM`/`gemma3_text`, 18 layers, 640 hidden). 72 LoRA tensors
> (18 × q_proj/v_proj × A/B), rank 8, 368,640 trainable of 268,098,176. 102 files, 3.87 GB.
>
> **Still NOT proven: on-device Gemma-3 training.** This is a train-capable package. No training step
> has run on a phone for this architecture, and #37's passing demo remains a SmolLM2 decoder.
>
> **A second defect surfaced while validating it — the parity probe was measuring nothing meaningful
> off the Llama family.** `PROBE_INPUT_IDS` were Llama-family ids, so a Gemma-vocabulary model scored
> ~25 nats against a `ln(262144) = 12.48` uniform floor: both graphs confidently wrong on gibberish.
> The delta inflated with them, and FunctionGemma passed its own integrity gate by **0.06 nats**
> (1.4417 against a 1.5 bound) for reasons that had nothing to do with the package. The probe now
> tokenizes real English with the **package's own tokenizer**, resolved once and shared by both legs:
>
> | package | before (borrowed ids) | after (own tokenizer) |
> | --- | --- | --- |
> | SmolLM2-135M-Instruct | 12.01 / 12.02, Δ 0.0111 | 3.1803 / 3.3264, Δ **0.1461** |
> | google/gemma-3-270m | 24.41 / 24.20, Δ 0.2116 | 3.2870 / 3.3057, Δ **0.0188** |
> | google/functiongemma-270m-it | 25.70 / 27.14, Δ 1.4417 | 5.1658 / 6.0495, Δ **0.8837** |
>
> FunctionGemma's remaining gap is the probe being off-distribution for a model fine-tuned hard on
> function-call JSON, not a defect: re-probed on such JSON the same package gives 4.4513 / 4.1526,
> Δ **0.2987**. Quantization error is largest exactly where the model is least confident.

> ### ⚠️ CORRECTION 2026-08-09: the fork is NOT risk-free, and it broke a passing device test
>
> This section originally said the export-profile fork carried "zero risk" because it never touches
> the ORT-training wheel. That is true and **beside the point**: bumping `transformers` changes the
> **exported inference graph for every model**, not just Gemma-3.
>
> Evidence. A package re-exported after the fork records `transformersVersion: 4.57.6`, and
> `ConversationResetTest.twoSequentialPromptsBothComplete` — which passed twice earlier the same day
> on 4.46.2-exported packages — now fails on the **second** prompt of a conversation:
>
> ```
> inference step failed: Non-zero status code returned while running Gather node.
> Name:'/model/Gather_5'  indices element out of data bounds, idx=5 must be within [-5,4]
> ```
>
> A `Gather` out of bounds on the second turn is a KV-cache / position-index shape difference, i.e.
> exactly what a modelling-library upgrade changes. It ran **first**, on a **freshly pushed** package,
> so this is not the package-mutation hazard described below.
>
> **Options, none of them free:**
>
> 1. **Revert the export floor to `>=4.45`** and accept that #37's gate cannot run. The universal lock
>    cannot hold a per-model transformers version, so "4.57 for Gemma-3 only" is not expressible.
> 2. **Keep 4.57.6 and fix the consumer.** If the newer graph is *correct* and the Native engine's
>    conversation-reset path makes a stale assumption about cache/position indices, that assumption is
>    the bug and the upgrade merely exposed it. `03_code_plans/01` owns the reset path.
> 3. **Two export profiles.** Real but heavy: a second uv extra pinned for Gemma-3 exports only.

### RESOLVED 2026-08-10 — isolated, root-caused, and it was **option 2**

The one-variable experiment was run and it decides the question: **keep the fork, the consumer was
wrong.** The bug predates 4.57.6; the newer graph merely stopped tolerating it.

**The experiment.** `HuggingFaceTB/SmolLM2-135M-Instruct` exported twice through the same front door,
`--stages inference --genai --validate`, changing only the `transformers` line:

| | 4.46.2 | 4.57.6 |
| --- | --- | --- |
| ir / opset | 9 / 20 | 9 / 20 |
| graph inputs / outputs | 63 / 61 | 63 / 61 (**identical names**) |
| nodes | 7606 | 7425 |
| initializers | 273 | 273 |
| `Gather` nodes | 400 | 398 |

The interface is unchanged — no input appears or disappears — so nothing about the package contract
moved. But the node *sets* differ, and the decisive entry is that **`/model/Gather_4` and
`/model/Gather_5` exist only in the 4.57.6 graph**. `/model/Gather_5` is the node named in the device
failure.

> ⚠️ **Methodology trap that invalidated the first attempt.** `uv pip install transformers==4.46.2`
> followed by `uv run --extra export ...` silently produces a **4.57.6** export: `uv run` re-syncs the
> environment to the lock before executing, undoing the pin. Both packages came out byte-identical with
> `transformersVersion: 4.57.6` and the "no difference" reading was an artefact. Invoke
> `.venv/bin/python -m mobiletransformers.cli.main ...` directly for a pinned leg. (Silver lining: it
> proved the export is **deterministic** — identical inputs gave a byte-identical `model.onnx`.)

**What the new node does.** Tracing its two inputs:

```
/model/Gather_5   data    <- Flatten(Cast(attention_mask))          # the mask, flattened to 1-D
                  indices <- Add(Range(...), Mul(Range(...), Gather_4))
/model/Gather_4           <- Shape(Cast(attention_mask))[const]     # a mask dimension
```

i.e. 4.57's mask preparation gathers the **flattened attention mask at absolute positions derived from
the cache length**. If the mask is shorter than `cached + new`, the computed index runs one (or more)
past the end. The 4.46.2 graph has no such node and never indexed the mask this way.

**Root cause, reproduced on the host** against the real 4.57.6 graph (30 layers, 3 KV heads, head_dim
64), driving the session exactly as `ORTGeneratorNative` does:

| mask length | position ids | result |
| --- | --- | --- |
| `past + new` | `past..past+new-1` | OK |
| `past + new` | `0..new-1` (the old code) | **OK — so the position ids are NOT the crash** |
| `past + new - 1` | either | **FAIL: `idx=5 must be within [-5,4]`** — the device error, exactly |

**So the failure is a short attention mask, not a position-id error.** An earlier reading of this
correction guessed the position ids; that guess is recorded here as wrong because the reproduction
above rules it out. (The position ids *were* also wrong — restarting rotary phases at 0 on turn 2
degrades output — and that is fixed too, but it is a quality bug, not this crash.)

**The structural fix, not a counted one.** The mask length was derived from
`ORTGeneratorNative.pastAttentionMaskLength`, a Kotlin-side counter maintained as
`attentionMask.size - 1`. That is a **second source of truth** for something the session already knows,
and this is the layer-identity problem again in a third namespace: two consumers re-deriving one fact.

- `InferenceSessionCache::pastSequenceLength()` reads the extent off the cached tensors themselves
  (`[batch, kv_heads, sequence, head_dim]` → dim 2) and is now the single authority.
- `generateWithKVCache` **fails closed** when `mask < cached + new`, naming all three numbers, instead
  of letting ORT report a node the caller has never heard of.
- `ORTGeneratorNative.createModelInputs` asks the session and warns on any disagreement with the
  counter; `resetConversation()` now also clears the **native** cache (`nativeResetKvCache`), which
  nothing did before — "reset" previously left the session holding the old conversation's keys.
- The planning moved into `internal/runtime/GenerationInputs.kt`, ORT-free and host-testable for the
  same reason `training_inputs.h` was pulled out of `train.cpp`. **5 JVM tests** now pin the turn
  boundary that previously only a phone could reach, including the across-the-seam invariant that the
  last position must index the last mask slot.

**Consequence for the pins:** none. `transformers>=4.50,<4.58` stays, `tests/unit/test_dependency_profiles.py`
is unchanged, and #37's architecture gate keeps its evidence. The 2026-08-09 13/14 run is explained
rather than discarded.

`tests/unit/test_dependency_profiles.py` pins the fork — including that the lock really does carry
both lines — and states the distinction the pins previously blurred: **upstream ceilings** (`<4.58`,
`optimum~=2.1.0`) may not be fought, **paired-stack reproductions** (`torch==2.7.1`, `peft==0.13.2`,
`transformers==4.46.2`) may be forked per profile but never floated in place.

**2. A wrong config binding, invisible until exercised.** The row bound
`Gemma3ForCausalLM -> Gemma3OnnxConfig`. That is the **multimodal** config: its `__init__` does
`super().__init__(config.text_config, ...)`, and `google/gemma-3-270m` is `model_type: gemma3_text`
with no `text_config`. Optimum maps `gemma3_text` to **`Gemma3TextOnnxConfig`**. So every training
export of a text-only Gemma-3 would have died with `AttributeError`, and the existing unit test
asserted the wrong class, making it complicit.

This is precisely what the registry's own caveat warned about — *"corrected but NOT exercised end to
end, because the dotted paths resolve lazily"*. The correction was itself wrong, and nothing could see
it.

**The generalizing guard**, `tests/export/test_registry_matches_optimum.py`, cross-checks **every**
registry row against Optimum's own `TasksManager` mapping rather than restating expectations by hand:
15 rows verified, 5 skipped (inference-only or absent from this transformers line). Verified to catch
the defect — restoring the old binding fails with *"bound to 'Gemma3OnnxConfig', but Optimum resolves
['Gemma3TextOnnxConfig']"*. Optimum already knows the right answer for every model type; the registry
is now checked against it instead of maintained in parallel.

~~**Still outstanding for #37** (self-checks 2 and 3): the tool-call validator, the intent binder with
its allowlist and dry-run default, and the differentiation gate.~~ **Partly closed 2026-08-10**: the
validator (`agent/FunctionCallValidator.kt`, 11 tests) and the binder (`agent/IntentBinder.kt`, 5
tests) both landed, closing self-check 2. **Only the differentiation gate (self-check 3) is open**, and
it is the last engineering item in the project. The architecture gate no longer blocks it.

### #36 — Federated Android client & gateway (`04_code_plans/04`)
- [x] Are `exportTrainableTensors`/`importTrainableTensors` JNI added, and is the codec byte-identical to Python (golden test)? *(**Codec half DONE 2026-08-10; JNI half NOT started.** `federated/AdapterTensorCodec.kt` is byte-identical to `tests/federated/fixtures/federated_record.golden.bin` - asserted directly, and it passed on the first run. 11 codec tests. `packages/WeightHandoffMap.kt` was the concrete blocker: it modelled only the merged/load-side fields, so it could not describe the rank-r factors at all; it now reads schema **1.1** (`checkpointNames`, `adapterDtypes`, `adapterShapes`), carries the `ADAPTER_ROLE_ORDER` constant and a `toCheckpointName` twin of `checkpoint_names.py`, and `adapterTensorSpecs()` fails closed on a pre-1.1 package naming the re-export rather than falling back to merged weights. **The header is built by hand, not via Gson** - Python's `json.dumps(sort_keys=True)` uses `", "`/`": "` separators WITH spaces and sorts recursively, so a Gson-serialized header decodes fine and fails the golden. **JNI landed 2026-08-10** as `nativeExportCheckpointTensor` / `nativeImportCheckpointTensor` (first use of `Ort::CheckpointState::UpdateParameter` in this project), plus `federated/FederatedRound.kt` composing them with the codec and the consent gate. **Deviation from the plan doc, deliberate:** it names `exportTrainableTensors(session, handoffMapPath) -> ByteArray`, i.e. the whole record assembled in C++. These move BYTES only. Assembling the record natively would be a SECOND implementation of the exact format the cross-language golden exists to keep from drifting, and would need `handoff_io.h` extended plus a C++ JSON writer reproducing Python's `sort_keys` separators — two implementations of one wire format is the failure this project keeps paying for. So C++ moves tensor bytes, Kotlin owns the format, and order/naming/dtype still come from `weight_handoff_map.json`. Import reads the EXISTING parameter to learn shape/dtype rather than trusting the sender's description, and refuses on a byte-count mismatch instead of truncating. `CheckpointTensorStore` is an interface so clipping and name-matching are host-testable; 6 round tests. **`gateway.py` + `federated serve` landed 2026-08-10**, verified on the REAL `TRAIN=1` SmolLM2 package: two client records over its 120 declared adapter tensors, weights 1 and 3, aggregate to exactly `(1*1 + 3*3)/4 = 2.5` in every tensor; 1,868,857 B global record. 8 tests. It is the round STATE MACHINE, not a web server - blobs in, validated, averaged via the existing `federated_average`, global record out; transport stays the transport's problem, which is what lets the aggregation be tested without a socket. Enforced: tensors matched **by name, never by position** (a record serialized in reversed order still aggregates correctly - the #35 defect), a client whose record disagrees with the package is **dropped rather than coerced**, dropout completes the round over survivors, and `min_clients` makes it FAIL rather than publish an aggregate two devices decided. **The device round-trip PASSES 2026-08-10** (S21 FE `SM-G990B` / Android 15 / arm64-v8a) — see "The #36 device round-trip" below.)*
- [x] Are the privacy/security gates (consent, TLS, auth, clipping/DP) addressed before any real-user run? *(**2026-08-10**: `federated/FederatedConfig.kt` + `FederatedConsent`, 5 tests asserting the REFUSALS - a test that only checked the happy path would pass against a gate that did nothing. `requireRoundIsPermitted()` refuses: a build without `BuildConfig.FEDERATION_ENABLED` (off by default, mirroring #22's `ADAPTER_UPLOAD_ENABLED`), absent consent, a non-https gateway, a blank client-auth token, and a non-positive clip norm. Consent is a **record with a policy version and a timestamp, not a boolean**: consent given for an older policy does not carry over, because what is shared having changed means the user never saw it. `dpNoiseMultiplier = 0` is legitimate for a closed cohort but must be stated rather than defaulted, and `usesLocalDp` makes it visible. Before this a grep for "consent" across the repo returned nothing.)*
- [x] Is it hard-gated on #35 passing first? *(**CORRECTION: #35 passed on 2026-08-09, so this gate is OPEN** - the rank-r vocabulary decision is settled and the byte golden is final, which is exactly what unblocked the codec above. The original note below was written mid-cycle when #35 had not yet passed; it is kept for its reasoning about WHY the gate existed, not for its verdict.)* *(original: **yes, and the gate is CLOSED as of 2026-08-09: #35 did not pass.** No #36 code was written this cycle, deliberately. The simulation now runs real client training but stops at the merged-vs-rank-r tensor-vocabulary conflict, and that conflict decides #36's wire format: if the record ever switches to rank-r adapters the checked-in `tests/fixtures/federated_record.golden.bin` — the whole point of which is to prove Kotlin/Python byte-identity — is regenerated. Building the Kotlin codec against a format that may change would be building the wrong thing.)*

### The #33 encoder legs — ALL PASS 2026-08-10

The 2026-08-09 pass proved the host training chain and 2026-08-10 made the *package* task-driven, but
nothing had ever run the encoder TRAINING export past `gen_artifacts`: `test_encoder_training_gate.py`
stops there and never calls `export_inference_package`. The first real attempt
(`make device-package MODEL=sentence-transformers/all-MiniLM-L6-v2 TASK=text-classification TRAIN=1 RAG=0`)
failed closed:

```
12 of 12 handoff entries name a checkpoint parameter that does not exist in checkpoint.
  base_model.model.bert.encoder.layer.0.attention.self.query.base_layer.weight
```

**It is the layer-identity problem in a second namespace, and the check caught it exactly as designed.**
The peft→ORT rewrite was spelled `base_model.model.model.` → `backbone.model.` in **four** places
(`artifacts/checkpoint_names.py`, `cpp/layer_name.h`, `packages/WeightHandoffMap.kt`, and open-coded
eight more times across `training/validators.py`, `training/merge_validators.py`). That is the real rule —
strip peft's `base_model.model.` wrapper, prepend ORT's `backbone.` wrapper — with a **decoder's own
first module (`model.layers…`) baked into it**. Identical output for every decoder; a silent no-op for
BERT, whose path is `bert.encoder.layer…`. Ground truth from a real MiniLM export + its checkpoint:

| space | spelling |
| --- | --- |
| `peft_mapping` key | `base_model.model.bert.encoder.layer.0.attention.self.query` |
| ORT checkpoint | `backbone.bert.encoder.layer.0.attention.self.query.base_layer.weight` |

Fixed by making the constants the **wrapper pair** (`base_model.model.` → `backbone.`), which is
byte-identical for decoders (pinned by the existing tests, plus new encoder cases in
`test_checkpoint_names.py`, `test_layer_name.cpp` and `CheckpointNameTest.kt`), and by routing the eight
open-coded copies through `to_checkpoint_name` — the Python half of what `layer_name.h` already did for
C++.

**Results after the fix** (S21 FE `SM-G990B` / Android 15 / arm64-v8a, 2026-08-10):

* the encoder exports end to end to a valid package — `features: [core, inference, train]`, 12 base
  layers, 24 trainable factors, 73,728 trainable parameters, `architectures: ["bert"]`;
* `EncoderTrainStepDeviceTest` runs 2 real steps on device through the per-sequence-label path
  (`cola_cls` → `TaskPreprocessor.classLabel` → `perSequenceLabel` → unpadded collation), finite losses
  `[0.72216845, 0.7223699]`, and **24/24 adapter factors moved** in the checkpoint — read back by name,
  because a run that computes a loss and applies nothing is indistinguishable from outside;
* device suite over the encoder package: **19 tests, 0 failures, 16 skipped**.

**Those 16 skips are the second thing this leg needed.** A `text-classification` package is
train-capable and installs identically to a decoder one, so every generation suite ran against it and
failed at the first `generate()` on a graph with no token loop — a task mismatch reported as a defect.
`DeviceModel.selectedTask` / `requireDecoder` read the task from the `optimum_config.json` the exporter
writes beside the graph, and the decoder-only suites now skip naming it. An unknown task counts as a
decoder, so packages predating that side-car behave exactly as before.

**Worth knowing:** an encoder package reports **`Training` only** to the facade — no `ModelFeature.Inference`.
That is correct rather than a gap: `isGenerationAvailable` keys off `generation_config.json`, which the
task-driven packaging deliberately does not emit for a task with nothing to generate. An encoder's
inference path is the embedding one (`generateEmbedding` / the RAG embedder), which already ships. It
does mean `TrainingWorker` cannot run on an encoder package, which is why `ScheduledTrainingDeviceTest`
skips there too.

**The device cache holds one package at a time**, and it currently holds the ENCODER. The 15/15 decoder
suite result below is from the SmolLM2 package; reproduce it with
`make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct TRAIN=1`. `scripts/device_package.sh` now
takes `PKG=` so the two host packages (`build/pkg`, `build/pkg-encoder`) can coexist — the gateway needs
the host copy of whichever package the device holds.

### The #36 device round-trip — PASSES 2026-08-10

Export adapter factors from a **live ORT checkpoint on hardware** → aggregate on the host → import the
aggregate back → prove the checkpoint moved. S21 FE `SM-G990B` / Android 15 / arm64-v8a, driven by
`scripts/federated_round_device.sh` (`make device-federated`).

The middle of a round is a host process, so the seam cannot be crossed inside one instrumentation run.
The script drives both halves and the step between them; `FederatedRoundDeviceTest` is two phases, each
`assumeTrue`-skipping without its input so an ordinary `make device-test` reports them as skipped:

| leg | measured |
| --- | --- |
| round 0 — export from the live checkpoint | 120 factors, **1,868,879 B** (1.78 MiB) upload payload |
| host — synthetic peer (×3) + `federated serve`, weights 1:3 | **1,868,857 B** global record |
| round 1 — import | **120/120** tensors written, **120/120 changed value**, and every one byte-equal to the aggregate |
| round 2 — import → bounded local train → export | **120/120** factors moved by training |
| reload after `destroySession(save=true)` | **120/120** factors present — the checkpoint on DISK moved |

`BuildConfig.FEDERATION_ENABLED` stays false by default; the runs pass `-PmtFederationEnabled=true`.
A test-only bypass inside `FederatedConfig` would have weakened the gate it is testing, so the switch
stays the build switch and the JVM refusal test `assumeFalse`s when the property is set.

**Two real defects, both invisible to either half alone:**

1. **`startTraining` released the session on every exit path** — with the checkpoint saved
   (`saveModelAtEnd`) or without, but always. That is why every post-training step in this library (the
   merge, the checkpoint cadence) happens *inside* `startTraining`: afterwards there is nothing left to
   act on. The federated export reads the checkpoint it just trained, so the first device run failed
   with "no live ORT training session". Fixed with `ORTTrainingConfig.keepSessionAtEnd` (default
   `false`, so every existing caller is unchanged); when set, the save happens via `saveModel` and the
   caller owns the teardown.
2. **A bounded round applied no update at all.** `optimizerStep` fires only on
   `globalStep % gradAccumSteps == 0` (and never at step 0), and `gradAccumSteps` defaults to **4** — so
   a `maxSteps = 1` round accumulates gradients, never applies them, and uploads the global adapter back
   **unchanged** while every callback reports a successful training run.

The second was hidden by a *weak assertion* first, which is the more useful lesson: comparing round 2's
export against the **aggregate** reported "60/120 factors moved" and passed. That 60 was **clipping**,
not training — `exportUpdate` clips to `clipNorm`, so every tensor the clip rescaled counted as
"trained". Comparing against round 1's **export** instead — same clipping on both sides — isolated
training and reported **0/120**, which is what exposed the accumulation defect. With `gradAccumSteps = 1`
it reports 120/120. *A federated round is exactly where a silently-empty update is invisible: the record
is well-formed, the right size, and aggregates fine.*

Also landed with it: `federated/NativeCheckpointTensorStore.kt` (the only binding of the JNI pair to
`CheckpointTensorStore`), `federated/FederatedTrainingRepository.kt` (import → bounded local train →
export, with the ordering asserted on the host over a substitutable `AdapterExchange`), and
`scripts/federated_peer_record.py`. The peer is **explicitly synthetic**: one phone is available and
`min_clients` is 2, and submitting the device's own record twice would produce an "aggregate"
numerically identical to what the device already holds — an import that writes back what was already
there cannot distinguish a working round from a no-op. So it proves the transport/aggregation/import
seam, **not** multi-device convergence (that is #35's simulation, which trains real clients).

### #37 — FunctionGemma architecture gate & intents (`04_code_plans/05`) · checkpoint
- [x] Did the architecture gate pass — Gemma-3 **inference**-graph export added as a registry entry? *(**GATE 1 PASSES 2026-08-09.** `google/gemma-3-270m` exports end to end through the normal front door and `mobiletransformers export --validate` produces a #13-valid package: 18 layers, canonical `logits` + 36 `present.N.key/value` + 36 `past_key_values.N.key/value`, optimum's own validation max-diff ~1e-4. It took a dependency fork AND a real registry defect fix — see "The #37 architecture gate" below. `inference_model_class` stays `None` by design: that field is the vendored GenAI-builder path, while the shipping inference export goes through optimum's `main_export`.)*
- [x] Does it **never** execute raw model output (allowlist + dry-run + validated tool calls)? *(**2026-08-10**: `agent/FunctionCallValidator.kt` + `agent/IntentBinder.kt`, **16 JVM tests**. The guarantee is structural rather than procedural: `IntentBinder.dryRun` accepts only a `ValidatedCall`, which nothing but the validator can construct, so raw model text cannot reach it; and the intent action is read from the app's `ActionSpec.allowedIntent`, never from model output — **a model selects an action, it cannot name an intent**, so the reachable intent set is fixed when the allowlist is built. `IntentBinder` holds no `Context` and has no `startActivity` call site; executing is the caller's decision with the caller's own `Context`, deliberately not offered as a convenience. Undeclared parameters are refused (so an `allowedIntent` cannot be smuggled in as one), missing ones are refused rather than defaulted, an **unrecognised validation rule rejects rather than passing by default** (a typo in the app's allowlist must not silently disable the check it was written to perform), and a duplicated action name fails at construction instead of `associateBy` silently keeping the last.)*
- [x] Does the train → validated-tool-call → dry-run-intent demo show ≥2 differentiators? *(**PASSES 2026-08-14 on the S21 FE `SM-G990B` / Android 15 / arm64-v8a**, `ToolCallDeviceTest` 2/2, 754 s. The chain runs end to end on hardware: a per-user action set generated from the app's own allowlist → 108 on-device LoRA steps (loss drop 99.5%) → merge → greedy generation → `{"actionName": "set_alarm", "parameters": {"time": "07:30"}}` for `wake me at 07:30`, **Accepted** by the app's own validator → bound to a dry-run intent. Three differentiators are demonstrated, not merely present: **personalized per-user action sets** (the dataset is derived from the allowlist, so accepted completions are accepted by construction), **privacy-preserving local data** (the corpus is generated on-device from a declaration and never leaves it), and **local validated tool-call generation with structural non-execution** (the second test, `theAllowlistStillRefusesAnActionTheAppNeverDeclared`, confirms an undeclared action is still refused after fine-tuning). This run was only a fair test after the transpose defect below was fixed; the first two attempts measured a broken merge.)*
- *This box was open for the whole project and is the last engineering item in it. Everything except the demo remains as recorded below.*

### The #37 demo run — PASSES 2026-08-14, after a root cause FOUND and FIXED: the merge wrote every weight transposed

> **Superseded header, twice.** This section first read "FAILS, root cause narrowed to merge numerics"
> and advised against spending a cycle on more steps. The narrowing was right; the diagnosis took three
> more rounds and two wrong hypotheses (see below). **The defect is found, fixed and verified — and the
> gate it was blocking now passes.**

**The gate result (S21 FE `SM-G990B` / Android 15 / arm64-v8a, 2026-08-14, `ToolCallDeviceTest` 2 tests
/ 0 failures / 754.4 s, against a freshly pushed pristine package):**

```
ToolCallDeviceTest: steps=108 lossDrop=99.5%
ToolCallDeviceTest: instruction='wake me at 07:30'
                    raw='{"actionName": "set_alarm", "parameters": {"time": "07:30"}}<|im_end|>'
                    -> Accepted
```

Compare the identical run before the fix, which emitted token 198 (newline) 48 times. Same package,
same corpus, same 108 steps, same 99.5% loss drop — **the only thing that changed is that the merged
weights are now stored in the orientation the inference graph reads them in.** The model was learning
the task correctly the whole time; the merge was destroying the result on the way out. Note also that
the emitted `<|im_end|>` is the EOS fix landing: the completion terminates rather than running to
`maxNewTokens`.

The second test (`theAllowlistStillRefusesAnActionTheAppNeverDeclared`, 2 ms) matters more than its
runtime suggests: it confirms fine-tuning a model to emit calls did **not** widen the reachable action
set. Refusal of an undeclared action is structural — it holds whatever the model emits.

**The bug:** `weight_merger.cpp` wrote merged weights in the merger's own convention
(`[out_features, in_features]`, PyTorch `nn.Linear` / checkpoint layout) into an ONNX `MatMul`
initializer that the inference graph reads as `[in_features, out_features]`. **Every merged weight was
stored transposed, for as long as on-device merging has existed.** Fixed in
`write_raw_tensor_atomic`, which now transposes 2-D float weights and verifies the result against the
handoff map's declared on-disk shape, failing closed on disagreement.

**Verified on device** (S21 FE `SM-G990B` / Android 15 / arm64-v8a), by
`PostMergeNumericsTest::aLargeAdapterDeltaSurvivesTheMergeIntoTheInferenceGraph` — **PASS, 1040 s**:

| cross-entropy on the same text | pristine | before fix | after fix |
| --- | --- | --- | --- |
| text the adapter memorised | — | 12.445 | **1.295** |
| unrelated English | 4.651 | 15.446 | **5.529** |
| uniform-prediction floor | 10.803 | | |

Before the fix both numbers sat *above* the uniform floor — the merged model was worse than random.
After it, the memorised text costs 1.295 nats (far below the 4.651 baseline: the model learned, and the
learning survived the merge) and unrelated English costs 5.529 (mild, expected forgetting after 100
steps on one repeated example). **This is the first numerically verified train → merge → generate on
this project.**

Proof of the mechanism, on raw bytes: after a merge whose delta was *exactly zero*
(`adapter_B l2=0.000000`, scale 1.0, correct shapes), `max|written − original.T| = 1.9e-09`. After the
fix the relation inverts — `max|written − original| = 1.9e-09`.

**A second, genuine bug was fixed on the way:** the LoRA merge scale read **uninitialized memory**.
`PeftMapping mapping;` is default-initialized, and `create_lora_mapping` emits only
`adapter_A`/`adapter_B` — so `alpha`/`rank` were never assigned and the merger computed
`base + <indeterminate float> * (B @ A)`. It is now `alpha / rank` (= 1.0 here, logged per layer).
Undefined behaviour on every LoRA merge ever run, but worth ~1% of the damage; the transpose was the
cause.

**Two hypotheses were wrong and are recorded so the method failure is not repeated:** (1) the scale
error was claimed to "fit the measurements exactly" — disproved by experiment, it moved the numbers
~1%; (2) "a zero-adapter merge is bit-exact identity" was reported as a control when `merge()` without
a prior training run **never executes a merge at all**, so the control was vacuous. The general lesson
— magnitude-based checks cannot detect a permutation — is in Operational knowledge.

### Re-validation after the transpose fix

Every training result recorded before 2026-08-14 was measured against the broken merge, so none of it
was evidence for the current code. Re-run on the S21 FE `SM-G990B` / Android 15 / arm64-v8a, each
against a **freshly pushed pristine package** (the merge rewrites the device copy in place, so a
re-used package re-merges to identical bytes and proves nothing):

| leg | result |
| --- | --- |
| **FULL DEVICE SUITE** | **23 tests / 0 failures / 3 assumption-skips / 3219 s, 2026-08-14.** The first end-to-end pass this suite has ever had — see "The device-suite mutation hazard" for why it could not run in one pass before. Skips are the encoder-package and federation-disabled assumptions, both by design. |
| **#37** `ToolCallDeviceTest` | **PASS** 2/2 — accepted `set_alarm` call, see above. Passed in three separate runs (754 s, 1018 s standalone, and within the suite). |
| **#18/#19** `TrainMergeGenerateTest` | **PASS**. `merged 60/90 tensors`, and generation after the merge reads `< the city of Paris, a city of>` — coherent English, identical to the pre-training baseline. Before the fix this leg produced 48 newlines and still passed, because its only assertion was `isNotEmpty()`. |
| **#9** `PostMergeNumericsTest` large-delta | **PASS** — 1.295 nats memorised vs 5.529 unrelated (table above). **Reproduced to the last digit in three independent runs** (`1.2951252753772409` / `5.529056724499491`), so the result is not a fixture artefact. |
| **#9** zero-delta merge identity | **PASS** — `zero-adapter merge: before=4.651482603007065 after=4.6514830258392115 (60/90 rewritten)`. A merge that genuinely ran (the rewrite count proves the write path executed) with an exactly-zero delta is the identity to float round-trip precision. This is the control that was reported vacuously earlier the same day and had to be retracted; it now measures what it claims. |

**On `TrainMergeGenerateTest`'s output being *unchanged* by training:** that is correct and expected,
not a weak result. One LoRA step at the default learning rate does not move a greedy argmax over 8
tokens; the test's evidence is the 60 changed `.bin` files, not the text. What the text now proves is
the thing that was silently false for months — that the model still *works* after a merge.

**The offline Python merger was never affected.** `merge_validators.py` computes merged tensors in
memory for validation and never writes `.bin` files, so exported packages always carried correct
weights. Only device merges corrupted them. (The `write_raw_tensor_atomic` comment asserting
offline/device byte-identity was describing a comparison nobody had run; corrected.)

**Cross-language agreement, now pinned by a test.** `derive_transpose_policy` run over a real export's
shapes resolves the package to `already_transposed_for_inference` — the same orientation
`weight_merger.cpp` independently *observes* from the tensors at merge time (`merge orientation:
transpose_for_inference=1 (observed=1)`). Two implementations, two languages, real data, same answer;
`test_the_derivation_agrees_with_a_real_exported_package` fails if either side drifts. The earlier
tests used synthetic fixtures, which is precisely how the original defect hid — the fixtures agreed
with the broken code because both said `no_transpose`.

**Still not re-validated:** the encoder-package legs (`EncoderTrainStepDeviceTest` and the 19-test
encoder suite) exercise a different package, and #9's atomic-overwrite-under-kill has never been
tested at all.

### Original narrowing from the PRE-FIX runs (kept — the elimination was sound)

*Everything in this subsection describes the two runs made **before** the transpose fix. It is kept
because the elimination it performed is what located the defect; read it as history, not as status.*

`ToolCallDeviceTest` had never executed (written 2026-08-14 with no device attached). It then ran
twice on a pristine `TRAIN=1` SmolLM2-135M-Instruct package, S21 FE / Android 15 / arm64-v8a.

**The result at that point: the model did not emit an accepted call. It emitted token 198 (newline)
48 times.**

What the run proves, and this is the useful part — *the failure is not where the handoff predicted*:

| link | measured |
| --- | --- |
| training converges | **yes, emphatically.** Loss 5.85 → 0.0051–0.0121 over 108 steps (99.5% drop). This is memorisation, as intended. |
| merge runs | **yes.** 60 PEFT mappings loaded, `merger_lora_fpin_fpout.onnx` applied, "Completed lora merger" for all 60 q_proj/v_proj tensors across 30 layers. |
| merged weights reach inference | **yes.** 60 "Loaded merged initializer: model.layers.N.self_attn.{q,v}_proj.MatMul.weight ← ….bin". **Zero** occurrences of the "loading base weights" fallback warning, so this is not a silent downgrade. |
| the prompt matches training | **yes.** Generation input tokens logged as `[1, 42037, 549, 418, 216, 32, 39, 42, 35, 32]` — BOS + `wake me at 07:30`, the same sequence the curator now builds. |
| generation | **collapses.** Greedy argmax is token 198 at every step. |

So convergence, dataset size, step count and prompt format were all **excluded** as causes, correctly.
The conclusion — "the merged weights are not numerically the weights that were trained" — was right,
and the suspected area (the adapter→inference weight-space mapping) was the right neighbourhood. The
actual defect was one step further out: not the `B·A` delta's orientation but the **whole merged
weight's** orientation on write, which is why it corrupted the model even when the delta was zero.

~~**Do not spend another cycle on more steps or a smaller corpus.**~~ Correct at the time and still
correct: the lever was never the dataset. The cause was the transpose above.

`PostMergeNumericsTest` passing (below) does **not** contradict this: it asserts that merging *changes*
the computation and stays finite, not that the delta is the correct one — the "merge happened without
being correct" shape the operational notes warn about.

**Two real defects were found and fixed getting here** (both in `ORTDataCurator` /
`TaskPreprocessor.formatsPromptForGeneration`, 4 JVM tests in `ChatFormattedTrainingPromptTest`):

1. **BOS mismatch.** Training tokenized with `prependBos = false` (the `tokenize` default) while
   `generate` always prepends BOS on the first turn. Fixed for tasks that opt in.
2. **No stop signal.** Completions carried no EOS, so a trained model runs to `maxNewTokens`.

A third, latent one is recorded but not reachable on this package: `ORTTokenizerNative` reads
`chat_template` only from `tokenizer_config.json`, while the SmolLM2 export writes it to a sibling
`chat_template.jinja`. So `chatTemplate` is null and **neither** training nor generation applies a chat
template (`W/ORTTokenizerNative: Chat template not found …`). For a package that *does* carry the key,
training and inference would previously have disagreed; that is now closed by the same opt-in.

Sizing note: `maxSteps` is an upper bound, not a target — training stops at the end of the epoch. The
first run asked for 120 steps and took 54 (108 rows / batch 2). The corpus is now 216 rows → 108 steps.

*(Original status, still accurate for the other four links:* Present: local validated tool-call generation + Android intent binding (above), personalized per-user action sets (`agent/mobile_actions.py` generates a user's dataset FROM their app's allowlist, so completions are by construction calls the validator accepts — 8 tests pin that seam), and privacy-preserving local data (the dataset is generated on-device from a declaration and never leaves it). On-device training is device-proven for decoders generally. It remains blocked for FunctionGemma specifically by `ort-training-local`'s `transformers==4.46.2` — a #2/#3 dependency decision, not a #37 one.

**Superseded 2026-08-14, twice:** this paragraph used to end "what is missing is the demo itself wired end to end … that has not been done", then was corrected to "what is missing is no longer the wiring but the merge-numerics defect the run exposed". Both are now closed: the demo is wired, the defect is fixed, and the run passes on hardware. Nothing in #37 is outstanding.)*

---

# Operational knowledge (permanent)

> Transferred out of `HANDOFF.md` on 2026-08-09 so it survives the handoff being reset for each new
> cycle. **This section is durable**: environment facts, hard-won gotchas, the device workflow, the
> defect patterns, and the final gate results. `HANDOFF.md` is now only "what the current cycle is
> doing"; anything a cold agent needs regardless of cycle lives here.

## Environment & dependency profiles

- **`uv` is the entrypoint** (`~/.local/bin/uv`). Two Pythons: system **3.10** (core/dev default) and
  **3.12** (required for the training profile).
- **Core/dev:** `uv sync --frozen --group dev --python 3.10` then `make check`
  (lint + format + typecheck + parity + guard + tests).
- **Training:** `uv sync --python 3.12 --group ort-training-local` then `make test-train`. The
  source-built `onnxruntime-training==1.23.0+cpu` wheel is **git-ignored** at
  `third_party/wheels/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl`; SHA256 and build
  flags are in `third_party/onnxruntime/manifest.json`.
- **Export:** `uv sync --extra export --python 3.12` (optimum-onnx + public onnxruntime).
- **Profile isolation is enforced by design.** `uv sync --extra export --group ort-training-local`
  **errors** — the four `onnxruntime`-providing profiles must never co-install
  (`[tool.uv] conflicts`).
- **`uv run` mutates the shared `.venv`.** After any export- or training-profile run, reset with
  `uv sync --frozen --group dev --python 3.10` **before** `make check`, or mypy fails on numpy's
  3.12-only stubs. This is the single most common way to "break" the repo.
- **Android:** `JAVA_HOME=/opt/android-studio/jbr` (JDK 17), SDK at `~/Android/Sdk`. AGP 8.5.1 /
  Kotlin 1.9.0 / Gradle 8.7 / compileSdk 34 / NDK r26.x / CMake 3.22.1.
- **arm64-v8a only.** `jniLibs/x86_64` has never had `libonnxruntime.so` or the tokenizers archives —
  absent here *and* upstream — so `libmobiletransformers.so` has never existed for x86_64. x86_64 was
  deliberately dropped from `abiFilters` rather than advertising an ABI that dies at
  `System.loadLibrary`. **No x86_64 emulator.**
- **Untracked vendored native deps** (all `.gitignore`d): `cpp/includes/google` (protobuf headers),
  `jniLibs/`, `aarLibs/`. Provisioned locally from the sibling `../ORTTransformer` checkout. Their
  absence is why CI's `android-assemble` self-skips. **They are the only thing a `git clone` does not
  bring**, so moving this repo to another machine means carrying, out of band:
  `jniLibs/arm64-v8a/` (`libonnxruntime.so`, `libort_gen.so`, `libonnxruntime-genai{,-jni}.so`,
  `libtokenizers_c.a`, `libprotobuf-lite.a`, `libobjectbox-jni.so`, `libonnxruntime4j_jni.so`),
  `aarLibs/onnxruntime-genai.aar` (~40 MB), `cpp/includes/` (~24 MB) — all under
  `android/MobileTransformers/MobileTransformers/src/main/` — plus, only if you need the training
  side, `third_party/wheels/onnxruntime_training-*.whl` (~632 MB, see the macOS note below).
  Everything else (`.venv*`, `build/`, `.gradle`, `.cxx`, `onnx_models/`, `dist/`) is recreatable.
- **macOS cannot run the training side without a rebuild.** The source-built wheel is
  `cp312-linux_x86_64` and **will not install or load on macOS**. Blocked there until it is rebuilt for
  `macosx_arm64`: the real train→merge→generate legs (#18/#19), the train-capable package
  (`make device-package TRAIN=1`), and #34's real training run. Everything else works — all core/dev +
  export + genai-smoke Python, every Android host build/test, the inference+GenAI device package and
  its device legs, #22 PEFT materialization (needs only the `train` extra, i.e. torch CPU) and #35
  federated. Rebuild via `scripts/build_ort_training_*.sh` + `third_party/onnxruntime/BUILD.md`, then
  repoint the `ort-training-local` group in `pyproject.toml`. *(Carried over 2026-08-14 from
  `MAC_WORKPLAN.md`, which was retired: it was never executed, nothing referenced it, and its premise —
  "#30/#31/#33/#34/#36/#37 not yet started" — was overtaken by the device runs on this Linux box.)*

## Upstream version ceilings (verified against PyPI 2026-08-09)

| package | pinned | why |
| --- | --- | --- |
| `optimum-onnx` | 0.1.0 | **is** the latest; only four releases exist |
| `optimum` | 2.1.0 | `optimum-onnx` declares `optimum~=2.1.0`; **2.3.0 cannot install** |
| `transformers` | 4.46.2 | `optimum-onnx` declares `<4.58`; also paired with the ORT-training wheel |
| `torch` | 2.7.1 | the wheel's real ABI (docs once guessed 2.5.1) |
| `numpy` | <2 (3.12 fork) | wheel built against the numpy 1.26 ABI |
| `onnx` | <1.19 (3.12 fork) | ORT 1.23 caps ONNX IR at 11; onnx>=1.19 emits IR 13 |

The `pyproject.toml` caps mirror upstream's own ceilings — they are **not** stale pins.

## Gotchas that each cost a cycle to learn

1. **optimum 2.1 removed `OnnxConfigWithLoss`.** It is **vendored** at
   `export/onnx_config_with_loss.py`. Do not re-add the optimum import.
2. **`TasksManager` discovery returns empty** unless you first import
   `optimum.exporters.onnx.model_configs` (decorator registration) and pass
   `library_name="transformers"`. Both handled in `export/registry.py`.
3. **ORT renamed its weight-only quantizer** (`MatMul4BitsQuantizer` →
   `MatMulNBitsQuantizer`, module too, with no alias). Resolved at call time by
   `export/quantizer_compat.py`. Hard-coding either spelling re-breaks the other ORT line.
4. **ORT-training's `onnxblock` writes `temp.onnx(.data)` into the CWD** and shares that name across
   Blocks, so the second Block of a `generate_artifacts` run hits
   `FileExistsError: External data file exists`. Worked around by
   `artifacts/builder.py::_onnx_external_data_overwrite`.
5. **`DynamicQuantizeLinear` has no gradient builder, and correctly so.** A quantized *weight* is
   frozen and dequantizes to float (`DequantizeLinear`), so the backward pass crosses it untouched; a
   quantized *activation* cannot be differentiated at all. **Never let activation quantization onto
   the gradient path.**
6. **ORT rewrites `Gemm` → `MatMul` *before* quantizing** and matches `nodes_to_exclude` against the
   **rewritten** name `<gemm>_MatMul`. Excluding a `Gemm` by its own name silently does nothing —
   which makes the exclusion mechanism look broken. Handled in `onnx_dynamic_quantization`.
7. **`LayerNormalization` must carry its optional `Mean`/`InvStdDev` outputs** or no gradient graph
   can be built through it (`GradientBuilderBase::O … was false`, naming neither op nor node).
   `torch.onnx` emits only `Y`. Handled by `artifacts/graph_prep.py`. RMSNorm decoders are unaffected
   (`SimplifiedLayerNormalization` already has 2 outputs).
8. **Grep guards match docstrings.** The secret-guard and dispatch-guard patterns hit prose that
   *mentions* the banned pattern; keep docstrings from spelling out `os.environ['X']` / `x == "lora"`.
9. **`langchain-objectbox` 0.1.0** declares a stale `langchain-core<0.2.0`; overridden via
   `[tool.uv] override-dependencies`.
10. **flwr is deliberately OUT of the universal lock** — it downgrades protobuf/rich/typer and bumps
    mypy, breaking `make check`. Out-of-band like the ORT wheel: `pip install "flwr[simulation]"`.
11. **The Android unit-test classpath has no `kotlin-reflect`** — `::class.members` throws
    `KotlinReflectionNotSupportedError`. Use behavioural assertions instead.
12. **`uv run` re-syncs the environment to the lock BEFORE executing.** So `uv pip install
    transformers==4.46.2 && uv run --extra export mobiletransformers export …` silently exports under
    the locked **4.57.6**, and the pin you just set is gone. This invalidated the first attempt at the
    #37 regression isolation — both "different" packages came out byte-identical with
    `transformersVersion: 4.57.6`, which reads as "no difference found" rather than as a broken
    experiment. For a deliberately-pinned leg, call `.venv/bin/python -m mobiletransformers.cli.main …`
    directly. (Silver lining from the same run: the export is **deterministic** — identical inputs give
    a byte-identical `model.onnx`, which is what makes graph diffing a usable technique here.)
13. **The onnxruntime profile collision is reachable by hand.** `[tool.uv] conflicts` stops uv from
    resolving them together, but running `uv sync --extra export` between two `TRAIN=1` device packages
    leaves **both** `onnxruntime` and `onnxruntime-training` installed. The symptom names neither:
    `ImportError: cannot import name 'PropagateCastOpsStrategy' from 'onnxruntime.capi._pybind_state'`.
    Recover with `uv pip uninstall onnxruntime onnxruntime-training` then a clean profile sync.
14. **Present is not usable, for `onnxruntime.training`.** The **public** onnxruntime wheel ships an
    `onnxruntime/training/` directory, so `find_spec` finds it under the export profile even though
    importing it dies (gotcha 13). Probe by importing what you actually need, not by looking for the
    module — `export/pipeline.py::_training_available` was rewritten for exactly this.
15. **`android.content.Intent` and `org.json.JSONObject` are stubbed in plain JVM unit tests**, and
    `isReturnDefaultValues = false` makes the stubs **throw** rather than return null. Tests touching
    either need `@RunWith(RobolectricTestRunner::class)`; Robolectric is already on the test classpath.
16. **Kotlin block comments NEST.** Writing a path glob such as `inference` + slash + star inside a
    KDoc opens a *nested* comment that swallows the closing delimiter, and the compiler reports
    **"Unclosed comment" at the end of the file** — pointing nowhere near the cause. Name such files in
    prose. Cost two builds on 2026-08-14, the second one in `PristinePackageRule.kt` *after* the same
    gotcha had already been written down in `PostMergeNumericsTest.kt` — a note only helps in the file
    someone is editing, so it now lives in both.
17. **Gradle reports `UP-TO-DATE` as success.** A JVM suite that never executed produces the same
    `BUILD SUCCESSFUL` as one that passed. Use `--rerun-tasks`, and read counts out of
    `build/test-results/**/*.xml` rather than the console.
18. **`set -o pipefail` before piping a `make` target to `tail`.** Without it the pipeline reports the
    exit status of `tail`, so a failing gate reads as a pass. This masked a `ruff format` failure
    inside an otherwise-green `make check` on 2026-08-14.

## Never touch the venv while an export is running

`uv sync` **recreates** the shared `.venv`. Running `uv sync --frozen --group dev --python 3.10` (the
routine profile reset) while a `scripts/device_package.sh` export was in flight under 3.12 pulled
`certifi` out from under the running process, and the export died with

```
Could not find a suitable TLS CA certificate bundle, invalid path: .venv/.../certifi/cacert.pem
```

which looks like a network or certificate problem and is neither. This is the same shared-`.venv`
hazard as the profile-reset rule above, in its concurrent form: **one venv, one user at a time.** Do
not run host gates in parallel with an export or a training run.

## ORT engine separation (why two ONNX Runtimes coexist)

GenAI 0.14 needs stock ORT ≥1.26; the Native path needs the source-built training ORT 1.23. Both would
otherwise share the soname `libonnxruntime.so` and the linker dedups them, silently giving GenAI the
training lib (observed → SIGABRT).

**Resolution:** ship GenAI its own stock ORT 1.27 as `jniLibs/<abi>/libort_gen.so` with a **raw-patched
SONAME** (*not* `patchelf`, which corrupts `verneed`), and raw-patch the genai `.so`'s `dlopen` target
`libonnxruntime.so` → `libort_gen.so`. Training ORT keeps `libonnxruntime.so`. Safe because each ORT
exports ~3 symbols (hidden visibility) and genai resolves via `dlsym` on its own handle — no
interposition. **The distinct SONAME is essential.** Reproducible via
`spikes/genai_external_swap/setup_ort_separation.sh`. The genai AAR is **not** a Gradle dependency; the
patched `.so` ships from `jniLibs`, and the consumer APK proof confirms it is vendored.

## Device workflow

```bash
make device-package MODEL=<hf-id> [TRAIN=1] [RAG=1]   # export + reshape + adb push
make device-test                                       # connectedDebugAndroidTest
make device-rss                                        # the four-point RSS table
```

Hard-won provisioning facts, all encoded in `scripts/device_package.sh`:

- **Push to the test app's external files dir, never `/data/local/tmp`** — the latter is SELinux
  `shell_data_file`; the app domain cannot list it and cannot write it at all, which the merge and
  checkpoint legs require.
- **`adb push` leaves the tree `shell`-owned mode 0770**, so `listFiles()` returns null. The script
  `chmod -R 777` afterwards. (Production installs are app-owned and writable; the RAG store creates
  `embedding/database/` *inside* the package.)
- **`df -m` is rejected by Android 15's toybox**, and under `set -e` killed the script silently.
- `TRAIN=1` is what makes `TrainMergeGenerateTest` run rather than skip; `RAG=1` likewise for
  `RagDeviceTest`; the `mt_genai_spike` staging is what makes the #10 suite runnable.
- A C++ exception crossing JNI used to call `std::terminate` and kill the **entire instrumentation
  run**, so later tests never reported. Converted to a Java exception.

## The layer-identity problem (the structural cause of ~11 device defects)

One layer is spelled five ways, and every consumer re-derived it ad hoc:

| where | form |
| --- | --- |
| inference graph | `model.layers.0.self_attn.q_proj.MatMul.weight` |
| ORT checkpoint | `backbone.model.layers.0.self_attn.q_proj.base_layer.weight` |
| `peft_mapping` key | `base_model.model.model.layers.0.self_attn.q_proj` |
| handoff map key | `base_model.model.model.…q_proj.base_layer` |
| merger runtime | `backbone.model.layers.0.self_attn.q_proj` |

**Fixed structurally**: one shared C++ normalizer (`cpp/layer_name.h`, googletest-covered, replacing
nine open-coded prefix rewrites) plus the export-time assertion
`artifacts/checkpoint_names.py::verify_handoff_names_resolve`, which would have caught three of those
defects on the host instead of on a phone. **Any new consumer of a layer name goes through the
normalizer.**

## Magnitude-based checks cannot detect a permutation (2026-08-14)

**The most expensive defect found in this project to date: every merged weight was written TRANSPOSED,
for as long as on-device merging has existed, and four independent gates could not see it.**

`weight_merger.cpp` computes in checkpoint convention — `base_layer.weight` is a PyTorch `nn.Linear`
weight, `[out_features, in_features]`, and the merger ONNX graph declares its `weight` input and
`merged_weight` output the same way. The inference initializer it writes into is an ONNX `MatMul`
right-hand side, `[in_features, out_features]`. The result was written raw.

Proven by pulling the bytes off the device after a merge whose delta was **exactly zero**
(`adapter_B l2=0.000000`, scale 1.0, shapes correct): `max|written − original.T| = 1.9e-09`, i.e. the
written weight was the transpose to float round-trip precision. Model quality went from 4.65 nats to
15.45 on the same text — *above* the 10.80 uniform-prediction floor, so worse than random.

### Why every gate missed it — the transferable part

| gate | why it was blind |
| --- | --- |
| shape checks | `q_proj` is **square** (576×576), so no shape ever disagreed |
| external-data reads | `v_proj` is `[576,192]` vs `[192,576]` — **identical element count**, so the raw read succeeded |
| L2 norm / absmax probes | **transpose-invariant.** Both matched to 6 decimals. This fooled the investigating agent too, which cleared the base weight on an L2 comparison before realising it could not distinguish a matrix from its transpose |
| byte hashes (`TrainMergeGenerateTest`) | a transpose changes the hash, so "the bytes changed" passed while confirming nothing about correctness |
| `PostMergeNumericsTest` | its `PROBE` is a hand-written array of arbitrary token ids, so cross-entropy sits at the uniform floor (10.19 nats) **by construction**; comparing two near-uniform numbers with a 2× bound is close to unfalsifiable |
| `TrainMergeGenerateTest`'s output check | `assertTrue(after.isNotEmpty())` — passes on 48 consecutive newlines, which is exactly what the corrupted model emitted |

**The rule: a norm, a sum, a maximum, a byte count or a checksum cannot detect a permutation of
elements.** Any check on a weight-shaped tensor whose correctness depends on element *order* must
compare element order — against a reference, a known element, or the same computation through the
other graph. Prefer "does the model behave" (cross-entropy on **real tokenized text**) over "do the
numbers look plausible".

*All six gates above were repaired 2026-08-14.* The two that carried the assertion —
`PostMergeNumericsTest` (now `PROBE_TEXT`, real tokenized English, plus a large-delta test that
asserts memorised text scores **below** unrelated text) and `TrainMergeGenerateTest` (now rejects
blank output and degenerate single-character repetition) — were each verified able to fail before
being trusted. **A gate that cannot be shown to fail has not been tested, it has been assumed.**

**Second-order cause — a declared contract nobody implemented, now CLOSED.** `ObservedInit.transposed`
was a field that **nothing ever assigned**, so `transposePolicy` read `"no_transpose"` by omission
across all 60 entries *and in the test fixtures*. The contract meant to describe weight layout was
never implemented on the producing side, and the consumer faithfully honoured a value nobody computed.

The generalisable trap: **the fixtures agreed with the broken code because both were derived from the
same unimplemented default.** Synthetic test data written by the same author as the code under test
cannot detect an assumption they share. The fix is a test that reads a *real* artifact
(`test_the_derivation_agrees_with_a_real_exported_package`).

Fixed on both sides, differently and deliberately:

- **Producer** (`artifacts/handoff_map.py`): the dead field is deleted and replaced by
  `derive_transpose_policy()`, which *observes* orientation — the merger computes
  `base + scale·(B @ A)`, whose shape is `(B.rows, A.cols)`; if that equals the on-disk weight shape
  the conventions agree, if it equals the reverse the on-disk tensor is the transpose, and anything
  else is incoherent and refused. `resolve_package_transpose_policy()` settles the package from its
  non-square layers, because **a square layer cannot decide its own orientation** — which is exactly
  why `q_proj` hid the defect.
- **Consumer** (`weight_merger.cpp`): does **not** trust the field, and observes orientation itself
  from the non-square tensors at write time. This is intentional. Every package already exported and
  already on a device declares `no_transpose` wrongly, so a consumer that honoured the declaration
  would re-break them all. A field is only safe to honour once no artifact in the wild carries a wrong
  value for it.

The two implementations agree on real data (`transpose_for_inference=1` ⟺
`already_transposed_for_inference`), and a test fails if either drifts.

**Three fail-open paths in the FIX ITSELF, found by reviewing it rather than by a test**, and worth
listing because each would have re-introduced silent corruption in a narrower case:

1. *The orientation scan stopped at the first decidable layer.* A package that mixed conventions would
   have been half-corrupted according to iteration order. It now checks **every** non-square layer and
   **fails closed** on disagreement, naming the layer — mirroring
   `resolve_package_transpose_policy` on the Python side.
2. *An unobservable package fell back to a hardcoded `transpose = true`.* Reachable, not theoretical:
   an MHA model without GQA adapts `q_proj`/`v_proj` at identical shapes, so **every** adapted weight
   is square and orientation cannot be observed at all. A guess would corrupt half of such packages,
   and the shape check cannot catch it because a square tensor's transpose still matches the declared
   shape. It now falls back to the map's declaration and logs loudly that it is doing so.
3. *`write_raw_tensor_atomic`'s last two parameters were defaulted* to "skip the shape verification"
   and "transpose". A new call site would have got an unverified transpose **by omission** — the same
   shape as the unassigned field that caused the original defect. Both are now required.

The general point: a fix for a fail-open bug is itself a good place to look for fail-open defaults.

**Is there another field like it? Checked — no.** An AST sweep over all 267 dataclass fields in `src/`
found 39 never assigned by name; every one is either set positionally, or never read, or both. The
seven that cross a process boundary (`FederatedTensor.aggregation`, `SupportRow.*`, `TaskDiscovery.*`)
are all constructed positionally and, in the federated case, validated fail-closed against
`SUPPORTED_AGGREGATIONS`. **`ObservedInit.transposed` was the only instance of the pattern.** Recording
the negative so the sweep is not repeated from scratch; the query is "declared, honoured by a consumer,
never produced".

**A second route to the same defect was found and closed while testing the fix.** Adapter roles are
looked up by a *derived* training-graph initializer name, and a miss left the role merely
"undescribed" — tolerable individually, but if nothing matches, every entry loses its `adapterShapes`,
orientation becomes unobservable, and the package silently declares `no_transpose` again. Reached not
by an unassigned field this time but by a **name spelling drift** — the fifth spelling of one layer,
the same family as the layer-identity problem above. `from_peft_mapping` now raises when specs were
supplied and not one entry matched, naming the lookup that missed. Found because a test written from
the docstring used the wrong spelling and passed vacuously; the test now pins the behaviour.

## The recurring failure shape (worth internalising)

Every expensive defect in this project has the same form: **two halves each verified alone, and the
seam between them unverified.** Byte-level and structural assertions all passed while the numbers were
never looked at; engine parity compared Native with Native because GenAI silently fell back; the merge
"happened" without being correct; a checkpoint's size was read without reading its dtypes.

The countermeasures that actually worked, and should be the default for new work:

- assert **across** the seam, not on either side of it (identical tokens through both graphs, ordered
  callback sequences, before/after byte fingerprints);
- prefer a **relative/self-calibrating** assertion to an absolute threshold, which encodes one model
  and silently measures the wrong thing when the fixture changes;
- make the failing path **name the entity** (tensor, node, file) — several bugs cost a full
  export→push→run cycle purely because the error named nothing;
- when a claim is corrected, **record the correction next to the claim**, not only in a newer entry.

## Gate results (final)

| gate | result |
| --- | --- |
| **0.1 GenAI adoption** | **ADOPT.** External-data swap changes GenAI output on device and desktop; `OgaCreateModelWithInitializers` confirmed absent (fork-only); ORT coexistence resolved by engine separation. Cross-engine equivalence and RSS proven 2026-08-08 on a real package. |
| **0.2 memory mapping** | **PASS as re-specified 2026-08-09.** The original whole-process 15% predated #9's base/trainable split, which moved 91.8% of weight bytes outside what `WeightSessionCache` loads. Scoped to the bytes mmap owns: 50,524 kB saved of 54,374 kB eligible = **92.9% of the attainable maximum**. GenAI is n/a by construction. Non-blocking either way — mmap is default-off (`debug.mtf.mmap_weights`). |
| **0.3 export toolchain** | **PASS (Path A).** optimum-onnx for inference export, `onnxruntime.training` called directly for training artifacts, `OnnxConfigWithLoss` vendored. Source-built wheel proven alive. |
| **0.4 packaging/install** | **PASS** except the licence: `uv`-installable, reproducible wheel build, AAR published to mavenLocal and consumed by an external app (105 MB APK carrying all 7 native libs). |

## Post-merge numerical correctness — CLOSED 2026-08-10

The standing debt HANDOFF called *"the highest-value remaining conformance assertion"*. The export gates
parameter budget and train-vs-inference loss on the host; **nothing checked the numbers after an
on-device merge.** `TrainMergeGenerateTest` hashes the trainable `.bin` files and says in its own
comment that this is not a numerical test; `TrainConvergenceTest` reads the training loss, which never
touches the merged inference graph at all.

Closed by `PostMergeNumericsTest` — **PASSES on device 2026-08-10** (S21 FE / Android 15 / arm64-v8a):

```
before  InferenceMetrics(argmax=273,   maxLogit=23.93, sum=590021.4, sumSq=7431348.1, xent=10.193 nats)
after   InferenceMetrics(argmax=42995, maxLogit=18.08, sum= 39537.6, sumSq= 725345.8, xent=13.622 nats)
```

All four statistics move, so the merge changed the **computation**, not merely the bytes on disk. The
cross-entropy uses the same causal shift as `artifacts/train_inference_parity.py`, so the number is
comparable to the host gate rather than merely internally consistent.

**How the numbers reach Kotlin at all.** They could not before: `performInferenceStep` samples
internally and returns a token id, so logits never crossed JNI. `logits_metrics.h` (ORT-free, **8
googletests**, `make test-cpp` now 38) computes the reduction, `nativeInferenceMetrics` returns
`[argmax, maxLogit, sum, sumOfSquares, crossEntropyNats]`, and the probe resets the KV cache so it
advances nothing. Four statistics rather than one because a constant shift leaves `argmax` alone and a
redistribution leaves `sum` alone.

### Three real bugs this test found before it could pass

1. **The logits pointer was dangling — a use-after-free on every forward pass.**
   `generateWithKVCache` did `std::unique_ptr<Ort::Value> output = …; return output->GetTensorMutableData<float>();`
   with `output` a **local**, destroyed at the `return`. The sampling path survived by reading one row
   immediately, before the allocator reused the pages. Reading the whole `[seq, vocab]` block (8 ×
   49152 floats) segfaults reliably, which is how it surfaced. The tensor is now owned by
   `InferenceSessionCache::last_output` and lives until the next pass — the lifetime every caller
   already assumed.

2. **`releaseInferenceSession` dereferenced a null session.** `createInferenceSession` returns 0 on
   failure and `destroySession()` is still reached through the normal release path, so
   `session_cache->inference_session` read offset 0x8 of a null pointer — `SIGSEGV, fault addr 0x8`,
   which kills the **entire instrumentation run** rather than failing one test. Same class as the C++
   exception that used to cross JNI and call `std::terminate`. Guarded.

3. **`_training_available()` answered the wrong question.** It used `find_spec("onnxruntime.training")`,
   and the **public** onnxruntime wheel ships an `onnxruntime/training/` directory too — so under the
   export profile it reported available, `_select_stages` selected a training stage that profile cannot
   build, and the export died inside `artifacts/builder.py` with
   `ImportError: cannot import name 'PropagateCastOpsStrategy'` — a traceback naming a symbol instead of
   the profile. It now performs the real import (the same symbols `builder.py` imports at module
   scope), because *present* and *usable* are different things here.

> ⚠️ **Operational note.** Running `uv sync --extra export` between `TRAIN=1` device packages left
> **both** `onnxruntime 1.27.0` and `onnxruntime-training 1.23.0+cpu` installed — the exact collision
> `[tool.uv] conflicts` exists to prevent, reached by hand rather than by uv. Symptom is bug 3's
> `ImportError`. Recover with `uv pip uninstall onnxruntime onnxruntime-training` then a clean profile
> sync.

## Device suite (S21 FE `SM-G990B` / Android 15 / arm64-v8a)

> **The suite's result depends on WHICH package is installed, and the device cache holds one.** Two
> recorded runs, both 0 failures:
>
> | package on device | result |
> | --- | --- |
> | `HuggingFaceTB/SmolLM2-135M-Instruct` `TRAIN=1 RAG=1` (decoder) | **15 / 15 pass**, 798.0 s — the table below |
> | `sentence-transformers/all-MiniLM-L6-v2` `TASK=text-classification TRAIN=1` (encoder) | **19 tests, 0 failures, 16 skipped** — 2026-08-10 |
>
> The encoder run's 16 skips are the generation suites declining a graph with no token loop
> (`DeviceModel.requireDecoder`), not missing coverage; the three that run are
> `EncoderTrainStepDeviceTest`, `ObjectBoxParityTest` and `ExampleInstrumentedTest`. The two
> `FederatedRoundDeviceTest` phases skip in both unless driven by
> `scripts/federated_round_device.sh` (they need `-PmtFederationEnabled=true` and a pulled/pushed
> record). **As of 2026-08-10 the device holds the ENCODER package.**

**2026-08-10: 15 / 15 pass, 0 failures, 798.0 s.** Freshly exported and pushed `TRAIN=1 RAG=1`
`HuggingFaceTB/SmolLM2-135M-Instruct` package, on the fix for the `transformers` regression.

| test | result | s |
| --- | --- | --- |
| `ConversationResetTest.threeSequentialPromptsEachEmitTheRequestedTokens` | PASS | 15.4 |
| `ConversationResetTest.aFreshSessionDoesNotInheritThePreviousConversation` | PASS | 24.7 |
| `DualEngineParityTest.bothEnginesEmitTheSameOrderedCallbackSequence` | PASS | 16.1 |
| `DualEngineParityTest.nativeAndGenaiAgreeOnGreedyFirstToken` | PASS | 15.4 |
| `ExampleInstrumentedTest.useAppContext` | PASS | 0.0 |
| `FacadeLoadGenerateTest.fromPretrainedGeneratesAndReportsInferenceFeature` | PASS | 14.0 |
| `GenAISpikeTest.genaiResolvesExternalDataAndSwapIsObserved` | PASS | 11.8 |
| `MemoryRssTest.nativeFourPointTable` | PASS | 16.3 |
| `MemoryRssTest.genAiFourPointTable` | PASS | 4.4 |
| `ObjectBoxParityTest.objectBoxRankingAndScoresMatchCosineReference` | PASS | 0.1 |
| `RagDeviceTest.ingestThenGroundedGenerate` | PASS | 18.1 |
| `ScheduledTrainingDeviceTest.chunkedTrainingResumesAcrossTheChunkBoundary` | PASS | 138.9 |
| `TrainConvergenceTest.trainingStartsFromPretrainedWeightsNotRandomOnes` | PASS | 131.2 |
| `TrainConvergenceTest.lossFallsOverTrainingSoTheMergeCarriesRealLearning` | PASS | 223.0 |
| `TrainMergeGenerateTest.trainMergeGenerateDivergesFromBaseline` | PASS | 147.6 |

`ConversationResetTest` is 15 tests rather than 14 because it gained a second case; both are now
non-vacuous (the old one asserted `tokenCount >= 0`, true of every possible outcome).

**`PostMergeNumericsTest` is NOT in this run** — it was written after the package was pushed. It needs
its own run on a pristine package.

### Superseded: 2026-08-09, 13 / 14 pass, 1 failure, 693.7 s

Kept because the failure is the regression analysed above, and the run is its evidence.

| test | result | s |
| --- | --- | --- |
| `ConversationResetTest.twoSequentialPromptsBothComplete` | **FAIL** | 16.6 |
| `DualEngineParityTest.bothEnginesEmitTheSameOrderedCallbackSequence` | PASS | 21.1 |
| `DualEngineParityTest.nativeAndGenaiAgreeOnGreedyFirstToken` | PASS | 20.6 |
| `ExampleInstrumentedTest.useAppContext` | PASS | 0.0 |
| `FacadeLoadGenerateTest.fromPretrainedGeneratesAndReportsInferenceFeature` | PASS | 16.2 |
| `GenAISpikeTest.genaiResolvesExternalDataAndSwapIsObserved` | PASS | 9.0 |
| `MemoryRssTest.nativeFourPointTable` | PASS | 16.3 |
| `MemoryRssTest.genAiFourPointTable` | PASS | 4.4 |
| `ObjectBoxParityTest.objectBoxRankingAndScoresMatchCosineReference` | PASS | 0.1 |
| `RagDeviceTest.ingestThenGroundedGenerate` | PASS | 18.3 |
| `ScheduledTrainingDeviceTest.chunkedTrainingResumesAcrossTheChunkBoundary` | PASS | 139.2 |
| `TrainConvergenceTest.trainingStartsFromPretrainedWeightsNotRandomOnes` | PASS | 131.6 |
| `TrainConvergenceTest.lossFallsOverTrainingSoTheMergeCarriesRealLearning` | PASS | 199.0 |
| `TrainMergeGenerateTest.trainMergeGenerateDivergesFromBaseline` | PASS | 100.8 |

**The one failure is a real regression, and it is the `transformers` fork's** — see the CORRECTION box
in "The #37 architecture gate". It ran first, on a pristine package, so it is not the mutation hazard
below.

**What this run also settled:** an earlier run the same day showed **two** failures
(`TrainConvergence(pretrained)` NaN, `TrainMergeGenerate` "merge wrote no new weights"). Both **pass
here**, confirming they were package mutation by the then-non-hermetic `ScheduledTrainingDeviceTest`,
not regressions — and confirming the hermetic fix works.

### The package-mutation hazard

The training suites **rewrite the package in place** and require a pristine one.
`TrainMergeGenerateTest` says so in its own failure message: *"the merge rewrites these files IN PLACE,
so a package that has already been merged re-merges to identical bytes. Re-push a pristine package
(`make device-package`) before re-running this suite."*

`ScheduledTrainingDeviceTest` trains too and sorts **before** both training suites, so its first
version turned two unrelated suites red. It now stashes and restores `train/checkpoint` +
`training_state.json` and deletes its own fixture. **A new device test that trains must do the same, or
the suite must be run on a freshly pushed package.**

## Migration record

All seven legacy roots (`trainer/`, `artifact/`, `inference/`, `tools/`, `peft_models/`, `database/`,
`evaluation/`) are gone; ~17.5k lines now live under `src/mobiletransformers/` and are linted and
type-checked for the first time (they were excluded from both gates throughout their life, which is
how they moved without ruff or mypy ever seeing them). Both migration allow-lists
(`test_no_src_to_legacy_imports.py`, the dispatch guard) are **empty**. The wheel is self-contained —
verified by installing into a clean venv and importing from outside the checkout.

# Tier 1 — 02_code_plans audit

Verified read-only against the tree on branch `restructure` (54e0a8e). Python gate re-run locally:
`.venv/bin/python -m pytest tests/hub tests/export tests/cli tests/support tests/adapter tests/unit/test_manifest.py -q`
→ **91 passed, 8 skipped** (skips = export profile / torch+safetensors absent). Android JVM tests counted
statically: **125 `@Test`** in `MobileTransformers/src/test` (40 facade, 14 hub) — matches the HANDOFF claim.
No Gradle run (per instructions), so "compiles" claims are unverified here.

## Summary table

| # | Plan | Claimed | Verified | % | Verdict |
| --- | --- | --- | --- | --- | --- |
| 14 | `03_hub_model_package_format.md` | `[x]` done 2026-07-14 | Module + fixture + parity oracle + 6 tests all real | **88%** | Substantially done; DoD's `docs/HUB_PACKAGE_FORMAT.md` absent, `default/` alias & `etag` never implemented |
| 15 | `05_one_command_export_cli.md` *(checkpoint)* | `[x]` done + real `_full_export` | Orchestrator + CLIs + card real; several declared outputs never emitted | **70%** | Over-claimed. `--validate` missing entirely, `--config` accepted-and-dropped, 3 declared artifacts unemitted |
| 19 | `01_hf_style_kotlin_facade.md` *(checkpoint)* | `[ ]` code-complete, device leg open | Facade/PEFT/exceptions/callbacks real, 40 JVM tests | **78%** | Solid; manifest validation skipped on load, `applyPeft` rank/alpha silently dropped, sample app unmigrated |
| 20 | `02_optimum_support_matrix.md` | `[x]` done 2026-07-14 | Statuses/models/matrix/CLI + 9 tests real | **75%** | Two normative violations (duplicated task priority; candidate list not from `config.yml`); one status is a proxy |
| 21 | `04_hub_pull_and_cache_flow.md` *(checkpoint)* | `[x]` done, Android downloader landed | Python legs strong; Android network half real but mis-wired | **70%** | Kotlin `VariantSelector` never called; WorkManager worker never enqueued; installer deletes live cache before rename |
| 22 | `06_adapter_pushback.md` | `[x]` done, "genuinely done" | Gate/card/CLI + Kotlin uploader real, 20 tests | **80%** | Mode-1 never writes `adapter_model.safetensors` in the CLI; Kotlin gate diverges from Python gate |

**Overall Tier-1 (02_code_plans) estimate: ~77%** — Python surface ≈ 80%, Android surface ≈ 72%. Every plan
has a working core; the shortfall is concentrated in (i) declared-but-unemitted package artifacts, (ii) two
Kotlin components that exist but are not on the live code path, and (iii) all three checkpoint workflows
being either fixture-substituted or device-gated.

---

## Checkpoint-workflow status (#15 / #19 / #21)

### #15 — export E2E (Python)
**Exists, but the "export" leg is substituted.**
- `tests/export/test_pipeline.py:87` `test_assemble_package_produces_valid_13_package` is the automated
  checkpoint. It feeds `assemble_package` the **committed fixture's already-built variant subtrees**
  (`tests/export/test_pipeline.py:89-95`) and asserts `MobileTransformersManifest.validate()` passes. So it
  proves reshape + manifest emit + #13 validation — **not** export.
- `tests/export/test_full_export_orchestration.py:25` exercises `_full_export` for real, but with **injected
  fake builders writing 1-byte `model.onnx`** (`:31`). Good coverage of stage selection / honest-features /
  genai gating / fail-closed; zero coverage of the actual optimum or ORT-training stages.
- What the plan's checkpoint text demands but nothing asserts: ORT training artifacts
  (`training_model.onnx`/`eval_model.onnx`/`optimizer_model.onnx`/`checkpoint/`), on-device merger graphs,
  `checksums.json`, and a **non-empty** `weight_handoff_map.json`. The fixture's `train/` dirs contain only
  `training_config.json` + `weight_handoff_map.json` (see `tests/fixtures/tiny_package/variants/*/train/`),
  so the "validated device-ready package" the checkpoint asserts is not train-capable.
- Training stage: `tests/integration/test_training_stage_smoke.py:28` covers **only step 2**
  (`gen_artifacts`), is `importorskip`-gated on the ort-training profile, and depends on a generated
  fixture (`tests/fixtures/tiny_trainable.onnx`) that is not committed.
- The real SmolLM2-135M run is an on-box claim in `agent_docs/HANDOFF.md` (session 2026-07-15); nothing in
  the tree records or reproduces it.
**Faked/injected:** stage builders, ONNX bytes, tokenizer. **Real:** manifest build, sha256/fileSizes,
per-variant checksums, #13 validation.

### #19 — train→merge→generate (Android, device)
**Test exists; never runs off-device.**
- `MobileTransformers/src/androidTest/java/.../TrainMergeGenerateTest.kt:23` does baseline `generate` →
  `train(maxSteps=1, mergeAtEnd=true)` → `merge()` → `generate` and asserts `after != baseline` (`:43`).
  It self-skips via `DeviceModel.requireCacheRoot()` / `assumeTrue(hasTraining(...))` (`:24-26`), so it needs
  a pushed train-capable package. Provisioning exists (`make device-package TRAIN=1`, `scripts/device_package.sh`).
- **Divergence from the plan's workflow:** the plan's sequence is `fromPretrained → applyPeft → train →
  merge → generate`; the test never calls `applyPeft`.
- Host substitute is `FacadeDelegationTest.kt:33` — a **hand-written fake `ModelSession`** recording call
  names. It proves delegation, not behavior.
**Faked:** the entire session in the host test. **Real (but unrun):** the device test.

### #21 — pull→load
**Python half real and automated; Android half only partially wired.**
- `tests/hub/test_pull.py:43/58/72` cover pull(inference-only) → correct file subset, sha256-mismatch
  fail-with-path, and install → exact `LLMRepository` layout with tokenizer flattening. The downloader is
  **injected** (`_fake_downloader`, `:17`), copying from the committed fixture — real `snapshot_download` is
  never exercised.
- Android: `PackageDownloaderTest.kt` (MockWebServer, sha256 verify + retry) and `DownloadPlannerTest.kt`
  (glob expansion) are genuine. **No test covers `HubDownloader.downloadAndInstall` end-to-end**, and the
  load leg (`FacadeLoadGenerateTest.kt`) is device-only.
- The self-check "Is variant selection identical Python↔Kotlin (deterministic)?" is marked `[x]`. It is
  **not**: see #21 gaps below — the Kotlin selector is not on the download path and the #21 policy layer was
  never ported.

---

## Per-plan findings

### #14 — Hub model package format (`02_code_plans/03`)

**Required:** `hub/package_format.py` with `SCHEMA_VERSION`, `REQUIRED_TOP_LEVEL_FILES`, `FEATURE_GROUPS`,
`VARIANT_SUBDIRS`, `sanitize_repo_id` (canonical, Kotlin-byte-identical); `build_manifest()`; the full
manifest field list; per-variant `checksums.json`; `optimum/{export_report,supported_tasks,optimum_config}.json`;
mapping of builder outputs into the variant tree incl. `chat_template.jinja` split; a tiny fixture that
validates clean against #13; `default/` alias policy documented; **`docs/HUB_PACKAGE_FORMAT.md`**.

**Verified present:**
- `src/mobiletransformers/hub/package_format.py:22-34` (constants), `:39` `sanitize_repo_id` implementing the
  exact 3-step algorithm, `:108` `build_manifest`, `:198` `write_manifest`, `:206` `write_variant_checksums`.
- Manifest emit covers every plan field except `etag` (`src/mobiletransformers/hub/package_format.py:164-195`).
- Fixture `tests/fixtures/tiny_package/` (32 files, two variants, one genai-capable), validates clean:
  `tests/unit/test_manifest.py:24` (Python) and `.../packages/PackagesTest.kt:59` (Kotlin `ManifestValidator`).
- Parity oracle `tests/fixtures/sanitize_repo_id_cases.json` asserted from **both** sides:
  `tests/hub/test_package_format.py:37` and `.../packages/PackagesTest.kt:33`; Kotlin impl at
  `.../packages/PackageFormat.kt:19`.
- Integrity round-trip / requiredFiles / downloadPlan-resolves / dual-engine sanity / F1 schema gate:
  `tests/hub/test_package_format.py:44,52,58,70,100,116`.

**Gaps:**
- **(a) `docs/HUB_PACKAGE_FORMAT.md` does not exist** — the plan's Definition of done names it explicitly
  ("mirrored in `docs/HUB_PACKAGE_FORMAT.md`"). `docs/MODEL_FORMAT.md` covers most of the same ground
  (layout, versioning, `downloadPlan`, `sanitize_repo_id`) but is a different page owned by #31.
- **(a) `default/` aliasing policy** (plan step 7) is neither implemented nor documented — `package_format.py`
  never mentions `default/`, `build_manifest` never emits or aliases it.
- **(a) `etag` map** never populated (no Hub HEAD path anywhere). Plan marks it optional, so low severity.
- **(a) `optimum/optimum_config.json`**, `licenses/`, `shared/config.json`, `shared/generation_config.json`,
  `shared/chat_template.jinja` exist **only in the hand-built fixture** — the real export
  (`export/pipeline.py:199-207`) emits just `export_report.json` + `supported_tasks.json`, and
  `assemble_package` copies only `shared/tokenizer`. The `core` download group
  (`hub/package_format.py:65-71`) therefore names files a real package does not contain.
- **(c) Per-variant `weightHandoff` is hard-coded** at `hub/package_format.py:143` to
  `variants/<id>/inference/weight_handoff_map.json` regardless of whether the variant has an `inference`
  subtree — a train-only or embedding-only variant would emit an unresolvable pointer and fail #13.
- **(d)** Fixture relocated `agent_docs/fixtures/tiny_package/` → `tests/fixtures/tiny_package/`. Sensible;
  the Kotlin test walks up to find it (`PackagesTest.kt:18-27`).

**Drift / doubtful claims:** none material. The self-check claims all check out. The `[x]` box is defensible
apart from the missing DoD doc.

---

### #15 — One-command export CLI (`02_code_plans/05`) · checkpoint

**Required:** `export`/`push` subcommands on the existing dispatcher; `export/pipeline.py::export_package`;
`export/model_card.py::render_model_card`; CLI flags `--model --task --peft --rank --quant --output
--variant --include-rag --embedding-model --validate`; config.yml overlay (step 1); 11 orchestration steps
incl. merger graphs, tokenizer/chat-template split, `--validate` desktop smoke; outputs incl.
`train/trainable_parameters.json` and `optimum/{export_report,supported_tasks,optimum_config}.json`;
`push --package --repo-id [--private] [--revision]` gated on the #13 validator.

**Verified present:**
- `src/mobiletransformers/export/pipeline.py:33` `parse_peft` (incl. `mars-opt0..4` validation), `:55`
  `quant_spec`, `:93` `plan_export`, `:142` `manifest_skeleton`, `:165` `assemble_package`, `:230`
  `export_package`, `:632` `_full_export` (stage-gated, injectable builders).
- Real inference stage `:383` `_build_inference_stage` → `export_inference` (#7) + empty `HandoffMap` (`:409`)
  + tokenizer stage (`:435`) + `_emit_genai_config` (`:454`).
- Real training stage `:526` `_build_training_stage` → `optimum_hf_export` → `gen_artifacts` →
  `export_inference_package` (which emits the trainable split, `frozen_base.onnx.data`, per-tensor `.bin`
  + `.sha256`, and the merger graphs — `inference/export_inference_package.py:186,289`).
- `_effective_features` (`:345`) never claims a subtree absent from disk; engine honesty at `:687`.
- CLI: `src/mobiletransformers/cli/export.py:15` (+`--genai`, `--stages`), `cli/push.py:20` with the #13
  validator gate at `:35` and injectable `uploader` at `:30`; both registered in `cli/main.py:25`.
- `export/model_card.py:8` renders base model, both licenses, version pins, android runtime, variant table.
- Tests: `tests/export/test_pipeline.py` (10), `tests/export/test_full_export_orchestration.py` (5),
  `tests/cli/test_cli.py` (5).

**Gaps:**
- **(a) `--validate` does not exist.** No flag in `cli/export.py:15-36`, no smoke in `pipeline.py`, no
  `export_report.json` pass/fail record. Plan step 11, the Manual test, and the Definition of done ("with
  `--validate`, that the recorded desktop smoke passed") are all unmet. The IMPLEMENTATION_ORDER `[x]` for
  #15 does not disclose this.
- **(a) Config overlay silently dropped.** `cli/export.py:34` declares `--config` but `run()` (`:40-82`)
  never reads it, and `plan_export`/`export_package` never touch `config/config.yml` or the #4 layering.
  Plan step 1 + the "Config-overlay test" unit test are both absent (`tests/cli/test_cli.py` has no such test).
  `docs/EXPORT.md:27` documents the flag as working — **doc overclaims code**.
- **(a) `optimum/optimum_config.json` never emitted** — `pipeline.py:202-207` writes only `export_report.json`
  and `supported_tasks.json`. Plan step 10 + DoD list three.
- **(a) `train/trainable_parameters.json` never emitted** anywhere in `src/`, `artifact/`, `inference/`
  (grep clean). Explicit output-contract item in the plan's Outputs section.
- **(a) `shared/chat_template.jinja` never produced by export.** `_populate_tokenizer_stage`
  (`pipeline.py:435-451`) copies tokenizer files but does not split `tokenizer.chat_template` out
  (plan step 7 / #14 step 3). Only the *consumer* side exists (`hub/pull.py:147`).
- **(b) Merger graphs land in the wrong directory.** `export_inference_package` writes them into the
  **inference** dir (`inference/export_inference_package.py:289`, called with `output_dir=inference_dir` at
  `pipeline.py:602-604`), while #14's layout and #21's cache layout both specify `train/merger_*.onnx`.
  Functionally consistent (the handoff map's `mergerModels` records the name), but it contradicts two
  normative layout blocks.
- **(a) `--include-rag` is a hard failure.** `_build_embedding_stage` (`pipeline.py:623-629`) unconditionally
  raises `ExportError("embedding stage is staged for the RAG phase (#26/#27)")`, yet `plan_export:115` adds
  `rag` to features. Requesting RAG aborts the whole export. Categorize **(d)** if #26/#27 own it, but the
  plan lists it as step 8 of #15.
- **(c) `push` flag naming drifts:** `--repo` not `--repo-id` (`cli/push.py:23`), and `--revision` is absent
  (plan and tier doc both show `--repo-id`/`--revision`). `create_repo` is called but `upload_folder` is
  invoked without `revision`/`commit_message` (`:52`).
- **(c)** Inference-only packages carry `model.onnx_data` (`export/normalize.py:67`) rather than the canonical
  `frozen_base.onnx.data` — only the training stage produces the canonical split
  (`inference/export_inference_package.py:49`). Defensible (all-frozen, empty handoff map) but it means a
  #15 inference-only package does not match the #14 "FLAT canonical layout" verbatim.

**Drift / doubtful claims:** IMPLEMENTATION_ORDER's #15 self-check line "Does one command go HF model →
validated device-ready package?" is `[x]`; a single command cannot, because the training stage requires a
**second invocation under a different uv profile** (`pipeline.py:549-557` explicitly errors if the inference
package isn't already there). That is an accurate engineering choice, but it is not "one command", and the
Definition of done's phrasing is not met literally.

---

### #19 — HF-style Kotlin facade (`02_code_plans/01`) · checkpoint

**Required:** `applyPeft`/`pushAdapter` + callback params on the model handle; sealed `PeftConfig`; the full
`MobileTransformersException` hierarchy; public→`ORT*` mapping per the plan's tables; feature detection +
manifest validation in `fromPretrained`; engine gating; `compat/LegacyAliases.kt`; sample-app migration
(MainActivity + 3 ViewModels); unit tests for config/PEFT/feature-gate/friendly errors; device workflow.

**Verified present:**
- `.../MobileTransformers.kt:31` `fromPretrained` with the exact FINAL signature; feature gate at `:82-84`
  (construction-time `FeatureNotInstalledException`); GenAI config gate at `:92-99`
  (`EngineUnavailableException`); `ModelNotInstalledException` at `:63/68`.
- `.../MobileTransformerModel.kt:41` `applyPeft`, `:43` `train`, `:49` `merge`, `:51` `generate`, `:57`
  `retrieve`, `:79` `pushAdapter`, `:82` `close` — no `ORT*`/repository type in any signature.
- `.../config/PeftConfig.kt` sealed `Lora/MarsOpt0/MarsOpt1/MarsQuantized`;
  `.../internal/config/PeftSupport.kt:18` taxonomy, `:31` package taxonomy parse (tolerates `train_config`
  wrapper), `:40` `validate` → `PeftMismatchException`.
- `.../MobileTransformersException.kt:20-58` — all six exception types with path-naming messages.
- `.../internal/config/ConfigMappers.kt:44` `TrainConfig.toOrt`, `:83` `GenerationConfig.toOrt(engine,
  mergedLoaded)` (engine→`type`, merge-state→`loadMergedWeights`), `:96` `RagConfig.toOrt`.
- `.../internal/runtime/RepositoryBackedModelSession.kt:54` — one `LLMRepository` + three repositories,
  1:1 callback adapters (`:93-125`, `:155-176`), `mergedWeightsLoaded` threading (`:68/115/129/143`).
- 40 facade JVM tests: `ConfigMapperTest`, `ConfigMappingDeltaTest` (engine→type, merge→loadMergedWeights,
  defaults, dataset), `ExceptionMessageTest` (6, all message-content assertions real),
  `FacadeDelegationTest`, `FeatureAndVariantTest`, `PeftMappingTest` (7), `RagConfigMapperTest`,
  `SamplingMappingTest`.
- Device checkpoint `androidTest/.../TrainMergeGenerateTest.kt` (see checkpoint section).

**Gaps:**
- **(a) The manifest validator is never run on an already-installed package.** Plan step 1: "Run the #17
  manifest validator." `MobileTransformers.kt:72-77` only asks `LLMRepository` whether configs exist.
  Grep confirms `ManifestValidator` has exactly zero callers outside `packages/` and its own test. A
  corrupt/incompatible manifest in the cache loads silently.
- **(a) `applyPeft` rank/alpha overrides are silently dropped.**
  `RepositoryBackedModelSession.kt:71` declares `appliedPeft` with the comment "the validated PEFT selection
  to apply on the next train() (rank/alpha overrides)"; it is assigned at `:83` and **never read** —
  `train()` (`:127`) builds `ORTTrainingConfig` from `TrainConfig` only. `PeftSupport.validate`
  (`internal/config/PeftSupport.kt:40-49`) compares method + optimization level and ignores rank/alpha
  entirely, so the plan's "Rank/alpha overrides that exceed the exported adapter shape are rejected the same
  way" is unimplemented. This is a dead field, not a stub — nothing fails closed.
- **(c) Public config surfaces are narrower than the plan's normative tables.**
  `config/PublicConfigs.kt:41-55` `TrainConfig` uses a flat `SchedulerType` enum + `minLearningRate`/
  `warmupSteps` instead of the plan's `sealed class Scheduler { Linear(startFactor,endFactor); Cosine(...) }`
  (so `startFactor`/`endFactor` are unreachable from the public API). `DatasetConfig` (`:91-96`) has 4
  fields; the plan specifies 8 — `dropLongSamples`→`removeLongSamples`, `split`→`datasetSplit`,
  `shuffle`→`datasetShuffle`, `testRatio` are missing, and `DatasetConfig.toOrt`
  (`ConfigMappers.kt:70-76`) maps only the 4. `GenerationConfig` (`:58-64`) has no `trackMetrics` (plan's
  mapping table lists it). Per-call `ORTRagArguments` overrides (plan's RagConfig note) are not exposed.
- **(c) `MobileTransformersException` is `open`, not `sealed`** (`MobileTransformersException.kt:18-21`).
  Deliberate and documented in-file (cross-package subclass `hub.AdapterUploadDisabledException`), but it
  contradicts the plan's normative `sealed class`.
- **(d) Sample-app migration not done.** `app/src/main/java/.../app/MainActivity.kt:73-76` still constructs
  `LLMRepository`/`InferenceRepository`/`TrainingRepository`/`RagRepository` directly; all three ViewModels
  still take raw repositories (`viewmodels/ConfigurationViewModel.kt:11`,
  `viewmodels/InferenceViewModel.kt:55`, `viewmodels/TrainingViewModel.kt:28`). Documented as deferred in
  IMPLEMENTATION_ORDER's #19 note. The plan's DoD only requires the app to *compile*, so this is a
  scope-deferral rather than a DoD violation.
- **(d) `compat/LegacyAliases.kt` intentionally skipped** — justified (#16 removed the `ortmobile` brand
  entirely, so there are no consumers). Documented.
- **(c) Device workflow unrun** and does not exercise `applyPeft`.

**Drift / doubtful claims:** the #19 self-check "`applyPeft`/`train`/`merge`/`generate`/`retrieve` map
cleanly onto existing repositories" is `[x]` and true for the five verbs; the parenthetical "applyPeft
validates via PeftSupport" is true but understates that only method+level are validated. The claimed tests
(`PeftMappingTest`/`ConfigMappingDeltaTest`/`ExceptionMessageTest`/`FacadeDelegationTest`) all exist with the
described assertions.

---

### #20 — Optimum support matrix (`02_code_plans/02`)

**Required:** `support/{statuses,models,matrix}.py`; `_select_task` as a **thin alias of
`export/registry.choose_task`** (explicitly "do NOT copy the priority list"); candidate input from
`config.yml SUPPORT_MATRIX.candidates` or `config/support_candidates.yml`; six inherited statuses with
`mobile_package_exportable` decided by an export-report read or a normalization dry-run; probes keyed by
`(modelId, variant)`; `support-matrix` CLI; unit tests incl. an F1 schema-version test and a version-capture
test asserting `transformersCeiling`.

**Verified present:**
- `src/mobiletransformers/support/statuses.py:13` `SupportStatus`, `:23` `STATUS_ORDER`, `:26`
  `USER_FACING_STATUSES`, `:35` `apply_inheritance`, `:53` `first_blocked`.
- `src/mobiletransformers/support/models.py:22` `CandidateEntry.to_row` (all required camelCase keys), `:54`
  `SupportMatrix.to_dict` with `schemaVersion`/`minReaderVersion`/`toolchain.transformersCeiling`/
  `statusOrder`/`userFacingStatuses`, `:74` `filtered_docs_dict` (F6).
- `src/mobiletransformers/support/matrix.py:73` `detect_candidate` (injectable `config_loader`/`tasks_lookup`/
  `versions`), `:107` `evaluate_statuses`, `:137` `ingest_probes`, `:149` `build_matrix`, `:182/189` writers.
- MARS check goes through the #6 architecture registry: `matrix.py:63-70` `resolve_architecture(...).target_modules`.
- `src/mobiletransformers/cli/support_matrix.py:23` with `--candidates/--probes/--out/--docs/--md`.
- Tests `tests/support/test_support_matrix.py` (9) + `tests/support/test_render.py` (drift guard for #31).

**Gaps:**
- **(a) Normative violation: the task-priority list is duplicated.** `matrix.py:24` declares its own
  `_TASK_PRIORITY` with the comment "kept in sync with export/registry.choose_task", and `:54` `_select_task`
  reimplements the selection. The plan's Interactions section says verbatim: "`_select_task` is a thin
  import/alias of `mobiletransformers/export/registry.py::choose_task` (#7) — do NOT copy the priority list
  into this module." `choose_task` exists at `export/registry.py:107` and is not imported here.
- **(a) Candidate input never wired to config.** No `SUPPORT_MATRIX` block in `config.yml` and no
  `config/support_candidates.yml` (grep clean across `config/`, `config.yml`, `src/`, `tests/`). Instead
  `cli/support_matrix.py:16-20` hard-codes three model ids, with a comment claiming it "mirrors config.yml
  SUPPORT_MATRIX" — a **false claim in the code**. Plan implementation step 1 unimplemented.
- **(b) `mobile_package_exportable` is a proxy, not a measurement.** `matrix.py:112` sets it to
  `entry.selected_task is not None` — identical to `optimum_exportable`. Plan step 3 requires either
  trusting an existing `optimum/export_report.json` or running the normalization dry-run (tokenizer load,
  generation-config availability, manifest writability). The status can therefore never be the earliest
  blocker, and the plan's inheritance example (`mobile_package_exportable:false`) is unreachable in practice.
- **(b) `train_artifacts_exportable` only checks MARS target modules** (`matrix.py:113`); ORT
  training-artifact availability (the other half of the plan's evidence source) is never probed.
- **(c) Probes are keyed by `modelId` only** (`matrix.py:137-146`, `:177` `probes.get(entry.model_id)`), not
  `(modelId, variant)`. Consequence: the plan's blocker string "no android inference probe recorded for
  variant cpu-int4" cannot be produced; the emitted blocker is the variant-free
  `f"no android probe recorded for {blocked}"` (`:132`).
- **(b) Missing tests named in the plan:** no F1 schema-version/`check_compat` test for the matrix envelope
  (the plan requires one; `models.py` emits the versions but no reader gates them anywhere), and no
  version-capture test asserting `transformersCeiling == "<4.58"` in the envelope. Also `tests/support/`
  contains no `test_inheritance.py` (the plan's filename) — the assertions live in `test_support_matrix.py`,
  which is fine.

**Drift / doubtful claims:** the `[x]` self-check "Is `model_support_matrix.json` generated truth with
inherited statuses + earliest-blocker attribution?" holds. The `[x]` on the box overall is optimistic given
the two normative violations and the proxy status.

---

### #21 — Hub pull & cache flow (`02_code_plans/04`) · checkpoint

**Required:** Python `pull_package`/`install_package` + `select_variant` + `pull`/`install-package` CLI;
Android `HubResolver`/`ManifestClient`/`VariantSelector`/`PackageDownloadWorker`/`PackageInstaller`;
deterministic selection algorithm **identical in Python and Kotlin** (7 steps incl. soft memory filter,
quantization preference, size tie-break, fallback-to-default with warning, 0.9× storage ceiling); HEAD
size/etag cross-check; Range resume; sha256 with named offender; manifest schema validation **before**
exposing to `LLMRepository`; atomic rename that never touches the live cache dir; WorkManager scheduling.

**Verified present (Python):**
- `src/mobiletransformers/hub/pull.py:53` `pull_package` — manifest-first (`:74-84`), variant select (`:88`),
  `downloadPlan`-derived `allow_patterns` (`:44`, `:91`), sha256 verify naming the offending path (`:100-105`);
  `:109` `install_package` — selected-variant handoff pre-check (`:128-130`), variant subtree copy (`:139-142`),
  `shared/tokenizer` + `chat_template.jinja` flattening (`:144-149`), `.partial` → `os.replace` (`:133-159`).
- `src/mobiletransformers/hub/variant_select.py:21` `Constraints`, `:34` `default_desktop_constraints`, `:39`
  `_variant_download_bytes`, `:59` `select_variant` (soft quant preference `:96`, size tie-break `:97`,
  default tie-break `:98`, 0.9× storage ceiling `:105-112`).
- `src/mobiletransformers/cli/pull.py:18` registers both `pull` and `install-package`.
- Tests: `tests/hub/test_variant_select.py` (6 rows incl. storage-budget fail-closed and abi=null match),
  `tests/hub/test_pull.py` (3).

**Verified present (Android):**
- `hub/HubResolver.kt` (resolve URLs + bearer), `hub/DownloadPlanner.kt:22` `planFiles` (glob→file list
  against `fileSizes`), `hub/PackageDownloader.kt:22` (streaming GET, `Range` resume `:66`, streaming
  SHA-256 `:73-91`, delete+retry then fail `:42-49`), `hub/HubDownloader.kt:20` (manifest first `:42-49`,
  plan `:53`, verify `:54-62`, install `:64`), `hub/PackageDownloadWorker.kt:16`,
  `packages/ModelPackageInstaller.kt:19`, `packages/VariantSelector.kt:9`.
- `MobileTransformers.kt:43-70` triggers pull-then-load when the package is absent.
- Tests: `PackageDownloaderTest.kt` (MockWebServer, verify + retry), `DownloadPlannerTest.kt` (3),
  `PackagesTest.kt` (installer/validator/selector).

**Gaps:**
- **(a) The Kotlin `VariantSelector` is never invoked on the download path.**
  `HubDownloader.kt:51` does `val variantId = variant ?: manifest.defaultVariant`. Device ABI
  (`Build.SUPPORTED_ABIS`), memory (`ActivityManager`), and storage (`StatFs`) are never consulted —
  plan step 5 unimplemented. A device that cannot run `defaultVariant` will download it anyway.
- **(a) The #21 selection policy was never ported to Kotlin.** `VariantSelector.kt:9-40` mirrors only #13's
  hard filters; there is no Kotlin equivalent of `variant_select.py`'s soft quantization preference,
  download-size tie-break, or storage budget. The IMPLEMENTATION_ORDER `[x]` "Is variant selection identical
  Python↔Kotlin (deterministic)?" is **not supported by the code**.
- **(a) `PackageDownloadWorker` is dead code.** Grep across the whole Android workspace finds no
  `WorkManager`/`OneTimeWorkRequest`/`enqueue` reference outside the worker's own file. `fromPretrained`
  calls `HubDownloader.downloadAndInstall` directly on the caller's coroutine
  (`MobileTransformers.kt:52-61`), so plan step 10 ("enqueue worker → await"), the `Constraints`
  (unmetered + storage-not-low), and the foreground-service path for large pulls are all absent.
- **(a) No manifest schema validation before exposing the package.** Plan step 8 requires it; neither
  `HubDownloader` nor `ModelPackageInstaller` calls `ManifestValidator`, and `fromPretrained` doesn't either
  (see #19). `PackageFormat.checkCompat` exists but is only exercised by tests.
- **(a) `ModelPackageInstaller.install` destroys the live cache before the rename.**
  `packages/ModelPackageInstaller.kt:41-47`: `if (target.exists()) target.deleteRecursively()` runs **before**
  `stagingRoot.renameTo(target)`; a crash between the two leaves the model gone. Plan step 9: "never touch
  the live `<cacheDir>/<sanitizedRepoId>`; only `.partial` is mutated until the final rename." (The Python
  `install_package` has the same shape at `hub/pull.py:157-159`, but `os.replace` onto an existing dir is
  not permitted for non-empty dirs, hence the rmtree — same exposure.)
- **(a) No HEAD size/etag pre-check** (plan step 6). `PackageDownloader.fetchToFile` issues a plain GET; the
  manifest's `fileSizes`/`etag` are never cross-checked before the transfer.
- **(c) Python `select_variant` fails closed instead of falling back.** `variant_select.py:70-75` delegates to
  #13's hard-filter selector, which raises `NoCompatibleVariant`. Plan steps 4 and 6 specify a **soft** memory
  filter (best-effort fallback + warning) and a fallback to `manifest.defaultVariant` with a
  constraint-mismatch warning. Neither exists — no warning path at all. `tests/hub/test_variant_select.py:48`
  even encodes the fail-closed behavior as expected.
- **(c) `ManifestClient.kt` / `PackageInstaller.kt` were folded** into `HubDownloader`/`ModelPackageInstaller`
  — a naming deviation, acceptable.
- **(c)** Python `pull_package` verifies sha256 only for files that happen to be on disk
  (`hub/pull.py:103`) — a file silently missing from the download is never flagged. Nothing asserts the
  requested set actually arrived.
- **(c) Device leg unrun** (MockWebServer instrumentation suite from the plan's Manual list does not exist as
  `androidTest`; only the JVM MockWebServer test).

**Drift / doubtful claims:** the `[x]` box and the "Android downloader … reuses `packages/` verify/select/
install" note are half-true: verify and install are reused, **select is not**.

---

### #22 — Adapter push-back (`02_code_plans/06`)

**Required:** `adapter/export.py::export_adapter_from_cache`; `adapter/convert.py::to_peft_layout` (Mode-1
only for LoRA + A/B factors present + rank/alpha known); `adapter/model_card.py` as a **thin wrapper over
`export/model_card.py`** adding adapter sections; `cli/push_adapter.py` with `--peft-only`; Mode-1 emits
`adapter_config.json` **+ `adapter_model.safetensors`**; Mode-2 emits the native subtree in a
`variants/<id>/train/`-style shape + `mobiletransformers_adapter.json`; card must carry the bold privacy
warning, exact base-model license, PEFT type + MARS level + rank/alpha, **dataset notes**, and re-apply
instructions; upload fails if a section is missing; gated/off-by-default Android `AdapterUploader.kt`.

**Verified present:**
- `src/mobiletransformers/adapter/export.py:46` `export_adapter_from_cache` — reads
  `train/training_config.json` + `train/weight_handoff_map.json`, cross-references `externalDataLocation`,
  collects `checkpoint_component_roles` (`:64`).
- `src/mobiletransformers/adapter/convert.py:42` `to_peft_layout` — the three-part deterministic gate
  (`:44` non-LoRA → None, `:46-48` component-role subset check against #6's `component_schema`, `:49-50`
  rank/alpha); `:99` `materialize_peft_weights` with injectable `FactorReader` (`:31`), PEFT key mapping
  (`:84`), and fail-closed on missing factors (`:150-152`).
- `src/mobiletransformers/adapter/model_card.py:10` `PRIVACY_WARNING` (bold), `:17` `render_adapter_card`,
  `:59` `assert_required_sections`.
- `src/mobiletransformers/cli/push_adapter.py:25` (`--cache-repo --repo-id --out --base-license --private
  --peft-only --dry-run`), `--peft-only` hard error at `:54-59`, Mode-2 subtree at `:94`.
- Kotlin `hub/AdapterUploader.kt:41` `AdapterPackageBuilder`, `:71` `AdapterModeGate`, `:80` `AdapterCard`
  (+`assertRequiredSections` `:104`), `:116` `AdapterUploadDisabledException`, `:122` default-off
  `BuildConfig.ADAPTER_UPLOAD_ENABLED`; wired into the `pushAdapter` stub at
  `internal/runtime/RepositoryBackedModelSession.kt:228-237`.
- Tests: `tests/adapter/test_convert.py` (9), `tests/adapter/test_push_adapter.py` (4),
  `hub/AdapterUploaderTest.kt` (7).

**Gaps:**
- **(a) The CLI never materializes `adapter_model.safetensors`.** `cli/push_adapter.py:67-72` writes
  `adapter_config.json` and leaves the comment "materialization is env-gated"; `materialize_peft_weights` is
  never called from any CLI or library entry point (only from the test). So a Mode-1 push uploads a PEFT
  config with **no weights** — the plan's DoD ("emits either a PEFT-compatible layout (`adapter_config.json`
  + `adapter_model.safetensors`)") and its integration test ("PEFT mode uploads `adapter_config.json` +
  `adapter_model.safetensors`") are unmet. `tests/adapter/test_push_adapter.py:21-28` asserts only the config.
- **(b) The safetensors path is untested in CI.** `tests/adapter/test_convert.py:77,106` `importorskip`
  torch/safetensors and **both skip in the current env** (verified: `-rs` output). The whole
  numpy→torch→safetensors leg is unexercised by the gate.
- **(c) Mode-2 layout drift.** `cli/push_adapter.py:94-127` writes the `.bin`s, handoff map,
  `training_config.json`, and `mobiletransformers_adapter.json` **flat at the adapter repo root**; the plan
  specifies "a `variants/<id>/train/` style subtree (#14)".
- **(c) `adapter/model_card.py` is not a wrapper.** It builds its own string (`:17-56`) and never imports
  `export/model_card.py::render_model_card`, contradicting the plan's "thin wrapper over
  `mobiletransformers/export/model_card.py` (#15) adding adapter-specific sections" and #15's "shared
  `model_card.py`" interaction.
- **(a) Card is missing the mandatory "Dataset notes" section** (plan's model-card mandatory list item 3);
  `assert_required_sections` (`:59-73`) checks privacy warning / `## Licenses` / peft method / `Rank:` only —
  it does **not** verify the *exact base-model license string* was supplied (a bare `--base-license`
  default of `"see upstream"` passes the gate).
- **(a) Kotlin gate diverges from the Python gate.** `hub/AdapterUploader.kt:76-77`:
  `peftMethod == "lora" && rank != null && alpha != null → PEFT`. It never checks A/B factor presence, which
  is the Python gate's decisive signal (`adapter/convert.py:46-48`). A factor-less LoRA is therefore Mode-1
  on device and Mode-2 in Python — a cross-language contract split. The test name
  `AdapterUploaderTest.loraWithFactorsIsMode1Peft` **overclaims** what the gate checks (the fixture's factors
  are irrelevant to the decision).
- **(c)** The Kotlin card's privacy wording differs from Python's and `AdapterCard.assertRequiredSections`
  checks only 2 of the 4 sections the Python assert checks.
- **(d) Device upload deliberately not implemented** — `pushAdapter` throws
  `NotImplementedFeatureException("pushAdapter upload (device leg)")` after preparing/validating the card
  (`RepositoryBackedModelSession.kt:236`). Correctly gated and documented.

**Drift / doubtful claims:** HANDOFF's "**#22 is genuinely done**" (line ~421) is not supported: the Mode-1
weight file is never written by the shipping path, and the on-device gate is a different algorithm.

---

## Tier-doc requirements not picked up by any code plan

From `agent_docs/02_tier1_hf_integrated_core.md`:

1. **`docs/HUB_PACKAGE_FORMAT.md` and `docs/ANDROID_CACHE_FORMAT.md`** (Implementation Sequence step 14) —
   neither exists. `docs/MODEL_FORMAT.md` (#31) covers roughly 70% of both. `PUBLIC_API.md` and
   `MODEL_FORMAT.md` do exist.
2. **Starter model zoo** (steps 13 + 15: "starter package generation for one smallest Optimum-supported
   text-generation model", "Hub upload for starter packages") — no plan in `02_code_plans/` owns this.
   `mobiletransformers push` is the mechanism, but there is no zoo definition, no `mobiletransformers/*`
   repo list, and no license-check step ("Starter model zoo hosting requires license checks" — Risks).
3. **`MobileTransformersApp Improvements`** (whole section) — only the facade migration is planned (#19,
   undone). Unowned entirely: the model-package selection/validation flow, the **package cache screen**
   (installed packages + manifest metadata + storage usage), the developer-settings screen for advanced
   engine/debug controls, and the adapter export/share action.
4. **`default/` directory** (tier doc + #14 layout: "`default/` exists so the simplest Android and CLI flow
   can download one known-good package") — not implemented, not aliased, not documented anywhere.
5. **HEAD metadata / `etag`** (tier doc Android pull step 5: "Use HEAD metadata when available to show size
   and validate ETag/commit information before downloading large files") — `etag` is in #14's field list but
   never emitted; no HEAD request exists in `PackageDownloader`.
6. **Support-matrix candidate promotion into `config.yml`** (#20 step 1 + tier doc's candidate families) —
   the commented families in `config.yml` were never promoted to a `SUPPORT_MATRIX.candidates` block.
7. **Tier-doc test list items with no implementation:** "CLI dry-run test from `config/config.yml`" (the
   dry-run test exists but reads no YAML), "Hub upload dry-run with generated model card" (exists for
   `push`, not for the starter-zoo flow), "Compatibility smoke for old `ortmobile` imports" (moot after #16).
8. **`pushAdapter` from device** is listed in the tier doc's Kotlin facade surface with the caveat "only if
   Hub upload is feasible from device; otherwise expose as Python-side package operation first" — the code
   correctly takes the fallback, so this is satisfied by deferral, not a gap.

---

## Remaining work, ordered

### Host-doable now (no device, no Gradle)
1. **#15** Add `--validate` (train-step + 1-token smoke, record pass/fail into `export_report.json`) — the
   only outright missing CLI flag in the tier. `src/mobiletransformers/cli/export.py`, `export/pipeline.py`.
2. **#15** Wire `--config` / `config.yml` overlay through `plan_export` (or delete the flag and fix
   `docs/EXPORT.md:27`), and add the plan's config-overlay unit test.
3. **#15** Emit `optimum/optimum_config.json`, `train/trainable_parameters.json`, and split
   `shared/chat_template.jinja` out of the tokenizer (`export/pipeline.py:199-207`, `:435`).
4. **#20** Replace `matrix.py:24/54` with `from mobiletransformers.export.registry import choose_task`
   (normative), and read candidates from `config.yml SUPPORT_MATRIX.candidates` /
   `config/support_candidates.yml` instead of `cli/support_matrix.py:16`.
5. **#20** Make `mobile_package_exportable` real (read `optimum/export_report.json` when present, else the
   tokenizer/gen-config/manifest dry-run) and key probes by `(modelId, variant)`.
6. **#22** Call `materialize_peft_weights` from `cli/push_adapter.py` in Mode-1 (guarded, with a clear error
   naming the profile when torch/safetensors are absent); add the dataset-notes card section and tighten
   `assert_required_sections`; move the Mode-2 output under `variants/<id>/train/`.
7. **#22** Align `hub/AdapterUploader.kt:76` with the Python gate (require A/B factor roles from the handoff
   map's `checkpointNames`) and rename/fix `AdapterUploaderTest.loraWithFactorsIsMode1Peft`.
8. **#21** Port the #21 selection policy to Kotlin and call `VariantSelector` from `HubDownloader.kt:51`
   with `Build.SUPPORTED_ABIS`/`ActivityManager`/`StatFs`; decide fail-closed vs the plan's
   fallback-to-default-with-warning and make Python and Kotlin agree.
9. **#21** Fix `ModelPackageInstaller.kt:41-47` (and `hub/pull.py:157`) so the live cache dir is replaced,
   not deleted-then-renamed (rename old → `.trash/` first, publish, then delete).
10. **#21/#19** Run `ManifestValidator` before exposing a package (both after install and in
    `fromPretrained` for already-installed packages).
11. **#19** Either implement `applyPeft`'s rank/alpha validation + threading into `ORTTrainingConfig`, or
    delete the dead `appliedPeft` field and drop the claim.
12. **#19** Fill the public-config gaps: `DatasetConfig` (4 missing fields + mappings), `GenerationConfig.trackMetrics`,
    `TrainConfig` scheduler start/end factors.
13. **#14** Write `docs/HUB_PACKAGE_FORMAT.md` (DoD item) and document/implement or explicitly retire the
    `default/` alias policy; guard `build_manifest:143` against variants with no `inference` subtree.
14. **#20** Add the F1 schema-version test + `transformersCeiling` assertion for the matrix envelope.
15. **#15** Strengthen the export-E2E checkpoint: assert `checksums.json`, `optimum/*`, merger graphs and a
    non-empty handoff map, and enrich `tests/fixtures/tiny_package/*/train/` with the four ORT artifacts.

### Device-required
16. **#19 checkpoint** — `TrainMergeGenerateTest` over a real train-capable package
    (`make device-package TRAIN=1 MODEL=<id>` → `make device-test`); add `applyPeft` to the sequence.
17. **#21 checkpoint** — `FacadeLoadGenerateTest` over a pulled+installed package; plus the plan's Manual
    instrumentation list (manifest-first ordering, grouped download, tamper→retry→fail, kill/resume via
    Range, atomic-rename visibility, post-install `isGenerationAvailable`) — none of these exist as
    `androidTest` yet.
18. **#21** WorkManager path: enqueue `PackageDownloadWorker` with `unmetered` + `storage-not-low`
    constraints and a foreground notification; verify scheduling/resume on device.
19. **#22** Real authenticated device upload behind `ADAPTER_UPLOAD_ENABLED` (after the security review the
    plan requires) and the ORT `CheckpointState` factor read on device.
20. **#20** Produce a real `build/support/android_probes.json` from the instrumentation runs so
    `android_*_ready`/`rag_ready` stop being uniformly `false`.

### Manual / user-run
21. **#15** Real tiny-model export with the source-built ORT-training wheel:
    `uv run --python 3.12 --group ort-training-local … --stages training` into an existing inference package,
    then re-validate — the only way to prove the two-profile flow end to end.
22. **#15** `tests/integration/test_training_stage_smoke.py` under `ort-training-local` (requires generating
    `tests/fixtures/tiny_trainable.onnx` via `make_tiny_trainable.py` first).
23. **#22** PEFT round-trip: export a Mode-1 adapter, reload with `PeftModel.from_pretrained` (needs
    torch+peft) — blocked on item 6 above.
24. **#20** Live un-mocked `support-matrix` run over the real candidate families (network + `AutoConfig`).
25. **#15/#21** A real `huggingface_hub` `snapshot_download` pull against a published package — every
    current pull test injects a fake downloader.

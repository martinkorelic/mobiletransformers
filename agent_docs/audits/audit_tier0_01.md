# Tier 0 — 01_code_plans (engine/export core) audit

Scope: the 6 plans in `agent_docs/01_code_plans/` (global #3, #7, #9, #10, #11, #12).
Method: read the plan + gate docs, then verified every claimed symbol/file/test against the tree
(`src/`, `inference/`, `artifact/`, `spikes/`, `tests/`, `android/MobileTransformersApp/MobileTransformers/src/`).
Read-only; no repo file changed. All paths absolute below unless obviously relative to the repo root
`/home/martin/Documents/Projects/Development/LLM_finetuning/mobiletransformers`.

## Summary table

| # | Plan | Claimed | Verified | % | Verdict |
| --- | --- | --- | --- | --- | --- |
| 3 | `06_source_built_ort_training_pipeline.md` | done (Gate 0.3 proven) | scripts + manifest + BUILD.md + fixture + smoke test + workflow all present; torch ABI locked (2.7.1); wheel sha matches manifest gate in CI | **90%** | Substantially true. Android AAR leg never built (manifest `ndk_version`/`abis`/`aar_sha256` = null) |
| 7 | `05_optimum_onnx_export_and_tasksmanager.md` | done | `export/{registry,inference_export,normalize,support_matrix,onnx_config_with_loss,torch_frontend}.py` all real; `trainer/builder.py` ladder gone; 27 tests | **93%** | True. Only the legacy `inference/builder.py:3237` ladder remains (explicitly out of scope, but its deferral rationale has now expired) |
| 9 | `01_unified_merger_and_external_data_export.md` | code-complete, device tests outstanding | merger collapse + golden test + export orchestrator + C++ save-side real; **no offline merge/write driver**; C++ literal merger branches + `TODO` still present; `gen_genai` not deprecated | **70%** | Overstated. Two DoD bullets are objectively unmet |
| 10 | `02_genai_external_data_swap_spike.md` | **Gate 0.1 = ADOPT** | all spike artifacts exist and are real; symbol/desktop/device swap evidence recorded in `spikes/genai_external_swap/README.md` | **80%** | Evidence real, but the gate was closed with 2 of 6 PASS criteria (#1, #4) admittedly unproven |
| 11 | `03_inference_engine_abstraction_native_and_genai.md` | code-complete, device leg open | interface/impls/JNI/factory/tests exist, **but GenAI is unreachable end-to-end** (2 wiring bugs); EP registry unconsumed; manifest engines unread | **65%** | Overstated. "One `ModelRuntime` … over one package" holds structurally, not behaviourally |
| 12 | `04_memory_mapping_experiments.md` | code-complete, device RSS table = manual | `mem_probe.h`, `mmap_tensor.h`, default-off branch, desktop spike, RSS re-export all exist | **45%** | Harness only. 2 of 4 experiments unimplemented, no unit tests, and the plan's own lifetime hazard is live in the code |

**Tier estimate (these 6 plans): ~72% complete.** The three Python-owned plans (#3/#7/#9-Python) are in
good shape; the device/engine half (#9-C++, #11, #12) is where the gap is.

## Gate status (0.1 / 0.2 / 0.3)

### Gate 0.1 — GenAI adopt/reject: **recorded as ADOPT, but 2 of 6 PASS criteria are unproven**
Evidence lives in `spikes/genai_external_swap/README.md:69-80` and is genuine (I verified the harnesses exist
and do what the table says):

| Criterion | Status in repo | Verified |
| --- | --- | --- |
| 6 symbol fork-only | PASS | `spikes/genai_external_swap/check_symbols.sh:41-52` really asserts absent/present against the AAR `.so` |
| 2 external swap changes output | PASS (desktop+device) | `spikes/genai_external_swap/desktop_spike.py:104-123` asserts `not allclose(L_base, L_swap)` on a **fresh** `og.Model` (`desktop_spike.py:37`) |
| 3 not constant-folded | **implied only** | no explicit guard exists — the plan required parsing `model.onnx` and asserting each `inferenceInitializerNames[role]` survives as a live external initializer (`02_genai…md:168`). Nothing in the tree does this |
| 5 Android relative-external resolution | PASS (device) | `cpp/genai_spike.cpp` + `androidTest/.../GenAISpikeTest.kt` exist |
| 7 RSS mmap-vs-copy | measured (informational) | `spikes/genai_external_swap/measure_rss.py` |
| **1 same package correct under BOTH engines** | **UNPROVEN** | harness `androidTest/.../DualEngineParityTest.kt` exists but **cannot pass today** — see #11 gaps (the GenAI leg returns `""`) |
| **4 GenAI peak RSS within `ACCEPTED_RSS_DELTA` of Native** | **UNPROVEN + threshold never ratified** | grep for `ACCEPTED_RSS_DELTA` across the tree: zero hits. No number was ever fixed, and `DualEngineParityTest` asserts tokens only, no RSS |

Verdict: Gate 0.1's *GenAI-side* questions are genuinely answered; the *cross-engine* half is not, and the
"ADOPT" decision in `IMPLEMENTATION_ORDER.md:60` is ahead of its own stated criteria.

### Gate 0.2 — mmap / RSS: **not reached**
- No four-point RSS table exists anywhere in the repo (searched `docs/`, `spikes/`, `agent_docs/`).
- The pre-registered ≥15% margin is nowhere ratified.
- Only experiments (a)-partial and (c)-partial are implemented (see #12 findings); (b) and (d) have no code.
- `IMPLEMENTATION_ORDER.md:259-260` correctly leaves both self-check boxes unticked.

### Gate 0.3 — ORT training toolchain: **PASS for the Python/desktop leg**
- `third_party/onnxruntime/manifest.json` records ORT SHA `9b25b6a838…`, build flags, `torch_version: 2.7.1`,
  wheel sha256 `87e6f3c6…`; the wheel is present (git-ignored) at `third_party/wheels/`.
- `tests/integration/test_ort_training_smoke.py:47-60` really calls `generate_artifacts(... AdamW ...)` and
  asserts the four artifacts; it `importorskip`s outside the training profile.
- `.github/workflows/ort-training-smoke.yml:36-42` re-hashes the wheel against the manifest (the plan's
  "provenance smoke") before syncing.
- **Outstanding:** the Android AAR half of the gate — `scripts/build_ort_training_android.sh` exists but was
  never run; `manifest.json` `ndk_version`, `android_api_level`, `abis`, `android.aar_sha256` are all
  null/empty, so "the device build matches the desktop wheel's ORT revision" is asserted by comment only.

## Per-plan findings

### #3 — `01_code_plans/06_source_built_ort_training_pipeline.md`

**Required:** two build scripts, `third_party/onnxruntime/{manifest.json,BUILD.md}`, uv local-wheel wiring +
conflicts, tiny fixture, CI smoke, torch ABI locked.

**Verified present:**
- `/scripts/build_ort_training_wheel.sh`, `/scripts/build_ort_training_android.sh`.
- `/third_party/onnxruntime/manifest.json` (all Python-side provenance fields populated, incl. the "locked
  unknown" `torch_version: 2.7.1` and the ABI constraints note), `/third_party/onnxruntime/BUILD.md`.
- `/pyproject.toml:83-87` — `[tool.uv] conflicts` isolates `ort-training-local` × `export` × `genai-smoke`.
- `/tests/fixtures/tiny_trainable.onnx` + `/tests/fixtures/training_config.json` + `/tests/fixtures/test_tiny_trainable.py`.
- `/tests/integration/test_ort_training_smoke.py` (artifacts + one-step finite-loss).
- `/.github/workflows/ort-training-smoke.yml`.

**Gaps:**
- (c) Android AAR build never executed → `manifest.json` android fields null (`ndk_version`, `abis`,
  `android_api_level`, `android.aar_sha256`). The device training C++ API in
  `cpp/session_cache.h` is currently satisfied by headers/libs vendored out-of-band from a sibling checkout,
  not by a manifest-checksummed AAR.
- (d) CI provisioning of the wheel deliberately deferred to #29 (`ort-training-smoke.yml:1-10` documents it).

**Drift / doubtful claims:** none material. The self-check note at `IMPLEMENTATION_ORDER.md:128-130` matches
what is on disk.

### #7 — `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`

**Required:** TasksManager discovery wrapper, `main_export` front door, normalizer, support matrix,
`EXPORT_FRONTEND_REGISTRY` (F3), migration spike, `trainer/builder.py` de-laddered.

**Verified present:**
- `src/mobiletransformers/export/registry.py:60,79,107,148,164` — `supported_onnx_tasks`, `discover_tasks`,
  `choose_task`, `EXPORT_FRONTEND_REGISTRY`, `resolve_frontend`.
- `src/mobiletransformers/export/inference_export.py:64,100` — `optimum_onnx_export`, `export_inference`.
- `src/mobiletransformers/export/normalize.py:61` `normalize_package`; `support_matrix.py:76,99`.
- `src/mobiletransformers/export/onnx_config_with_loss.py` (vendored) + `torch_frontend.py:28` (fail-closed).
- `trainer/builder.py:270-282` — `resolve_architecture(config)` + `choose_task(...)` + vendored
  `OnnxConfigWithLoss`; no `architectures[0]` ladder (`grep` confirms only a comment remains at `:267`).
- `spikes/optimum_migration/check_symbols.py`; `tests/export/*` = 27 tests, 6 profile-gated.

**Gaps:**
- (c) real full-size export smoke is manual; it was in fact run under #15 (SmolLM2-135M) per
  `IMPLEMENTATION_ORDER.md:276`, so this is effectively closed but not automated.
- (d) `inference/builder.py:3237-3241` still carries an `architectures[0] ==` ladder. #7 deferred it "gated by
  the Optimum-vs-GenAI decision" — **that decision was made (Gate 0.1 = ADOPT, 2026-07-15), so the deferral
  rationale has expired and the debt now has no owner.**

**Drift / doubtful claims:** none. This is the best-evidenced plan of the six.

### #9 — `01_code_plans/01_unified_merger_and_external_data_export.md`

**Required (DoD, plan lines 134-137):** one `export_inference_package.py` entry producing the flat package;
**offline (`artifact/merger.py`) and device (`WeightMerger`) both writing merged tensors to the exact
handoff-map filenames**, both failing closed; the `onnx_builder.py:628-641` dispatch **and the C++ literal
branches** gone; `gen_genai` a deprecated shim or removed.

**Verified present:**
- `src/mobiletransformers/config/registry/merger.py:57-91` — real `build_merger_model`; LoRA/MARS graph
  builders at `:107` / `:222`, quant axes honored independently.
- `tests/unit/test_merger_builder.py:56-68` — genuine byte-for-byte golden equivalence over the full
  family × quant_in × quant_out cross-product against `tests/fixtures/merger_golden/*.onnx` (8 goldens on
  disk); `:90-131` numerical check vs a numpy reference under the export profile.
- `inference/export_inference_package.py:213-315` — base/trainable split (`_split_external_data:149`),
  per-tensor `.bin` + atomic `.sha256` sidecars (`:283`), handoff map via #8's codec (`:268-293`),
  `mergerModels` (`:186-192`), genai session entry (`:195-210`), `model_input`/`adapter` fail closed (`:233`).
- `tests/unit/test_export_inference_package.py` — 5 tests that really assert the flat layout, map validity,
  sha256↔sidecar agreement, frozen-base exclusion, and both fail-closed paths.
- `artifact/onnx_builder.py:630-645` — the `peft_method == "lora"/"mars"` dispatch is genuinely replaced by
  `emit_merger_models` / `resolve_merger`.
- C++ save side: `cpp/weight_merger.cpp:135-157` `write_raw_tensor_atomic` (tmp → `std::filesystem::rename` →
  `.sha256`), `:585` `load_handoff_map`, `:961-995` `save_merged_parameters` keyed by
  `externalDataLocation[role]`; `inference_name` string-rewrite is gone; `cpp/handoff_io.h` is a single shared
  reader with a real `check_compat` mirror (`:47-57`).
- Kotlin caller: `ORTTrainerNative.kt:602-614` points merge at `inference/` (no `merged/` subdir).
- `config/config.yml:184` `handoff_mode: external_initializer`.

**Gaps:**
- **(a) The offline merge/write driver does not exist.** `artifact/merger.py` (55 lines) only *emits merger
  ONNX graphs* (`emit_merger_models:28`); its own header says "The numerical merge itself runs on device"
  (`artifact/merger.py:10`). Plan step 6 and the DoD require the offline driver to run the merger, look up
  `externalDataLocation[role]`, write temp → fsync → `os.replace`, write `.sha256` and update the entry's
  `sha256[role]`. Consequences: the plan's **"Atomic-overwrite smoke (Python)"** integration test is
  unwritable, and the **"offline-vs-device byte-identical parity"** manual test has no offline side to compare
  against — i.e. the #9 self-check "[x] Do offline and on-device merge emit identical external-initializer
  filenames" is only true of the *emit* half.
- **(a) The C++ literal merger dispatch is still there.** `cpp/weight_merger.cpp:622-645` `get_merger_type`
  still returns `"mars_q"/"lora_q"/"lora"` from an if/else over parameter presence and still carries
  `// TODO: Custom merger model?` at `:641` — the plan says this TODO is closed. `run_merger_model` still
  branches on `merger_type == "lora"` / `"lora_q"` / `"mars_q"` at `:672, :708, :759, :833, :853`. The DoD
  bullet "the C++ literal branches … are gone, replaced by registry/handoff-map lookups" is **false**. (Only
  the *filenames* and the *session map keys* are map-driven, at `:600-618`.)
- **(a) Device merge is fail-open, not fail-closed.** `save_merged_parameters:967-971` logs and `continue`s
  when a merged layer has no handoff entry; `save_role:984-988` returns on a missing role; `run_merger_model`
  `:650-653` returns when the merger session is absent; `merge_and_export_weights` returns `true`
  unconditionally (`:1085`). Plan step 8 requires an explicit abort. A partial merge therefore looks like a
  success to `ORTTrainerNative`.
- **(a) On-device string rewrite survives:** `weight_merger.cpp:1069`
  `replace_prefix(base_layer_name, "base_model.model.model.", "backbone.model.")` — a hard-coded name rule the
  map was meant to own ("no string-munging on device", plan line 19).
- **(a) Post-merge checksum contract is self-contradicting (likely device breakage).** The exporter stamps
  `entry.sha256[role]` with the hash of the *exported* bytes
  (`inference/export_inference_package.py:284-287`); the device merger overwrites the `.bin` and refreshes
  only the sibling sidecar (`weight_merger.cpp:151-155`) — it never rewrites `weight_handoff_map.json`
  (verified: no writer for that path in any `.cpp/.h`). But
  `internal/runtime/HandoffPrecondition.kt:60-69` prefers `entry.sha256[role]` **over** the sidecar, so after
  a successful on-device merge the next load throws
  `MissingArtifactException("checksum mismatch …")`. This breaks exactly the train→merge→generate path
  `androidTest/.../TrainMergeGenerateTest.kt` is meant to prove.
- **(a) `gen_genai` was never deprecated.** `artifact/onnx_builder.py:387` still defines the full
  `gen_genai(..., weight_input=True, ...)` path (with `force_dequantize_external_and_save` still at `:187`)
  and it is still called at `:590`; `inference/builder.py:328` `make_genai_config` is still called at `:3286`.
  No `DeprecationWarning`, no delegation. The plan's stated purpose — collapsing the *two overlapping export
  paths* — is achieved only by *adding a third* (`export_inference_package.py`), and `inference/builder.py`
  received none of the plan's steps 1-5 edits.
- **(a) Multi-variant merger emission gap:** `export_inference_package._emit_merger_models:186-192` emits
  exactly one `MergerSpec`. `artifact/merger.emit_merger_models` supports the MARS + LoRA mix
  (`extra_methods`) that the device actually needs (`onnx_builder.py:637-644` uses it), so a MARS package
  produced through the #15 pipeline will lack the LoRA merger sessions for its non-MARS layers — and the
  device will silently skip those layers (see fail-open above).
- (b) `genai_config.json` config entries: only `session.model_external_initializers_file_folder_path` is
  added (`export_inference_package.py:52,208`), and only when the *training* stage runs. The plan's step 5
  "plus the harmless ORT entries from Tier-0 (`use_env_allocators`, `qdq_matmulnbits_accuracy_level`, …)" is
  unimplemented; `export/pipeline.py:492` writes `"session_options": {"provider_options": []}` with no
  `config_entries` at all.
- (c) genuinely device-only and legitimately outstanding: atomic-overwrite-under-kill, byte-parity, native
  load-and-generate.

**Drift / doubtful claims:**
- `IMPLEMENTATION_ORDER.md:232` claims the merger dispatch replacement and `weight_merger.cpp` rewrite is
  complete; the literal branches and the `// TODO: Custom merger model?` are still in the file.
- `src/mobiletransformers/config/registry/merger.py:8-12` docstring still says `build_merger_model` "fails
  closed rather than silently emitting a wrong graph" *until #9 wires it* — stale, it is wired.
- `IMPLEMENTATION_ORDER.md:226` "atomic rename + checksum enforced on both the Python and C++ sides" — the
  Python side enforces it for the **export** sidecars only; there is no Python merge writer.

### #10 — `01_code_plans/02_genai_external_data_swap_spike.md`

**Required:** `desktop_spike.py`, `measure_rss.py`, `genai_spike.cpp`, `check_symbols.sh`, the desktop
integration checks (swap / constant-folding guard / config pass-through), the device manual runs, and a
written Gate 0.1 decision.

**Verified present:**
- `spikes/genai_external_swap/{desktop_spike.py,measure_rss.py,check_symbols.sh,build_tiny_genai_model.sh,setup_ort_separation.sh,README.md}`
  — all real, runnable, and consistent with the README's claims. `desktop_spike.py:53-69` handles **both** the
  #9 per-tensor layout and the single-blob builder layout, and never perturbs `frozen_base.onnx.data`.
- `cpp/genai_spike.cpp` + `GenAISpike.kt` + `androidTest/.../GenAISpikeTest.kt`.
- The ORT-coexistence resolution (`libort_gen.so` distinct-SONAME) is scripted and reproducible
  (`setup_ort_separation.sh:16-30`) — this is the strongest artifact in the tier.
- Gate 0.1 decision recorded at `spikes/genai_external_swap/README.md:69-110` and mirrored in
  `IMPLEMENTATION_ORDER.md:237-249`.

**Gaps:**
- (a) **Constant-folding guard** (plan line 168, gate criterion 3) has no implementation — it is inferred
  from the logits diff, not asserted.
- (a) **Config pass-through smoke** (plan step 6 / line 169): nothing sets a probe `log_id` or checks ORT
  logs; the only `log_id` in the tree is the static one in the legacy `inference/builder.py:344`.
- (c) Gate 0.1 legs #1/#4 are device-only *and* currently blocked by the #11 wiring bugs below, not merely by
  the absence of a #9 package.
- (d) `ACCEPTED_RSS_DELTA` was never defined — the plan required ratifying a number at Gate 0.1.

**Drift / doubtful claims:** README `:75` "criterion 3 … implied by #2" is honest but the plan called it a
hard-fail condition needing its own guard. `IMPLEMENTATION_ORDER.md:60` marks #10 `[x]` (done) while its own
text concedes #1/#4 ride with #11 — the box is ahead of the gate.

### #11 — `01_code_plans/03_inference_engine_abstraction_native_and_genai.md`

**Required:** one `ModelRuntime` + `InferenceEngine` + `EngineCapabilities`; Native adapted; GenAI impl over
`genai_runtime.cpp`; manifest `supportedEngines`/`defaultEngine` read; `EXECUTION_PROVIDER_REGISTRY` driving
Native's EP order; selector/fallback in `LLMRepository`; dead code deleted; a **streaming-parity harness**.

**Verified present:**
- `runtime/ModelRuntime.kt:19-33` interface, `:36-41` `EngineCapabilities`, `:57-72` `EngineRegistry`,
  `:78-86` `GenAiSupport`, `:93-136` `ModelRuntimeFactory` (pure `selectEngine` + `create` with
  catch→Native); `runtime/InferenceEngine.kt` is the single enum.
- `ORTGeneratorGenAI.kt:18-138` implements `ModelRuntime` and emits
  `onStartGeneration → onPartialResult* → onCompletion` (`:65,:84,:95`), `onError` at `:105`.
- `ORTGeneratorNative.kt:12,22-43` adapted to `ModelRuntime`; `loadMergedWeights` probe replaced by
  `HandoffPrecondition` (`:72-82`).
- `cpp/genai_runtime.cpp` — full session wrapper over the stable C API (`OgaCreateModel:59`, streaming
  decode `:143`), registered in `cpp/CMakeLists.txt:38`; `ORTGenAINative.kt` / `onnx-genai.cpp` are gone
  (guarded by `test/.../NativeLoadRegressionTest.kt:47-53`).
- `test/.../runtime/RuntimeSelectionTest.kt` — 7 real tests covering the selection matrix and EP-registry
  parity with an injected probe. Module-wide JVM tests: 125 `@Test`; androidTest: 7.
- `androidTest/.../DualEngineParityTest.kt` exists as the device harness for Gate 0.1 #1.

**Gaps:**
- **(a) GenAI is unreachable through the public facade — two independent breaks.**
  1. `internal/config/ConfigMappers.kt:86-94` builds `ORTGenerationConfig` with `type = "genai"` but **never
     sets `engine =`**. `ModelRuntimeFactory.selectEngine` reads `config.engine` (`ModelRuntime.kt:125`), which
     is therefore always `null` → default `NATIVE`. GenAI can never be selected via the facade.
  2. `repository/LLMRepository.kt:357-368` (`prepareRetriever`) and `:391-401` (`runGenerationStream`) and
     `:441` still `when (finalGenConfig.type) { "native" -> … else -> Log.e("Unknown generation type") }`.
     With `type = "genai"` nothing is constructed and `generate` returns `""`.
  Net effect: `DualEngineParityTest` (Gate 0.1 #1) would compare a real Native token against `""` and fail;
  the self-check "[x] one `ModelRuntime` … over the same package … `ModelRuntimeFactory.create`"
  (`IMPLEMENTATION_ORDER.md:252`) is structurally true but behaviourally false.
- **(a) Manifest engine fields are not read.** `LLMRepository.kt:273-275` admits it in a comment and passes
  the default `setOf("native","genai")`. Plan step 5 ("read `variants[].supportedEngines` + `defaultEngine`")
  is unimplemented; and no `defaultEngine` field exists in `packages/MobileTransformersManifest.kt` at all
  (only `defaultVariant` + per-variant `supportedEngines`).
- **(a) `EXECUTION_PROVIDER_REGISTRY` is declared but unconsumed.** Grep shows `providersFor` used only by
  `RuntimeSelectionTest`. Native's EP still travels as a raw string
  (`ORTGeneratorNative.kt:97` → JNI → `session_cache.h::setSessionOptions`). The plan's "Native composes its
  `SessionOptions` EP append-order from the registry" is not done; the rows' `available` probes for
  xnnpack/nnapi are hardcoded `{ true }` (`ModelRuntime.kt:61-62`), so F3 is cosmetic here.
- **(a) `genaiAvailable()` is a stub.** `cpp/genai_runtime.cpp:172-175` `nativeGenAiAvailable` returns
  `JNI_TRUE` unconditionally — plan step 7 required running the #10 symbol check once (cached) plus the Gate
  0.1 flag.
- **(b) Callback parity is not enforced anywhere automatable.** The plan's Integration test ("shared recorder
  asserts both engines emit the same ordered event *types* over a scripted token stream", plan line 122) does
  not exist; only the device `DualEngineParityTest` (token equality, not event sequence). Neither engine's
  `generate` emits `onModelLoadStart`/`onModelLoadEnd`, which the plan's contract lists first
  (`03_inference…md:70`; the callbacks are declared at `LLMRepository.kt:41-42`).
- **(a) Engine-local string dispatch re-introduced.** `ORTGeneratorGenAI.kt:118-123` maps sampling with
  `when (method) { "top_k" -> 1; "top_p" -> 2; else -> 0 }` — duplicating and **fail-opening** what #24 made a
  single fail-closed source on Native (`ORTGeneratorNative.kt:305`
  `SamplingMethod.fromWire(...).nativeOrdinal`). Contradicts the canonical "enums own every closed string set"
  + "fail closed is the default" decisions.
- (b) `ORTGeneratorGenAI` ignores `seed` (stored at `genai_runtime.cpp:87`, never applied in `nativeStart`),
  so sampled generation is not reproducible and cannot be compared across engines except at temp 0.
- (b) `ORTGeneratorGenAI.capabilities.supportsLoadMergedWeights = true` is hardcoded (`:32`) while Native
  computes it from `HandoffPrecondition` — the two engines report different truths for the same folder.
- (c) genuinely device-only: same-folder dual-engine smoke, streaming parity at temp 0, merged-weights
  reflected per engine.

**Drift / doubtful claims:** `IMPLEMENTATION_ORDER.md:61` "`LLMRepository` wired via factory" is true of
`makeModelRuntime` but omits that the `when(type)` gate upstream of it rejects the GenAI config the facade
produces. HANDOFF (`agent_docs/HANDOFF.md:~487`) calls the manifest wiring "a small follow-up"; it is, but
combined with the `engine`-not-propagated bug it means no GenAI selection path is live at all.

### #12 — `01_code_plans/04_memory_mapping_experiments.md`

**Required:** `mem_probe.h`, `mmap_tensor.h`, a file-backed loader alongside the copy path,
`spikes/mmap/{measure_rss.py,base_blob_mmap_spike.py}`, unit tests for both headers, the desktop correctness
invariant, the config pass-through smoke, and the four experiments (a)-(d) with a four-point RSS table.

**Verified present:**
- `cpp/mem_probe.h:18-50` — `read_rss_kb()` + a `parse_vmrss_kb()` written specifically to be unit-testable;
  `LOG_RSS` macro at `:54`, wired at `session_cache.h:66,133`.
- `cpp/mmap_tensor.h:19-82` — correct move-only RAII `MmapRegion`.
- `session_cache.h:70` env-gated `MTF_MMAP_WEIGHTS` branch, `:91-111` maps the per-tensor `.bin` and wraps it
  with `Ort::Value::CreateTensor`, falling back to the copy path for quantized/odd tensors; copy path remains
  the shipping default.
- `spikes/mmap/measure_rss.py` (a true re-export, as the plan demanded) and
  `spikes/mmap/base_blob_mmap_spike.py:51-93` (byte-identical-logits invariant, hard-fail on divergence).

**Gaps:**
- **(a) The plan's own lifetime hazard is live.** `mmap_regions_` is cleared inside `clearWeights()`
  (`session_cache.h:216-217`), and `InferenceSessionCache`'s ctor calls
  `weight_session->clearWeights()` immediately after the `Ort::Session` is constructed
  (`session_cache.h:517-521`). The plan states explicitly: "The `clearWeights()` early-free … is
  **incompatible** with mmap lifetime — any mmap experiment must move release to the destructor or it will
  fault on first run" (`04_memory_mapping_experiments.md:126`, restated at `:34`). The header comment at
  `session_cache.h:48-50` even claims the regions are "freed only in clearWeights()/destruction", which is the
  wrong half of the requirement. First device run with `MTF_MMAP_WEIGHTS=1` is a use-after-unmap candidate.
- **(a) Experiment (a) — the highest-leverage one — is not implemented on device.** The mmap branch covers the
  *trainable per-tensor* `.bin` files only; the frozen-base blob path (add
  `session.model_external_initializers_file_folder_path` to the Native session options and measure) is absent
  — `session_cache.h:786-795` sets the legacy entries and still forces
  `session.use_ort_model_bytes_for_initializers = "0"` with no toggle.
- **(a) Experiment (b)** (skip `AddExternalInitializers`, rely on folder resolution behind a
  `loadStrategy = FOLDER_RESOLUTION` flag) — no such flag or branch exists.
- **(a) Experiment (d)** (GenAI passthrough of `use_ort_model_bytes_*` / `use_memory_mapped_ort_model` via
  `genai_config.json` `config_entries`) — no code emits those keys anywhere.
- **(b) No unit tests.** Both plan Unit items (`read_rss_kb` fixture parse, `mmap_tensor` destructor lifetime)
  are unwritten — there is **no C++ test target in the module at all** (`cpp/` contains no test sources and
  `CMakeLists.txt` defines none), so `parse_vmrss_kb`'s testability is unexercised.
- (b) The desktop invariant script exists but there is no recorded run/output; it is not in `tests/`, so CI
  never touches it.
- (c) The four-point RSS table (the actual Gate 0.2 artifact) is device-only and absent.

**Drift / doubtful claims:** `IMPLEMENTATION_ORDER.md:62` calls #12 "code-complete"; against this plan's DoD
(a table with (a)-(d) deltas + a v1-optional/required recommendation) roughly one experiment and the harness
exist. The self-check boxes are correctly left unticked, but "code-complete" overstates the code.

## Cross-plan gaps for this tier

1. **No end-to-end dual-engine path exists**, so Gate 0.1 #1/#4 cannot be closed even with a real #9 package:
   the facade never sets `ORTGenerationConfig.engine`, and `LLMRepository`'s `when(type)` has no `"genai"`
   arm. This is the single highest-value fix in the tier (2 small edits: `ConfigMappers.kt:86` and
   `LLMRepository.kt:357/391/441`).
2. **The merge→load checksum contract is inconsistent** (exporter stamps map `sha256`, device merger only
   refreshes the sidecar, Kotlin precondition prefers the map). Blocks #9's native-load smoke and #19's
   train→merge→generate device checkpoint.
3. **No offline (Python) merge writer** ⇒ the offline-vs-device parity and Python atomic-overwrite tests in
   #9 are structurally unwritable; the "single merge contract, two implementations" claim rests on one
   implementation.
4. **Registry-as-data (F3) is half-applied on the device side**: EP registry unconsumed by Native; C++
   merger-variant dispatch still literal; `get_merger_type`'s `TODO` open.
5. **Fail-closed is inconsistently applied on the C++ merge path** (skip-and-continue, unconditional `true`)
   against the canonical decision that any unsatisfiable contract raises before side effects.
6. **Three inference-export paths coexist** (`gen_genai`, `inference/builder.make_genai_config`,
   `export_inference_package`) where the plan promised one; `genai_config.json` content differs by which path
   produced it, which directly affects the Gate 0.2 experiment (d) keys.
7. **No C++ unit-test harness anywhere**, so every C++ deliverable in #9/#11/#12 is "compiles + links"-tested
   only.
8. **Two gates were closed/marked ahead of their own criteria** (#10 `[x]` with #1/#4 open; #12 called
   code-complete with 2 of 4 experiments unwritten). `ACCEPTED_RSS_DELTA` and the ≥15% Gate 0.2 margin were
   never ratified as numbers.

## Remaining work, ordered

### Host-doable now (no device)
1. Set `engine = …` in `ConfigMappers.GenerationConfig.toOrt` and add the `"genai"` arm (or drop the
   `type` switch in favour of `ModelRuntime`) in `LLMRepository.kt:357/391/441`. Unblocks Gate 0.1 #1/#4.
2. Decide the post-merge checksum authority (sidecar wins after merge, or the merger rewrites the map's
   `sha256[role]`) and align `HandoffPrecondition.kt:60-69` with `weight_merger.cpp:151-155`.
3. Replace `ORTGeneratorGenAI.samplingMethodInt` with `SamplingMethod.fromWire(...).nativeOrdinal`; apply
   `seed` in `genai_runtime.cpp::nativeStart`.
4. Implement the offline merge/write driver in `artifact/merger.py` (run merger ONNX → map filenames → tmp +
   fsync + `os.replace` + `.sha256` + entry `sha256`), plus the Python atomic-overwrite test.
5. Make the C++ merge path fail closed (`save_merged_parameters` abort + `merge_and_export_weights` returning
   false); resolve `get_merger_type` from the handoff map / `mergerModels` and collapse the
   `merger_type == "lora"/…` branches to a variant→IO-name table; close `weight_merger.cpp:641`'s TODO.
6. Emit the multi-variant merger set from `export_inference_package` (mirror `emit_merger_models`'s
   `extra_methods`) so MARS packages carry their LoRA mergers.
7. Add the missing automated checks: constant-folding guard over `model.onnx` (#10), config pass-through
   smoke (#10/#12 d), the streaming-parity event-sequence harness (#11), and a JVM/`gtest`-less
   host test for `parse_vmrss_kb` (or wire a small C++ test target).
8. Emit the full `session_options.config_entries` set (external folder + `use_env_allocators` +
   `qdq_matmulnbits_accuracy_level` + the mmap keys) from one place, and make `export/pipeline.py:492` use it.
9. Deprecate/delete `gen_genai` + route `inference/builder.py`'s remaining `architectures[0]` ladder through
   the architecture registry (owner needs assigning — #7's deferral condition has expired).
10. Move mmap release out of `clearWeights()` into the cache destructor before anyone runs
    `MTF_MMAP_WEIGHTS=1`; add the Experiment (a) base-blob folder-resolution config key behind a flag.
11. Ratify `ACCEPTED_RSS_DELTA` and the Gate 0.2 ≥15% margin as written numbers.

### Device-required
- Gate 0.1 #1/#4: same #9 package under both engines + RSS comparison (`DualEngineParityTest`, after fix 1).
- #9: on-device atomic-overwrite-under-kill, offline-vs-device byte parity (needs host fix 4 first), native
  load-and-generate over a real package.
- #11: streaming parity at temp 0; merged-weights reflected per engine.
- #12: the four-point RSS table for the baseline + experiments (a)-(d), the lifetime smoke (second generation
  on an mmap-backed session), and the `use_ort_model_bytes_for_initializers` toggle smoke.

### Manual / user-run
- Build the ORT-training **Android AAR** at the pinned SHA and populate `manifest.json`'s
  `ndk_version`/`abis`/`android_api_level`/`aar_sha256` — the last open leg of Gate 0.3.
- Produce the real #9 package (`make device-package TRAIN=1`, two-profile run) that every device leg above
  depends on.
- Re-run `spikes/mmap/base_blob_mmap_spike.py` and `spikes/genai_external_swap/desktop_spike.py` on that
  package and record the outputs into the spike READMEs (Gate 0.2 evidence).

# Restructure Implementation — Handoff

**Branch:** `restructure` · **Nothing is committed** (the human commits). **Date:** 2026-07-13.

This is the running handoff for the staged restructure in `agent_docs/IMPLEMENTATION_ORDER.md`. It
records what is done, the environment/gotchas a cold agent needs, and the next steps.

---

## Status: plans #1–#8 done, #9 code-complete (global order)

| # | Plan | State |
| --- | --- | --- |
| 1 | `00_code_plans/01` package & uv scaffolding | ✅ done |
| 2 | `00_code_plans/03` dependency profiles & ORT wheel | ✅ done |
| 3 | `01_code_plans/06` source-built ORT training pipeline | ✅ done (Gate 0.3 proven) |
| 4 | `00_code_plans/02` config layering | ✅ done |
| 5 | `00_code_plans/10` code quality & module health | ✅ done |
| 6 | `00_code_plans/09` typed models/enums/registries | ✅ contract layer done; `build_merger_model` graph-collapse closed by #9 |
| 7 | `01_code_plans/05` optimum ONNX export & TasksManager | ✅ done (inference + training export proven E2E; `OnnxConfigWithLoss` vendored — see below) |
| 8 | `00_code_plans/07` weight handoff map & tensor codec | ✅ Python owner layer done (schema + codec + `check_compat`); C++/Kotlin consumers ride #9/#23/#18 |
| 9 | `01_code_plans/01` unified merger & external-data export | 🟡 code-complete 2026-07-14 (Python A/B/C tested; C++ D/E compile+link-verified on arm64-v8a). **Manual on-device tests outstanding**; native *load* side is #23. |

Per-plan detail + every deviation is in `IMPLEMENTATION_ORDER.md` under each plan's self-check block
(dated `Done 2026-07-13` notes). Read those first.

## What exists now (new/changed since the research tree)

- `pyproject.toml` (hatchling, uv), `uv.lock`, `Makefile`, `config/config.yml` (YAML-only).
- `src/mobiletransformers/`: `cli/` (argparse dispatcher + stubs), `config/` (`settings.py`,
  `constants.py` (+11 enums), `models.py` (Pydantic v2), `registry/{peft,architecture,merger}.py`,
  `resolve()`), `codegen/enums.py` (parity), `exceptions.py`, `utils/{logging,yaml}.py`,
  `_typing.py`, `py.typed`, `public_api.txt`.
- `src/mobiletransformers/export/`: `registry.py` (task discovery + `EXPORT_FRONTEND_REGISTRY`),
  `inference_export.py` (`export_inference` front door), `normalize.py`, `support_matrix.py`,
  `onnx_config_with_loss.py` (**vendored** from optimum 1.24 — see #7 note), `torch_frontend.py`
  (reserved fail-closed fallback). `spikes/optimum_migration/check_symbols.py` (the migration spike).
  `tests/export/` (11 core + 6 export-profile-only tests).
- `src/mobiletransformers/artifacts/`: `handoff_map.py` (**owns** `weight_handoff_map.json` schema +
  `TrainableTensorCodec`), `versioning.py` (canonical `check_compat` + `SchemaVersionError`).
  `tests/unit/test_handoff_map.py` + `test_tensor_codec.py`; `tests/fixtures/check_compat_cases.json`
  (shared cross-language version-gating fixture).
- `schemas/` (generated + checked in: `enums.json`, `*.schema.json`).
- `third_party/onnxruntime/{manifest.json,BUILD.md}`, `third_party/wheels/README.md`,
  `scripts/build_ort_training_*.sh`, `.github/workflows/ort-training-smoke.yml`.
- `tests/{unit,integration,fixtures}/` — 71 unit/fixture tests + 1 skipped integration (training).
- 11 Kotlin enum mirrors under
  `android/ORTransformer/ORTransformersMobile/src/main/java/com/martinkorelic/ortmobile/constants/`.
- Legacy root packages (`trainer/`, `artifact/`, `inference/`, `tools/`, `database/`, …) are
  **untouched except**: `config.py` + `tools/parser_config.py` are now deprecation shims, and each
  monolith carries a `# DECOMPOSE(#5):` note. Every old import still resolves.

## Environment & how to run (READ THIS — non-obvious)

- **`uv` is the entrypoint** (installed at `~/.local/bin/uv`, 0.11.28). Two Pythons: system **3.10.12**
  (core/dev default) and **3.12.11** (`/usr/bin/python3.12`, required for the training profile).
- **Core/dev:** `uv sync --frozen --group dev` then `make check` (lint + typecheck + parity + 71 tests).
- **Training profile (cp312 only):** `uv sync --python 3.12 --group ort-training-local` then
  `make test-train`. The source-built `onnxruntime-training==1.23.0+cpu` wheel lives (git-ignored) at
  `third_party/wheels/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl`
  (copied from the sibling repo `../on_device_llm_finetune/dist/`; SHA in `manifest.json`).
- **Profile isolation is enforced:** `uv sync --extra export --group ort-training-local` **errors by
  design** (`[tool.uv] conflicts`) — the `onnxruntime` providers must never co-install.
- **`uv run` mutates `.venv`.** Running a `uv run` with different `--group`/`--extra` re-syncs the shared
  `.venv`; if you interleave training and core runs you can end up with `torch` present but `numpy`
  upgraded → onnxruntime "import numpy failed". Fix: `uv sync --frozen --group dev` to reset to a clean
  core env. Always pass explicit `--group`/`--python` when running training tests.

## Gotchas discovered (all handled; don't re-learn them)

1. **cp312-only wheel** → the `ort-training-local` group and its `numpy<2` / `onnx<1.19` pins are all
   marked `python_version == '3.12'` so the universal `uv lock` resolves across 3.10–3.13.
2. **Real torch ABI is 2.7.1** (docs guessed 2.5.1) — the "one unknown" was already resolved by the
   existing build. Recorded in `manifest.json`.
3. **numpy<2** required (wheel built against numpy 1.26 ABI) and **onnx<1.19** required (ORT 1.23
   runtime caps ONNX IR at 11; onnx≥1.19 emits IR 13 → optimizer model won't load). Both pinned to the
   3.12 training fork; export/other profiles keep numpy/onnx 2.x via uv's forked resolution.
4. **Heavy profiles need Python ≥3.11** — `optimum-onnx` pulls `onnxruntime≥1.24`, which dropped cp310
   wheels. Core/dev stay 3.10-clean; validate `export`/`genai`/`train` under 3.12.
5. **`export-rocm` is an empty placeholder** — ROCm wheels need a dedicated AMD index (deferred).
6. **langchain-objectbox 0.1.0** declares a stale `langchain-core<0.2.0`; overridden to the proven
   `>=0.3.74,<0.4` via `[tool.uv] override-dependencies` (matches the working env).
7. **Grep guards match docstrings** — the secret-guard and dispatch-guard patterns will hit prose that
   *mentions* the banned pattern; keep docstrings from spelling out `os.environ['X']` / `x == "lora"`.
8. **optimum 2.1 removed `OnnxConfigWithLoss`** (the optimum-onnx split); `export()`, `main_export`,
   `TasksManager`, `*OnnxConfig` all survive. We **vendor** the wrapper in
   `export/onnx_config_with_loss.py` (deps still present). Don't re-add `from optimum... import
   OnnxConfigWithLoss` — it will fail. **TasksManager discovery** returns empty unless you first
   `import optimum.exporters.onnx.model_configs` (decorator registration) and pass
   `library_name="transformers"` — both handled inside `export/registry.py`.
9. **`.venv` Python version flips with profiles.** `uv sync --extra export` needs ≥3.11 and rebuilds
   `.venv` on **3.12**, leaving a numpy 2.x that makes `make check`'s mypy (target 3.10) fail on
   numpy's 3.12-only stub syntax. Reset the core/dev env explicitly on 3.10:
   `uv sync --frozen --group dev --python 3.10`. Then `make check` is clean.

## Next steps for the next agent (in order)

Follow `IMPLEMENTATION_ORDER.md` "How to execute a plan" protocol. **Confirm scope with the human
before starting** if it spans Android/C++ or later-plan territory.

### #7 — DONE. Optimum export front door (`01_code_plans/05`)
The architecture registry's lazy `*OnnxConfig` dotted paths **do resolve** under optimum 2.1 (the export
profile), and discovery/task-selection is wired via `TasksManager` + `EXPORT_FRONTEND_REGISTRY` with no
`architectures[0]` ladder. The `trainer/builder.py` ladder is replaced by `resolve_architecture()` +
`choose_task`. Both export paths proven E2E on real models. See the #7 self-check note in
`IMPLEMENTATION_ORDER.md` for the full record. **The one thing #8/#9 must inherit:** the normalized
package emits **canonical HF KV/IO names** (`input_ids`/`attention_mask`/`position_ids` +
`past_key_values.<i>.key/value` in, `logits` + `present.<i>.key/value` out) matching
`inference/builder.py` `make_genai_config`, flat with one `model.onnx_data`. The `weight_handoff_map.json`
in #8 must key its tensor identity onto exactly these names.

### #8 — DONE. Weight handoff map & tensor codec (`00_code_plans/07`)
Python owner layer complete: `artifacts/handoff_map.py` owns the `weight_handoff_map.json` schema +
`TrainableTensorCodec`; `artifacts/versioning.py` owns the canonical `check_compat`. See the #8
self-check note in `IMPLEMENTATION_ORDER.md`. **What #9 must inherit:** build a `HandoffMap` via
`TrainableTensorCodec.from_peft_mapping(...)`, `validate()` it (fail-closed invariants incl. the
quantized-name bug), and emit `weight_handoff_map.json` with `HandoffMap.save()`; the merger's on-device
external-data layout uses `entry.externalDataLocation` / `entry.mergedTensorNames`, and the merger ONNX
filenames go in `HandoffMap.merger_models` keyed by `MergerVariant`. `check_compat` + `SchemaVersionError`
are reusable by any versioned contract (manifest #13, support matrix #20, federated #35).

### #9 — `01_code_plans/01_unified_merger_and_external_data_export.md` (next)
**Dual-engine core + finishes #6's merger work.** Implement the single `build_merger_model(MergerSpec)`
ONNX-graph builder (currently a fail-closed stub in `config/registry/merger.py`) collapsing
`artifact/merger.py`'s four `create_*_merger_model{,_2}` factories, with the golden-equivalence test.
Emits the external-initializer package + `weight_handoff_map.json` (via #8's codec/`HandoffMap.save`).
The C++ `weight_merger.cpp` dispatch to the handoff map/registry is the Android-side half — **confirm
scope with the human**: the Python graph-builder + emit is the automatable core; the C++ merge/save
rewrite is device-tested. `resolve_merger`/`MergerSpec` already exist and are tested (#6).

### #9 — `01_code_plans/01_unified_merger_and_external_data_export.md`
Dual-engine core + **finishes #6's merger work**: implement the single `build_merger_model(MergerSpec)`
ONNX-graph builder (currently a fail-closed stub in `config/registry/merger.py`) collapsing
`artifact/merger.py`'s four `create_*_merger_model{,_2}` factories, with the golden-equivalence test;
wire the C++ `weight_merger.cpp` dispatch to the handoff map/registry. `resolve_merger`/`MergerSpec`
already exist and are tested.

### Then #10 → #22 (Tier 1) per the table
#10 GenAI swap spike → #11 engine abstraction → #13 manifest/cache → #14 hub format → **#15 export E2E
checkpoint** → #16 Android Gradle rename (isolated!) → #17 facade → #18 training lifecycle → #19 HF
Kotlin facade → #20 support matrix → #21 hub pull → #22 adapter pushback. Tiers 2/3 + release
(#23–#37) follow.

### Cross-plan debts left for their owners (don't lose these)
- **Legacy dispatch rewrites** (`trainer/builder.py`, `inference/builder.py`, `artifact/onnx_builder.py`)
  → registries: ride with #7 (training) / gated for inference by the Optimum-vs-GenAI decision.
- **`build_merger_model` graph collapse + C++ merger** → #9.
- **Kotlin typed-config field swaps** (`ORT*Config.kt`, `FileUtil.kt fromWire`) → #17/#19, after the
  Android rename #16. Enum **mirrors** already exist and pass parity.
- **CI wiring** (`make lint/typecheck/parity` gates, the ORT-training smoke wheel provisioning) → #29.
  The `.github/workflows/ort-training-smoke.yml` is `workflow_dispatch`-only and self-skips without the
  (git-ignored) wheel until #29 decides how CI gets it.
- **Broaden `mobiletransformers.__all__`** to export config models/enums/registries → once #7/#9
  exercise them; finalized at #32. Update `src/mobiletransformers/public_api.txt` when you do (a test
  guards it).

## Session close (2026-07-13, cont.)

Landed **#7** (optimum export front door) and **#8** (weight handoff map + tensor codec) this session —
both the Python owner layers, fully tested. Highlights for the next agent:
- **optimum 2.1 migration is resolved**: `OnnxConfigWithLoss` is vendored (`export/onnx_config_with_loss.py`);
  don't reintroduce the optimum import. See the #7 plan doc's "Implementation notes".
- **`check_compat` + `SchemaVersionError`** now exist as a reusable helper (`artifacts/versioning.py`)
  with a shared fixture (`tests/fixtures/check_compat_cases.json`) — use them for the manifest (#13),
  support matrix (#20), and federated (#35) contracts; don't re-implement version gating.
- **`HandoffMap`/`TrainableTensorCodec`** (`artifacts/handoff_map.py`) is the emit/validate contract #9
  builds on (`from_peft_mapping` → `validate` → `save`).
- Full detail is in each plan's "Implementation notes" section + the per-plan self-checks in
  `IMPLEMENTATION_ORDER.md`.

**Repo left clean:** env reset to core/dev on Python 3.10 (`make check` green: 108 passed, 6 skipped);
`uv lock --check` + `uv build --wheel` clean; `dist/` removed. **Nothing committed** (human commits).
**Next: #9** (`01_code_plans/01`) — Python merger graph-builder + emit is the automatable core; the C++
`weight_merger.cpp` rewrite is device-tested — confirm scope before starting the Android/C++ half.

## Session close (2026-07-14)

Landed **#9** (unified merger & external-data export) — **full scope incl. the C++ half** (user-confirmed).
Code-complete: Python A/B/C fully tested (core gate green: **122 passed, 8 skipped**; numerical merger
tests pass under the export profile), C++ D + Kotlin E compile+link-verified. Details in the #9 self-check
note in `IMPLEMENTATION_ORDER.md`. Highlights for the next agent:
- **`build_merger_model`** (`config/registry/merger.py`) is wired; `tests/unit/test_merger_builder.py` pins
  it byte-for-byte to committed goldens (`tests/fixtures/gen_merger_golden.py` → `merger_golden/`).
  Regenerate goldens with `python tests/fixtures/gen_merger_golden.py` (core env, onnx-only; it stubs the
  legacy module's unused onnxruntime import).
- **`inference/export_inference_package.py`** is the single export entry: base/trainable external split +
  `weight_handoff_map.json` emit (via #8's codec) + `mergerModels` + `genai_config` session entry. Uses it,
  don't add a second handoff_map module. `model_input`/`adapter` modes fail closed (v1).
- **C++ `weight_merger.cpp`**: `load_handoff_map` + a C++ `check_compat` mirror; save writes raw bytes to
  `externalDataLocation[role]` (atomic rename + `.sha256`; compact SHA-256 host-verified). `inference_name`
  string-rewrite deleted.
- **Native build deps are untracked & were missing from this checkout** — `cpp/includes/google` (protobuf
  headers), `jniLibs/`, `aarLibs/`. All `.gitignore`d (added `aarLibs/` this session). I provisioned them
  locally from the sibling **`../ORTTransformer`** checkout to run the arm64-v8a native build. **They are
  intentionally NOT committed** (policy: large vendored native artifacts stay out-of-band, like the ORT
  wheel). A provisioning README for these belongs with the Android rename/CI plans (#16/#29). x86_64 native
  link can't be verified here — the source repo's `jniLibs/x86_64` is itself incomplete.
- **Outstanding for #9 (manual/device):** on-device atomic-overwrite-under-kill, offline-vs-device
  byte-identical `.bin` parity, native load-and-generate smoke. The native **load** side
  (`ORTGeneratorNative.loadMergedWeights` / `session_cache.h`) still probes `inference/merged` — its
  migration to the handoff map is **#23** (flagged `DECOMPOSE(#23)` at both sites); until then a device
  build won't *see* the in-place merges. Do #23 before expecting merged-weight generation on device.

**Repo left clean:** env reset to core/dev on Python 3.10 (`make check` green: **122 passed, 8 skipped**);
`uv lock --check` + `uv build --wheel` clean; `dist/` removed. **Nothing committed** (human commits).
**Next: #10** (`01_code_plans/02`, GenAI external-data swap spike) — or complete #9's manual device tests +
#23 (native load) first if a real merged-weight device run is needed.

## Validation commands (paste-run)
```bash
uv sync --frozen --group dev && make check          # lint + typecheck + parity + 71 tests
uv run python -m mobiletransformers.codegen.enums --check   # cross-lang enum/schema parity
uv sync --python 3.12 --group ort-training-local && make test-train   # ORT toolchain alive (Gate 0.3)
uv build --wheel                                    # wheel = only src/mobiletransformers (+ py.typed, public_api.txt)
uv lock --check                                     # lock consistent with pyproject
```

## Session close (2026-07-14, cont. — #16 Android rename)

Landed **#16 as a FULL rename (option B)** — user-confirmed, superseding the doc's isolate-only option A.
Everything renamed off the legacy brand; **zero `ortmobile`/`orttransformer`/`ORT(T)ransformer` in live
code/config** (historical `agent_docs/*` migration docs intentionally left as records — 19 files).
- **Structure:** `android/ORTransformer` → `android/MobileTransformersApp`; SDK module
  `ORTransformersMobile` → `MobileTransformers` (`:MobileTransformers`); sample `app` unchanged.
- **Packages:** SDK `com.martinkorelic.ortmobile` → `com.martinkorelic.mobiletransformers`; app
  `com.martinkorelic.orttransformer` → `com.martinkorelic.mobiletransformers.app` (+ matching `applicationId`).
- **Native (lockstep):** `libmobiletransformers.so`, CMake `project("mobiletransformers")`,
  `loadLibrary("mobiletransformers")`, all 22 JNI symbols (SDK→`Java_com_martinkorelic_mobiletransformers_*`,
  MainActivity→`..._mobiletransformers_app_MainActivity_*`).
- **Python lockstep:** `codegen/enums.py::KOTLIN_CONSTANTS_RELPATH` (new path), tokenizer file
  `mobiletransformers_tokenizer_config.json` (writer+reader), `ORTransformerGenerator`→`MobileTransformerGenerator`,
  `ORTMobileObjectBoxProcessor`→`MobileTransformersObjectBoxProcessor`, and the `evaluation/mobile/*.sql`
  process names → new applicationId.
- **Verified:** arm64-v8a native links `libmobiletransformers.so` (JNI symbols confirmed via `llvm-nm`);
  `:MobileTransformers:compileDebugKotlin` + `:app:compileDebugKotlin` green; `make parity` OK (walks the
  new Kotlin constants path); `make check` green (122 passed, 8 skipped). x86_64 native link still blocked
  by the upstream-incomplete `jniLibs/x86_64` (pre-existing). Full `assembleDebug` + device install are manual.
- **Note:** directory moves used plain `mv` (not `git mv`) so the untracked vendored deps
  (`jniLibs/`, `cpp/includes/google`, `aarLibs/`) travelled with the tree; git will see renames at commit.
  **Nothing committed** (human commits). **Next: #13** (manifest/cache bridge) or **#23**+#9 manual device
  tests — see `plans/read-through-the-recent-sunny-turtle.md` "Next phases".

## Session close (2026-07-14, cont. — #14/#13/#15/#20 package+export+matrix phase)

Landed a **four-plan no-device phase** (user-confirmed: Python + Kotlin, no physical device). Full Python
gate green (**176 passed, 8 skipped**); Kotlin `compileDebugKotlin` (both modules) + **10 JVM tests** green.
- **#14** `src/mobiletransformers/hub/package_format.py` — owns the manifest schema: `sanitize_repo_id`,
  `build_manifest` (stream-hashed `sha256`/`fileSizes` + `downloadPlan`), `write_manifest`/`write_variant_checksums`.
  Committed shared fixture `tests/fixtures/tiny_package/` (+ generator `make_tiny_package.py`) and
  `tests/fixtures/sanitize_repo_id_cases.json`.
- **#13** Python `src/mobiletransformers/artifacts/manifest.py` (`MobileTransformersManifest.validate` +
  `select_variant`, reusing `versioning.check_compat` w/ `MANIFEST_READER_VERSION`); Kotlin cache-bridge
  `android/.../mobiletransformers/packages/` — `PackageFormat` (sanitize + `checkCompat` mirrors),
  `MobileTransformersManifest`, `ManifestValidator`, `VariantSelector`, `ChecksumVerifier`,
  `ModelPackageInstaller` (atomic `renameTo`), `CacheIndex`. `LLMRepository` untouched (#13's design).
  Cross-language parity pinned by the shared `check_compat_cases.json` / `sanitize_repo_id_cases.json`.
- **#15** `export/pipeline.py` (`plan_export`/`export_package`/`assemble_package`) + `export/model_card.py`
  + `cli/export.py` + `cli/push.py` wired into the dispatcher. The export-E2E **checkpoint automated leg**
  (dry-run plan + `assemble_package` → validates against #13) is CI-covered; the **real full-model export**
  (`create_model`/`gen_artifacts`) is **env-gated** (optimum + ORT-training profiles) — `_full_export`
  raises a clear message until run under those profiles. NOT device-gated.
- **#20** `src/mobiletransformers/support/` (`statuses`/`models`/`matrix`) + `cli support-matrix`. Detection
  (`AutoConfig`/`TasksManager`) is injectable (mocked in CI); the three `android_*`/`rag` ready-statuses
  read `android_probes.json` and honestly degrade to `false`+blocker when absent.
- **Makefile** `test` target extended to collect `tests/hub tests/support tests/cli`.
- **Deferred (not this phase):** #15 real full-model export (env-gated); the single on-device
  install→generate smoke (#13); #10/#11/#12 (device/GenAI-gated engine track); #21 hub pull + #22 adapter
  push-back (Python-first, now unblocked by #13/#14). **Nothing committed** (human commits).

## Session close (2026-07-14, cont. — #21/#22 Hub round-trip)

Landed the **Python Hub round-trip** (Python-first, no device). Full gate green (**196 passed, 8 skipped**).
- **#21** `src/mobiletransformers/hub/variant_select.py` (`Constraints` + `select_variant`: soft quant
  preference, download-size tie-break, 0.9× storage-budget ceiling, layered over #13's hard-filter) +
  `hub/pull.py` (`pull_package` manifest-first `snapshot_download` + sha256-verify; `install_package` =
  the Python cache-bridge — mirror of #13's Kotlin `ModelPackageInstaller` — with tokenizer flattening +
  atomic `.partial`→`os.replace`). `cli/pull.py` registers `pull` + `install-package`. `downloader`
  injectable → pull/install smokes run offline over `tests/fixtures/tiny_package`.
- **#22** `src/mobiletransformers/adapter/{export,convert,model_card}.py` + `cli/push_adapter.py`.
  `export_adapter_from_cache` (pure) → `AdapterPackage`; `to_peft_layout` gate (pure metadata): clean LoRA
  → Mode-1 `adapter_config.json`, else (all MARS, factor-less LoRA) → Mode-2 native subtree +
  `mobiletransformers_adapter.json`; `--peft-only` errors. Card carries a bold privacy warning + exact
  base license, asserted before upload; `uploader` injectable.
- **Guard note:** `test_no_string_literal_dispatch_in_src` bans `peft_method == "..."` — used
  `PEFTMethod.MARS.value`/`.LORA.value` comparisons instead of raw literals.
- **Makefile** `test` now also collects `tests/adapter`.
- **Deferred:** #21 Android downloader (OkHttp/WorkManager) + `fromPretrained` device load; #22 PEFT
  `adapter_model.safetensors` materialization (torch/peft, `train` extra — `convert.materialize_peft_weights`
  raises until run under that profile) + on-device `AdapterUploader.kt`. **Nothing committed** (human commits).

**Tier-1 Python surface is now complete** (#1–#9 done/code-complete, #13/#14/#15/#16/#20/#21/#22 done).
Remaining: the device track (#10/#11/#12 engine, #23 native load, on-device #9 manual tests) and the
Android facade (#17/#19), all gated on device testing; then Tier-2/3 (#23–#37).

## Session close (2026-07-14, cont. — #28/#29/#25 release-foundation + RAG boundary)

Landed a **no-device release-foundation + RAG-boundary phase** (user-confirmed bundle). All three legs
verified: Python `make check` **196 passed, 8 skipped**; Kotlin `:MobileTransformers` + `:app`
`compileDebugKotlin` green; **23 JVM unit tests** green (`JAVA_HOME=/opt/android-studio/jbr`, JDK 17;
Android SDK at `~/Android/Sdk`).
- **#28** Real root `Makefile` (replaces the #5 stub) — thin wrappers over the `mobiletransformers` CLI
  + Gradle; `make help` self-documents; profile-isolated `setup`/`setup-export`/`setup-train`/`setup-genai`;
  non-destructive `clean-generated`. New `scripts/{android_build_aar,publish_local_maven,run_smoke}.sh`
  (fail-closed stubs; the AAR/Maven bodies are #30's). Console-script already wired in `pyproject.toml`.
- **#29** `.github/workflows/ci.yml` (fast → export-smoke → android-assemble; `fail-fast:false`,
  per-job `timeout-minutes`) + `device.yml` (`workflow_dispatch` + nightly `schedule`). Fast runs
  lint/typecheck/parity + tests; export-smoke installs the export profile (3.12) and runs the
  wiring smoke. **android-assemble self-skips** without the git-ignored vendored native deps
  (`aarLibs/`,`jniLibs/`), mirroring `ort-training-smoke.yml` — it is authored + gated, not proven on a
  hosted runner. YAML parse-verified. The CI-provisioning of those native deps + the ORT wheel stays
  the open question (tied to #30).
- **#25** New `com.martinkorelic.mobiletransformers.rag` package: `VectorStore` (+ `RagDocument`/`RagMatch`),
  `ObjectBoxVectorStore` (wraps `ORTVectorDatabase`, preserves COSINE / `1 - distance` similarity /
  `minScore` / embedding-strip / non-ranked text path), test-only `InMemoryVectorStore` (pure cosine),
  `DimensionRegistry` (single declared dimension source — `ORTVectorDatabase.SUPPORTED_DIMENSIONS`
  delegates to it; fail-closed on unsupported), `VectorStoreRegistry` (F4 pluggable backends, `objectbox`
  default). `ORTRetriever.query` now routes through the boundary; **`RagResult.documents` migrated
  `List<Pair<VectorEntityInterface,Double>>` → `List<RagMatch>`** and its one consumer (sample-app
  `InferenceViewModel`) updated in lockstep (`RagMatch` destructures as `(document, score)`).
  - *Gotcha:* Android unit-test classpath has **no `kotlin-reflect`** — a `::class.members` assertion
    threw `KotlinReflectionNotSupportedError`; replaced with a reflection-free behavioral check
    (returned `RagMatch.document` equals the inserted `RagDocument`, proving no embedding leaks).
  - #17 is a *nominal* prereq only — #25 wraps existing RAG classes and needs no facade code; done ahead
    of #17 as a no-device slice (consistent with the prior no-device phases).
- **Deferred:** #25 ObjectBox parity smoke (Android, manual); `SearchType` String→enum swap in
  `ORTRagConfig` rides with #17/#19; `docs/EXPORT.md`/`docs/RAG.md` are #31. **Nothing committed** (human commits).

**No-device surface now covers #28/#29 (release foundation) + #25 (RAG boundary) on top of Tier-1.** The
next genuinely-CI/no-device candidate is thin (parts of #31 docs); everything else substantial —
#10/#11/#12 engine, #17/#18/#19 facade, #23 native load, #24 sampling, #26/#27 RAG ingest/grounded,
#30 AAR consumer build — needs a device or the on-device engine, which the user has deferred.

## Session close (2026-07-14, cont. — #31 docs (partial) + Mac work plan)

Landed the **contract-locked slice of #31** (pure Python/markdown, no device) plus a Mac cold-start
guide, ahead of the user moving to a **macOS box without a device** for a few days. Python gate green
(**198 passed, 8 skipped** — 2 new render tests).
- **#31 (partial)** — `docs/EXPORT.md` (#15/#2 CLI + profiles; flags verified against `cli/export.py` +
  `cli/push.py --repo`/`cli/pull.py --repo-id`), `docs/RAG.md` (scoped to the locked #25 `VectorStore`
  boundary; #26 ingestion + #27 grounded-gen marked not-yet), `docs/PUBLIC_API.md` (Python `__all__`
  from `public_api.txt` + CLI subcommands; Kotlin facade pending #17/#19). New `support/render.py`
  (`render_matrix_markdown`) + `support-matrix --md` flag emit **`docs/COMPATIBILITY_MATRIX.md`
  rendered from the matrix** (F6, axis legend enumerated from the #6 enums so it can't drift); the
  committed doc is a network-free representative sample from
  `tests/fixtures/gen_compat_matrix_doc.py`, drift-guarded by `tests/support/test_render.py`
  (regenerate live under the export profile). `CHANGELOG.md` + `docs/RELEASE_CHECKLIST.md` skeletons
  created (finalized by #32). #31 box stays **open** — device/facade-gated pages
  (ARCHITECTURE/MODEL_FORMAT/CONFIGURATION/ANDROID_SDK) + the CI link-check remain.
- **`agent_docs/MAC_WORKPLAN.md`** (expanded) — now carries **prerequisites** (Track A Python-only;
  Track B full Android with pinned versions: AGP 8.5.1 / Kotlin 1.9.0 / Gradle 8.7 / compileSdk 34 /
  NDK r26.x / CMake 3.22.1 / JDK 17), a **"what to move to the Mac"** table (the Android vendored deps
  `aarLibs`/`jniLibs`/protobuf headers are host-OS-agnostic → copy via Drive/zip and Android builds work;
  the ORT-training wheel is linux/cp312 → cannot move), and a **per-plan coverage map**. Headline: with
  Track B, ~80% of the remaining implementation across Tiers 1–3 is Mac-doable (implement + compile/type +
  unit/integration mock tests); the **device is only the final acceptance leg** per plan. Genuinely
  out of reach on Mac: real ORT training-artifact gen / real generate (Linux/device), and the #10/#12
  measurement spikes. **Top Mac tasks:** #22 `materialize_peft_weights` (cache `.bin`→safetensors, torch
  CPU — **no ORT-training needed**, the stub docstring is pessimistic), #31 `MODEL_FORMAT.md`+
  `CONFIGURATION.md` (locked), and **#35 federated Flower sim (entirely host Python)**.

**Nothing committed** (human commits) — and note the Mac plan hinges on committing + pushing this work
first, since the tree is the only copy.

## Session close (2026-07-15 — #22 finish + #31 docs + #35 federated + #17/#18 Android facade)

Landed a **five-plan "everything host-doable" phase** (user-confirmed) on the **Linux** box (Android
toolchain + ORT-training wheel both present here — the `MAC_WORKPLAN` was written ahead of a move that
hadn't happened). All legs verified host-side: Python `make check` **211 passed, 10 skipped**; enum parity
OK; `uv lock --check` clean; Android `:MobileTransformers:testDebugUnitTest` **41 JVM tests green** +
`:MobileTransformers`/`:app` `compileDebugKotlin` green. Only physical-device legs deferred.

- **#22 DONE** — `adapter/convert.py::materialize_peft_weights` implemented (was the last stub). It reads the
  LoRA A/B factors from the ORT `CheckpointState` (`train/checkpoint`, mirrors `onnx_transfer_trained_weights`)
  and writes `adapter_model.safetensors` (numpy→torch→`safetensors.torch.save_file`) with PEFT keys
  (`base_model.model.<module>.lora_A/B.weight`). **The factor read is injectable** (`factor_reader`) so the
  numpy→torch→safetensors + key-mapping path is unit-tested (`tests/adapter/test_convert.py`, behind
  `importorskip torch/safetensors`) **without** building a real checkpoint; the default reader stays
  env-gated on `onnxruntime-training`. Verified under the `train` profile (3.12): 2 materialize tests pass.
  The clarifying finding: the A/B factors live in the CheckpointState, **not** as flat `.bin` in `inference/`
  (those are the merged tensors) — the MAC note's "raw .bin" was optimistic, but it's still no-device on this
  Linux box.
- **#31 (still partial)** — added the two contract-locked pages: `docs/MODEL_FORMAT.md` (manifest +
  `weight_handoff_map.json` on-disk contract, sourced from `hub/package_format.py`/`artifacts/manifest.py` +
  `artifacts/handoff_map.py`) and `docs/CONFIGURATION.md` (enum vocab, Pydantic cross-boundary models, the
  three registries as extension points). `docs/ARCHITECTURE.md` + `docs/ANDROID_SDK.md` still await #23/#24/#30.
- **#35 code-complete (box open)** — new `src/mobiletransformers/federated/`: `adapter_record.py`
  (`FederatedAdapterRecord`, a thin wrapper over `TrainableTensorCodec`/`HandoffMap` — invents no ordering;
  pinned byte serialization = uint32 LE header len + JSON header + codec-order payloads, the #36 golden),
  `flower_sim.py` (pure `federated_average` FedAvg + `save_global_adapter` + a lazy-flwr `run_simulation`
  manual leg), `flower_client.py` (lazy-flwr `ClientApp`/`ServerApp` + ORT `fit` seam — manual leg),
  `cli/federated.py` (`mobiletransformers federated simulate`, wired into the dispatcher). Tests
  `tests/federated/` (roundtrip/format-version/comm-size/fedavg/dropout + a committed byte golden
  `fixtures/federated_record.golden.bin`, regen: `python -m tests.federated.gen_serialization_golden`) all
  run in the **core** env (pure numpy). **Flower is deliberately OUT of the universal `uv.lock`** — adding
  `flwr[simulation]` (ray/pyarrow) downgraded protobuf 7→6 / rich / typer repo-wide **and** bumped mypy
  (1.11→1.19, which broke `make check`'s yaml typecheck). So flwr is out-of-band like the ORT wheel:
  `pip install "flwr[simulation]"` for the manual sim leg (documented in `pyproject.toml` + the
  `run_simulation` error). **Deferred (manual):** the real N-client ORT-`fit` sim + aggregated-adapter
  logits-differ smoke (runnable here under the ORT-training profile, but treated as user-run).
- **#17 code-complete (box open)** — Android facade foundation under
  `android/.../com/martinkorelic/mobiletransformers/`: `MobileTransformers.fromPretrained` +
  `MobileTransformerModel` (train/merge/generate/retrieve/capabilities/close), public `config/PublicConfigs.kt`
  + `runtime/{ModelSession,RuntimeCapabilities,Results,InferenceEngine}.kt`, `packages/ModelFeature.kt`,
  `MobileTransformersException.kt`, `internal/config/ConfigMappers.kt` (`toOrt()`),
  `internal/runtime/RepositoryBackedModelSession.kt` (wraps the existing `LLMRepository` + 3 sub-repos;
  callback→result adaptation via `CompletableDeferred`). **`InferenceEngine` is a placeholder flagged
  `DECOMPOSE(#11)`** — #11 owns it; do not add a second engine enum. JVM tests (`src/test/.../facade/`):
  config-mapper round-trip (defaults == `ORT*Config()`), feature/engine semantics, manifest variant-select,
  and facade→session delegation via a hand-written fake `ModelSession` (no mock framework on the classpath).
  **Deferred (device):** real `fromPretrained→generate` on a device.
- **#18 code-complete (box open)** — new `training/` package: `TrainingJob`/`TrainingStatus`/`TrainingEvent`/
  `CheckpointInfo`/`TrainingJobManager` (+`TrainingJobSpec` WorkManager seam, **no WorkManager dep added**) +
  `TrainingEventAdapter` (callback→status/event mapping). Light edits: `ORTTrainerNative` gained
  `@Volatile cancelRequested` checked at the epoch/step loop tops (cooperative cancel; **no `TrainingState`/
  `SchedulerState`/JSON format change**). `TrainingResult` (in `runtime/Results.kt`, the #17 type) enriched
  with `checkpoint`/`summary`. JVM tests: adapter status/event mapping (scripted callbacks) + `CheckpointInfo`
  round-trip (format preserved). **Deferred (device):** resume-no-double-count, profileMetrics summary, the
  instrumented train→merge→generate smoke; `TrainingEvent.Metric` is defined but not emitted (the
  `TrainingCallback` surface carries no metrics — honest, not silently faked).

**Boxes:** #22 is genuinely done. #31 stays *partial*. #35/#17/#18 are **code-complete with device/manual
legs open** — leave `[ ]` (like #9) until their manual legs run. **Nothing committed** (human commits).
**Next candidates:** run the deferred #35 sim under the ORT-training profile here; or #23 (native load) /
#24 (sampling) / #26–#27 (RAG) for continued host-doable Kotlin; device legs when a device is available.

*(Housekeeping: `HANDOFF.md`'s earlier dangling ref to `plans/read-through-the-recent-sunny-turtle.md`
never existed — the real order lives in `IMPLEMENTATION_ORDER.md` "Global order".)*

## Session close (2026-07-15, cont. — #10 GenAI external-data-swap spike / Gate 0.1)

Ran the **#10 spike** on the real `onnxruntime-genai-android-0.14.0.aar` + a connected device (Galaxy S21
FE, SM-G990B, arm64). Full record: `spikes/genai_external_swap/README.md`. **Verdict: Gate 0.1 GenAI side
PASSES on device.** F2 (external-data swap) is validated, and the one real blocker — ORT-runtime coexistence
(genai needs stock ORT ≥1.26, Native needs the training ORT 1.23) — was **RESOLVED** via engine separation
(distinct-soname `libort_gen.so` for GenAI; training ORT stays `libonnxruntime.so`). Both coexist on device.

- **Symbol check (#6/#8): PASS** — `spikes/genai_external_swap/check_symbols.sh` on the AAR arm64 `.so`:
  `OgaCreateModelWithInitializers` ABSENT (fork-only confirmed), `OgaCreateModel` present, 23 `OgaGenerator*`.
- **Desktop swap smoke (#2/#3): PASS** — built a tiny real GenAI model
  (`build_tiny_genai_model.sh` → SmolLM2-135M, standalone `.venv-genai-spike`) and `desktop_spike.py` shows
  overwriting external weights changes GenAI logits on a fresh `og.Model()` (`|ΔL|=39.6`); RSS +144 MB on a
  199 MB blob = mmap/lazy, not 2× copy.
- **Device build/link/install: PASS** — `genai_spike.cpp` (JNI, stable C API) + `GenAISpike.kt` +
  `GenAISpikeTest.kt` compile, link against the real genai `.so`, package, and install on device.
- **Device generate: ✅ PASS (RESOLVED)** — `GenAISpikeTest` passes on device: `OgaCreateModel` loads the
  model, generates a token (relative external data resolved → #5), and overwriting one external weight
  changes the output on a fresh model (token 28→6156, fp 1.518e8→9.82e7 → #2/#3). **Both ORTs coexist in one
  process.**
- **Root cause + fix — ORT engine separation.** The genai AAR ships **no `libonnxruntime.so`**; GenAI
  `dlopen`s the app's. The app ships the **source-built ORT-training 1.23**, but **genai 0.14 needs stock ORT
  ≥1.26** (pip meta `onnxruntime>=1.26.0`; desktop worked with 1.27.0) → SIGABRT. Fix (reproducible via
  `spikes/genai_external_swap/setup_ort_separation.sh`): ship GenAI its own stock ORT **1.27** as
  `jniLibs/<abi>/libort_gen.so` (SONAME **raw-patched** — NOT `patchelf`, which corrupts `verneed`), and
  raw-patch the genai `.so`'s `dlopen` target `libonnxruntime.so`→`libort_gen.so`. Training ORT stays
  `libonnxruntime.so` (linked by `libmobiletransformers.so`). Safe because each ORT exports only ~3 symbols
  (hidden visibility) and genai resolves ORT via `dlsym` on its own handle → no interposition; the **distinct
  SONAME is essential** (same soname → linker dedup → genai gets the training lib back, observed).
- **Setup changes (vendored, git-ignored):** real AAR at `aarLibs/onnxruntime-genai.aar` (was a 1.3 MB stub);
  `jniLibs/arm64-v8a/{libonnxruntime-genai.so (dlopen-patched), libort_gen.so (stock 1.27, soname-patched)}`;
  `cpp/onnxruntime-genai/*.h` ← AAR clean upstream headers (`.fork.bak` kept); dead `ORTGenAITokenizer.kt`
  reduced to a compiling stub (DECOMPOSE(#11)). **The genai AAR is NOT a Gradle dependency** (Java unused);
  the patched genai `.so` ships from `jniLibs`. `build.gradle.kts` keeps a harmless `packaging{ jniLibs
  pickFirsts }`. **Nothing committed** (human commits).
- **Next:** #11 (inference engine abstraction) is now unblocked *and* has a proven ORT-coexistence design to
  build on — promote `genai_spike.cpp` into the `ModelRuntime` GenAI engine, keep `libort_gen.so`/patched
  genai from `setup_ort_separation.sh`. Remaining Gate 0.1 legs #1/#4 (cross-engine equivalence + RSS) need a
  real File #9 package (per-tensor externals) so Native + GenAI read the SAME folder — rides with #9's
  real-export leg.

## Session close (2026-07-15, cont. — #11 inference engine abstraction)

Landed **#11** (dual-engine `ModelRuntime`) on top of the proven ORT separation. Compiles + links (arm64),
**48 SDK JVM tests green**, device build loads and GenAI still works (`GenAISpikeTest` OK). New JNI symbols
(`ORTGeneratorGenAI_native*`, `GenAiSupport_nativeGenAiAvailable`) exported.
- **New:** `runtime/ModelRuntime.kt` (interface + `EngineCapabilities` + `EXECUTION_PROVIDER_REGISTRY`/
  `EngineRegistry` F3 + `GenAiSupport` probe + `ModelRuntimeFactory.selectEngine`(pure) / `.create`(device,
  transparent fallback to Native)); `ORTGeneratorGenAI.kt` (GenAI engine, loop-in-Kotlin callback parity);
  `cpp/genai_runtime.cpp` (session-handle streaming wrapper: `nativeCreate/SetSampling/Start/IsDone/Step/
  LastToken/Release` + `nativeGenAiAvailable`, promoted from the spike, stable C API).
- **Changed:** `runtime/InferenceEngine.kt` finalized as the canonical enum (retired #17's `DECOMPOSE(#11)`
  placeholder); `ORTGeneratorNative` now `: ModelRuntime` (added `capabilities`/`load`/`release`, `generate`
  is `override`); `ORTGenerationConfig` gained `engine: InferenceEngine? = null` (+ `overrideConfig`);
  `LLMRepository.ortNativeInference` is now `ModelRuntime?` selected via `ModelRuntimeFactory.create`
  (`makeModelRuntime`), `destroySession`→`release`.
- **Deleted:** `ORTGenAINative.kt` + `cpp/onnx-genai.cpp` (dead).
- **JVM tests:** `runtime/RuntimeSelectionTest.kt` — selection/fallback matrix + EP-registry parity (probe
  injected; no JNI).
- **Deferred to the #9-package step:** the full `ORTGeneratorGenAI.generate()` device smoke + dual-engine
  same-folder run (Gate 0.1 #1/#4) need a real File #9 package — the builder model is GenAI-format only and
  lacks the mobiletransformers tokenizer/inference-config the Native engine + `ORTTokenizerNative` require.
  `LLMRepository` currently offers `{native,genai}` to the factory; wiring `supportedEngines`/`defaultEngine`
  from the manifest variant (#13) is a small follow-up. **Nothing committed** (human commits).

### Next: real File #9 package (the dual-engine cross-engine validator) — SCOPED, not yet built

The #11 device dual-engine smoke + Gate 0.1 #1/#4 (same folder correct under BOTH engines + RSS) need a
**real File #9 package** that both `ORTGeneratorNative` and GenAI read from one `inference/` dir. The tiny
builder model (`build/genai_spike_model`) is GenAI-format only (single-blob, HF tokenizer) — Native +
`ORTTokenizerNative` can't consume it.

Producing one = **implementing the deferred #15 `_full_export`** (`src/mobiletransformers/export/pipeline.py`
— currently `raise NotImplementedError`). Stages already exist in the legacy root and must be wired:
`inference/export_inference_package.py` (base/trainable external split + `weight_handoff_map.json` +
`genai_config.json`, the #9 inference entry), `inference/builder.py` (`make_genai_config`), and
`artifact/onnx_builder.py` (`create_model`/`gen_artifacts`/`onnx_checktrain` for training artifacts +
mergers). Env: the **export** profile (optimum-onnx, 3.12) for the inference graph; **ort-training** for the
training artifacts/mergers — these are in **conflicting** `[tool.uv]` profiles, so the inference-side export
(what the cross-engine test needs) should be produced under the export profile alone; the training side is a
separate profile run. Output must carry the mobiletransformers tokenizer (`mobiletransformers_tokenizer_config.json`)
+ `generation_config.json` (Native) alongside `genai_config.json` (GenAI) so both engines load the same dir.

Once a real package exists: push it (internal filesDir), add a `ModelRuntime`/`ORTGeneratorGenAI` instrumented
test that loads it under each engine and asserts the same token (Gate 0.1 #1) + RSS within threshold (#4).

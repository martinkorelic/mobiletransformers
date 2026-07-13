# Restructure Implementation — Handoff

**Branch:** `restructure` · **Nothing is committed** (the human commits). **Date:** 2026-07-13.

This is the running handoff for the staged restructure in `agent_docs/IMPLEMENTATION_ORDER.md`. It
records what is done, the environment/gotchas a cold agent needs, and the next steps.

---

## Status: plans #1–#8 done (global order)

| # | Plan | State |
| --- | --- | --- |
| 1 | `00_code_plans/01` package & uv scaffolding | ✅ done |
| 2 | `00_code_plans/03` dependency profiles & ORT wheel | ✅ done |
| 3 | `01_code_plans/06` source-built ORT training pipeline | ✅ done (Gate 0.3 proven) |
| 4 | `00_code_plans/02` config layering | ✅ done |
| 5 | `00_code_plans/10` code quality & module health | ✅ done |
| 6 | `00_code_plans/09` typed models/enums/registries | ✅ owned contract layer done; consumption wiring deferred to #7/#9 |
| 7 | `01_code_plans/05` optimum ONNX export & TasksManager | ✅ done (inference + training export proven E2E; `OnnxConfigWithLoss` vendored — see below) |
| 8 | `00_code_plans/07` weight handoff map & tensor codec | ✅ Python owner layer done (schema + codec + `check_compat`); C++/Kotlin consumers ride #9/#23/#18 |

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

## Validation commands (paste-run)
```bash
uv sync --frozen --group dev && make check          # lint + typecheck + parity + 71 tests
uv run python -m mobiletransformers.codegen.enums --check   # cross-lang enum/schema parity
uv sync --python 3.12 --group ort-training-local && make test-train   # ORT toolchain alive (Gate 0.3)
uv build --wheel                                    # wheel = only src/mobiletransformers (+ py.typed, public_api.txt)
uv lock --check                                     # lock consistent with pyproject
```

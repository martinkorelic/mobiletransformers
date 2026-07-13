# Optimum-ONNX Inference Export + TasksManager Front Door

**Priority #7 | Prerequisites: #2 (`00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`), #4 (`00_code_plans/02_config_layering_settings_constants.md`), #6 (`00_code_plans/09_typed_models_enums_and_registries.md`, architecture registry) | Blocks: #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`), #15 (`02_code_plans/05_one_command_export_cli.md`); feeds #20 (`02_code_plans/02_optimum_support_matrix.md`)**

## Purpose

Replace the hard-coded per-architecture export branch in `trainer/builder.py:261-272` (an `if config.architectures[0] == "LlamaForCausalLM": ...` ladder over `LlamaOnnxConfig / GemmaOnnxConfig / Phi3OnnxConfig / Qwen2OnnxConfig / OPTOnnxConfig / BertOnnxConfig`) with a **support-discovery wrapper around `optimum.exporters.tasks.TasksManager`** plus a normalized `optimum.exporters.onnx.main_export` run. This becomes MobileTransformers' single inference-export front door: given an HF model id, it discovers the supported ONNX task, exports the inference graph, normalizes naming/KV-cache/tokenizer/generation-config/external-data into repo conventions, and emits a `model_support_matrix.json` status row.

> **Architecture registry split with `00_code_plans/09`.** TasksManager discovers the *task*; it does **not** pick the per-architecture `OnnxConfig` class (the `LlamaOnnxConfig / GemmaOnnxConfig / BertOnnxConfig / …` the old ladder selected). That class mapping is `09`'s **architecture registry** (`ArchitectureSpec.onnx_config_class`). So the training-graph export resolves its config class via `09.resolve_architecture(config).onnx_config_class` — never a re-introduced `architectures[0] ==` ladder. Adding an architecture is a registry entry, here and in `inference/builder.py`.

It also runs **the migration spike** that Tier 0 flagged: under `optimum-onnx 0.1.0` (optimum~=2.1), `main_export` and `TasksManager` import paths are preserved, but the repo's lower-level `from optimum.exporters.onnx import OnnxConfigWithLoss, export` (`trainer/builder.py:14`) are the symbols **most likely to have moved or been removed**. This plan verifies them explicitly and defines the fallback.

## Touched / new files

- NEW `mobiletransformers/export/registry.py` — the TasksManager wrapper: `discover_tasks(model_id)`, `choose_task(...)`, `is_supported(model_type)`.
- NEW `mobiletransformers/export/inference_export.py` — `export_inference(model_id, out_dir, task=None, opset=20, trust_remote_code=...)` driving `main_export`, then normalization.
- NEW `mobiletransformers/export/normalize.py` — graph/KV-cache name normalization, external-data relayout, tokenizer + `generation_config.json` placement into the repo package shape.
- NEW `mobiletransformers/export/support_matrix.py` — builds/updates `model_support_matrix.json`.
- NEW `spikes/optimum_migration/check_symbols.py` — the import-survival spike (runs in the **train+export unified** profile, where `OnnxConfigWithLoss`/`export` actually matter; export-only profile only needs `main_export`/`TasksManager`).
- EDIT `trainer/builder.py` — `optimum_hf_export` calls `registry.choose_task` instead of the `config.architectures[0]` ladder for the *training* export path; the legacy ladder stays only as a documented fallback for training-graph construction (see step 6).
- Inputs from #4: config layering for `opset`, `trust_remote_code`, `task` override, HF token.

## Data contracts / interfaces

- **`discover_tasks(model_id) -> {model_type, supported_tasks}`**: reads `AutoConfig.from_pretrained(model_id).model_type` (NOT `architectures[0]` — `model_type` is the TasksManager key, e.g. `"llama"`, `"phi3"`, `"qwen2"`), then `TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")`.
- **`choose_task(supported_tasks, override=None) -> task`**: selection order **`text-generation-with-past` > `text-generation` > `feature-extraction` > `sentence-similarity` > explicit `override`**. `with-past` is preferred because the inference engine needs the KV-cache (past/present) graph; fall back to `text-generation` only if `with-past` is unsupported. An explicit user `override` is honored last (it forces a task even outside the auto order) and recorded in the matrix.
- **`main_export` invocation**: `optimum.exporters.onnx.main_export(model_name_or_path=model_id, output=out_dir, task=task, opset=opset, trust_remote_code=trust_remote_code, ...)` (CLI equivalent: `optimum-cli export onnx --model <id> --task <task> ...`). Record the exact `optimum-onnx` version, `optimum` version, `transformers` version, opset, exporter mode, and whether `trust_remote_code` was needed — these go into the package manifest and the support matrix.
- **Export-frontend registry (F3)**: the export *front door* is selected from an `EXPORT_FRONTEND_REGISTRY` keyed by frontend (`optimum-onnx` — the default/durable inference exporter — and `torch.onnx` — the manual graph path used by Fallback A; future frontends slot in the same way), so the chosen exporter is **data, not an `if/elif`**. Each row carries its export callable + an availability/capability probe; `inference_export.py` resolves the frontend from config and the support-discovery result rather than branching on import success. Adding a frontend = registry row + enum member, no business-logic edit — the same closed-set-is-data principle as the architecture registry (`09`). The migration spike (step 6) feeds the `optimum-onnx` row's probe; if `OnnxConfigWithLoss`/`export` are gone, the training-graph export switches to the `torch.onnx` row without touching callers.
- **Normalized output (repo conventions)**: a package dir with the inference `model.onnx` (+ single external `.data`), normalized input names (`input_ids`, `attention_mask`, `position_ids`), KV-cache names (`past_key_values.<i>.key/value` in, `present.<i>.key/value` out — the same scheme `inference/builder.py:325-356` already writes into `genai_config.json`), tokenizer files, `generation_config.json`, and an external-data layout matching File #9's expectation (one immutable base blob; trainable tensors as external initializers later overwritten by the merger).
- **`model_support_matrix.json` statuses** (per model, canonical names owned by `02_code_plans/02`): `optimum_exportable`, `mobile_package_exportable`, `train_artifacts_exportable`, `android_inference_ready`, `android_training_ready`, `rag_ready`. This plan sets `optimum_exportable` (from discovery) and `mobile_package_exportable` (from a successful normalized export); later plans flip the rest.

## Implementation steps

1. **Discovery wrapper** (`registry.py`): implement `discover_tasks` / `choose_task` / `is_supported` over `TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")`. Unknown `model_type` ⇒ return empty supported set, status `optimum_exportable=false`, and a blocker note — do not raise.

2. **Inference export** (`inference_export.py`): resolve task via `choose_task`, run `main_export` into a temp dir, capture the version/opset/mode metadata. Keep this in the **export-only** profile (public `onnxruntime` + `optimum-onnx[onnxruntime]`); never co-sync with the training profile (`00_code_plans/03`).

3. **Normalize** (`normalize.py`): rename graph IO to the repo's canonical input/KV-cache names; relay out external data into one base blob; copy tokenizer + `generation_config.json`; emit the package shape File #9 consumes. Fail closed if any expected output (logits, present.*) is missing.

4. **Support matrix** (`support_matrix.py`): write/merge the per-model row. Set `optimum_exportable` from step 1 and `mobile_package_exportable` from steps 2–3 success. Leave training/android/rag statuses for later plans.

5. **Wire `trainer/builder.py`**: replace the `config.architectures[0]` ladder (`trainer/builder.py:261-272`) with `registry.choose_task` for the export *task* **and** `09.resolve_architecture(config).onnx_config_class` for the per-architecture `OnnxConfig` class (no ladder remains). The training export still needs `OnnxConfigWithLoss` to wrap that class — see the spike (step 6) for whether it survives.

6. **The migration spike** (`spikes/optimum_migration/check_symbols.py`, run in the train+export unified profile): assert **separately**:
   - `from optimum.exporters.onnx import main_export` — expected to resolve (paths preserved per Tier 0).
   - `from optimum.exporters.tasks import TasksManager` — expected to resolve.
   - `from optimum.exporters.onnx.model_configs import LlamaOnnxConfig, GemmaOnnxConfig, Phi3OnnxConfig, BertOnnxConfig, Qwen2OnnxConfig, OPTOnnxConfig` — expected to resolve (these are the model-config classes the ladder used).
   - `from optimum.exporters.onnx import OnnxConfigWithLoss, export` — **the at-risk pair.** These are the lower-level symbols used to build the *training* graph (`OnnxConfigWithLoss` wraps the config at `trainer/builder.py:278`; `export(my_model, ocl, onnx_path, opset, ...)` at `trainer/builder.py:369`).
   - Report a 4-line PASS/FAIL matrix. Do not infer survival of `OnnxConfigWithLoss`/`export` from `main_export` working — test them independently.

7. **Fallback definition** (documented, implemented only if step 6 fails on `OnnxConfigWithLoss`/`export`):
   - **Fallback A:** keep the inference export on `main_export` (durable), and construct the **training** graph manually — `OnnxTrainerWrapper` (`trainer/builder.py:84-92`) exported via `torch.onnx.export` directly (the wrapper already exposes the `input_ids/attention_mask/position_ids/labels` forward), bypassing `OnnxConfigWithLoss`/`export`.
   - **Fallback B (last resort):** pin a legacy `optimum==1.23.x` env solely for the training-graph export, isolated as its own uv profile with a sunset issue (this is the Tier 0 "Path B, temporary").
   - The inference path is unaffected by the fallback — `main_export` is the durable inference exporter regardless.

## Interactions

- **#2 (dependency profiles)**: export-only vs train+export-unified are mutually-exclusive `onnxruntime` providers. `main_export` inference export runs in export-only; the `OnnxConfigWithLoss`/`export` training path and its spike run in train+export-unified. Never co-sync.
- **#4 (config layering)**: supplies `opset`, `task` override, `trust_remote_code`, HF token to the exporter.
- **#9 (unified merger/external-data export)**: consumes the normalized package; the canonical KV-cache/initializer names this plan emits are what File #9's `weight_handoff_map.json` maps merged tensors onto. Naming must agree.
- **#20 (`02_code_plans/02_optimum_support_matrix.md`)**: the reporting layer that reads/extends `model_support_matrix.json` this plan seeds.
- **`inference/builder.py`**: its `make_genai_config` (`inference/builder.py:325`) already emits the canonical input/KV-cache naming — reuse those exact name templates in `normalize.py` so the native and GenAI engines agree.

## References

- `https://huggingface.co/docs/optimum/` — Optimum ONNX exporter (`main_export`) + TasksManager task discovery.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Discovery smoke** (`pytest tests/export/test_registry.py`): `discover_tasks` for `llama`, `phi3`, `qwen2` returns non-empty supported sets including `text-generation-with-past`; one unknown/unsupported `model_type` returns empty + `optimum_exportable=false` (no raise). (TasksManager lookup is a metadata-only call; cache/mock the config fetch to keep it fast.)
- **Task-selection unit test:** given a supported set, `choose_task` returns `text-generation-with-past`; with it removed, returns `text-generation`; with an `override`, returns the override.
- **Export-frontend registry unit (F3):** `EXPORT_FRONTEND_REGISTRY` resolves `optimum-onnx` by default and `torch.onnx` when selected; an unknown frontend key fails closed; selection is table lookup, not `if/elif`.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Support-matrix merge smoke:** feed two synthetic discovery/normalization results (fixtures, no real export) into `support_matrix.py`; assert `model_support_matrix.json` has one row each with `optimum_exportable=true` and `mobile_package_exportable` reflecting normalization success, and that re-merging is idempotent.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Inference export smoke:** `main_export` one tiny text-generation model (e.g. SmolLM2-135M) in the export-only profile; assert the normalized package has `model.onnx` + single `.data`, canonical IO names, present logits + `present.*`, tokenizer files, `generation_config.json`. Record opset + exporter mode. (Real multi-minute export.)
- **Migration spike (the decisive one):** run `check_symbols.py` in the **train+export-unified profile** (needs the source-built ORT-training wheel); produce the 4-line import PASS/FAIL matrix, with `OnnxConfigWithLoss`/`export` reported independently from `main_export`/`TasksManager`.
- **Support-matrix end-to-end:** export two real models and confirm the rows match the merge-smoke expectations on live exports.
- **Fallback smoke (only if spike fails):** export the training graph via the chosen fallback (torch.onnx of `OnnxTrainerWrapper`, or legacy optimum profile) and confirm it still feeds `artifacts.generate_artifacts` (File #8 pipeline).

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- A single export front door (`inference_export.py` over `EXPORT_FRONTEND_REGISTRY`) replaces the `trainer/builder.py:261-272` `architectures[0]` ladder: task via `registry.choose_task`, per-architecture `OnnxConfig` via `09.resolve_architecture(...).onnx_config_class`, no ladder remaining.
- A tiny model exports to a normalized package (canonical IO/KV-cache names, one base `.data`, tokenizer + `generation_config.json`) that File #9 consumes, and `model_support_matrix.json` carries `optimum_exportable` + `mobile_package_exportable` rows.
- The migration spike's 4-line PASS/FAIL matrix is recorded; if `OnnxConfigWithLoss`/`export` are gone, the documented fallback (torch.onnx frontend, or pinned legacy optimum) is selected via the registry and still feeds `generate_artifacts`.

---

## Implementation notes — done (2026-07-13)

Implemented and proven end-to-end on real models; `make check` green. Env tested: **optimum 2.1.0 /
optimum-onnx 0.1.0 / transformers 4.46.2 / torch 2.7.1** (the `export` profile, Python 3.12).

### The spike outcome changed the fallback strategy (a better third option)
The migration spike (`spikes/optimum_migration/check_symbols.py`) found: `main_export`, `TasksManager`,
the `*OnnxConfig` model_configs, **and `export()` all survive**; only **`OnnxConfigWithLoss` was removed**
(no replacement anywhere in optimum 2.1). Because `export()` survives *and* its deps
(`OnnxConfig`/`OnnxConfigWithPast`/`DummyLabelsGenerator`/`DEFAULT_DUMMY_SHAPES`) all survive, the chosen
resolution is **to vendor `OnnxConfigWithLoss`** (`src/mobiletransformers/export/onnx_config_with_loss.py`,
adapted from optimum v1.24.0) and keep the training-graph export on the durable `export()`. This is
strictly better than the plan's menu:
- **Not Fallback A** (torch.onnx graph reconstruction) — unnecessary; `export()` works.
- **Not Fallback B** (pin legacy optimum) — unnecessary; no second env.
- The `torch.onnx` `EXPORT_FRONTEND_REGISTRY` row remains **declared + fail-closed**
  (`export/torch_frontend.py`), reserved for the day `export()` itself disappears.

Provenance/licensing: the vendored file carries an Apache-2.0 attribution header (optimum, © HuggingFace)
to be enumerated in `THIRD_PARTY_NOTICES.md` by the relicense pass (#32).

### Two discovery gotchas (handled inside `export/registry.py`)
- `TasksManager`'s ONNX task map is **empty** unless `optimum.exporters.onnx.model_configs` is imported
  first (its `@register_tasks_manager_onnx` decorators do the registration). `registry._ensure_onnx_registered()` does this.
- `get_supported_tasks_for_model_type(model_type, "onnx", library_name="transformers")` — the
  `library_name` kwarg is now required (deprecation → error path otherwise).

### What landed (file inventory as built)
- `src/mobiletransformers/export/`: `registry.py` (discovery + `EXPORT_FRONTEND_REGISTRY` keyed by the new
  **Python-only** `ExportFrontend` enum — deliberately excluded from `ENUM_REGISTRY`, no Kotlin mirror),
  `inference_export.py` (`export_inference` over `main_export`, captures optimum/optimum-onnx/transformers
  versions), `normalize.py` (verifies canonical IO/KV names, fail-closed on missing `logits`/`present.*`,
  consolidates external data to one `model.onnx_data`), `support_matrix.py` (idempotent merge; sets
  `optimum_exportable` + `mobile_package_exportable`, seeds the 4 deferred statuses `None`, preserves
  later-plan values), `onnx_config_with_loss.py` (vendored), `torch_frontend.py` (reserved).
- `trainer/builder.py`: the `architectures[0]` ladder is replaced by
  `resolve_architecture(config).load_onnx_config_class()` + `choose_task`; the broken
  `from optimum... import OnnxConfigWithLoss` is repointed to the vendored module. (Also fixed a latent
  unbound-`ocl` bug on unknown architectures — the old ladder had no `else`.)
- Tests: `tests/export/{test_registry,test_support_matrix,test_onnx_config_with_loss}.py` — 11 run in
  core/dev, 6 skip there and pass under `uv run --extra export`.

### Deferred (per plan ownership)
- Real full-size model export smoke (e.g. SmolLM2-135M) is a user-run manual test (the automated E2E uses
  `hf-internal-testing/tiny-random-LlamaForCausalLM`).
- `inference/builder.py` dispatch rewrite stays gated by the Optimum-vs-GenAI decision (not #7).
- The `torch.onnx` frontend body only if a future optimum removes `export()`.
- **Env note:** `uv sync --extra export` rebuilds `.venv` on Python 3.12; reset the core/dev env with
  `uv sync --frozen --group dev --python 3.10` before `make check` (else mypy trips on numpy 3.12 stubs).

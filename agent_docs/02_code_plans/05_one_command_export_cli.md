# One-Command Export CLI

**Priority #15 | Prerequisites: #7 (`01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`), #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`), #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`) | Blocks: #21 (hub pull), starter-zoo generation, #22 (adapter push-back uses the same model-card builder)**

## Purpose

Collapse the current multi-script export flow (`inference/builder.py create_model` → `tools/onnx_builder.py convert_pipeline` → manual config/tokenizer/merger emission) into **one command** that turns a HuggingFace repo id into a MobileTransformers-ready package directory (the #14 layout) plus `mobiletransformers_manifest.json`, `checksums.json`, `weight_handoff_map.json`, and `optimum/export_report.json`.

```bash
mobiletransformers export \
  --model Qwen/Qwen2-0.5B \
  --task text-generation-with-past \
  --peft mars-opt1 \
  --rank 8 \
  --quant qint8 \
  --output build/packages/qwen2-0.5b-mobile \
  [--variant cpu-int4] [--include-rag] [--embedding-model sentence-transformers/all-MiniLM-L6-v2] [--validate]
```

And a companion publish command:

```bash
mobiletransformers push \
  --package build/packages/qwen2-0.5b-mobile \
  --repo-id mobiletransformers/Qwen2-0.5B-mobile \
  [--private] [--revision main]
```

The CLI is a thin orchestrator. It must reuse, not reimplement, the existing builder stages and the contracts owned elsewhere (Optimum export #7, unified merger/external-data #9, manifest+cache bridge #13, package format #14, handoff map #8/07).

## Touched / new files

New (pkg root scaffolded by #1; config layering from #4 `00_code_plans/02`):

- `mobiletransformers/cli/__main__.py` — argparse/typer dispatcher registering subcommands `export`, `push`, `pull`, `install-package` (last two from #21). Console-script entrypoint `mobiletransformers = mobiletransformers.cli.__main__:main` in `pyproject.toml`.
- `mobiletransformers/cli/export.py` — `run_export(args)` orchestration.
- `mobiletransformers/cli/push.py` — `run_push(args)` + model-card generation (shared with #22).
- `mobiletransformers/export/pipeline.py` — `export_package(...)` programmatic API (the CLI is a wrapper over this).
- `mobiletransformers/export/model_card.py` — `render_model_card(manifest, package_dir) -> str`.

Reused existing code (wrapped, not forked):

- `inference/builder.py`: `create_model(model_name, input_path, output_dir, precision, execution_provider, cache_dir, **extra_options)`, `Model.make_genai_config()`, `Model.save_processing()`, `parse_hf_token()`. This is the inference-graph + `genai_config.json` + tokenizer producer.
- `tools/onnx_builder.py`: `convert_pipeline(...)`, `gen_artifacts(...)` (ORT training artifacts via `onnxruntime.training.artifacts.generate_artifacts`), `gen_genai(...)`, `force_dequantize_external_and_save(...)`.
- `artifact/merger.py`: `build_merger_model(MergerSpec, output_path)` — the single unified merger builder from `00_code_plans/09` (it replaces the legacy `create_*_merger_model{,_2}` factories; do not call those). The CLI iterates `09.resolve_merger(peft_method, quant_in, quant_out)` and emits one merger per spec.
- `tools/tokenizer_export.py`: `export_tokenizer_config(...)` (writes `tokenizer/` + `ortmobile_tokenizer_config.json`).
- `tools/parser_config.py`: section constants `TRAIN_CONFIG`, `INFERENCE_CONFIG`, `ARTIFACT_CONFIG`, `TASK_NAME_TO_DATASET`.
- `config.py`: `HF_TOKEN`; `config.yml` sections `TRAIN_BUILDER`, `INFERENCE_BUILDER`, `ARTIFACT_BUILDER`.
- `mobiletransformers/hub/package_format.py` (#14): `build_manifest`, `FEATURE_GROUPS`, repo-shape constants.

## Data contracts / interfaces

### CLI flags → internal config

| Flag | Maps to | Notes |
| --- | --- | --- |
| `--model` | `TRAIN_BUILDER.model_id`, `create_model` `model_name`, manifest `baseModelId` | required (HF id or local path) |
| `--task` | Optimum `TasksManager` selected task; manifest `selectedTask` | optional; auto-selected if omitted (see step 2) |
| `--peft` | `TRAIN_BUILDER.train_method` + `mars.optimization_level` | `lora` → `train_method=lora`; `mars-opt0..opt4` → `train_method=mars`, `mars.optimization_level=0..4` (see `peft_models/mars/config.py` `MarsConfig.optimization_level` 0–4) |
| `--rank` | `TRAIN_BUILDER.lora_rank` / `MarsConfig.r` | default 8 |
| `--quant` | `INFERENCE_BUILDER` precision + `weight_type` | `qint8`→`weight_type=QInt8` dynamic; `int4`→`precision int4` (MatMul4Bits); `fp16`→no quant |
| `--output` | package dir root | required |
| `--variant` | package variant id | default derived from `--quant`+EP, e.g. `cpu-int4` |
| `--include-rag` / `--embedding-model` | `gen_rag_config=True`, `embedding_model` path | optional |
| `--validate` | run desktop smoke after build | optional |

`peft` parsing detail: `mars-opt1` → `("mars", optimization_level=1)`. Validate against `MarsConfig` (opt levels 0–4; levels with quantization set `modules_to_quantize` from `config.yml TRAIN_BUILDER.mars.modules_to_quantize`).

### Outputs (the #14 package layout)

After a successful `export`, `--output` contains the `variants/<id>/{train,inference,embedding}` + `shared/` + `optimum/` tree, with:
- `train/training_config.json` carrying the keys `gen_artifacts()` already emits: `requires_grad`, `peft_mapping`, `rank`, `alpha`, `peft_target`, `trainable_parameter_count`, plus injected `peftMethod`, `modelId`.
- `train/weight_handoff_map.json` (schema owned by #8/07) emitted by the merger/external-data stage (#9).
- `train/trainable_parameters.json` — the per-tensor trainable initializer name list (one-file-per-tensor manifest), derived from `requires_grad`.
- `inference/base/model.onnx(+.data)` (frozen quantized base) and `inference/merged/<per-tensor files>` (#9/01 split).
- `inference/generation_config.json` (`{"type":"native",...}`) and, for GenAI-capable variants, `inference/genai_config.json` from `Model.make_genai_config()`.
- `mobiletransformers_manifest.json`, `checksums.json`, `optimum/{export_report,supported_tasks,optimum_config}.json`.

## Implementation steps

`export_package()` orchestration (each step delegates to existing code):

1. **Load config + settings.** Read `config.yml` defaults (sections via `parser_config` constants), overlay CLI flags through the #4 settings layering. Resolve `HF_TOKEN` from env (`config.py`). Build the effective `TRAIN_BUILDER` / `INFERENCE_BUILDER` / `ARTIFACT_BUILDER` dicts.
2. **Task discovery/validation (#7).** Load `AutoConfig` (`trust_remote_code` per flag), read `model_type`/`architectures`. Call the #7 `TasksManager` wrapper: `get_supported_tasks_for_model_type(model_type, "onnx")`. If `--task` given, validate it's supported; else auto-pick priority `text-generation-with-past` > `text-generation` > `feature-extraction` > `sentence-similarity`, else error asking for `--task`. Capture `supportedTasks`, `selectedTask`, `trustRemoteCode`, optimum/transformers versions for the report.
3. **Inference ONNX export (#7 + `inference/builder.py`).** Call `create_model(model_name=--model, input_path=None, output_dir=<tmp build>/inference, precision=<from --quant>, execution_provider="cpu", cache_dir=<tmp>, hf_token=..., **extra)`. This yields `model.onnx` + `model.onnx.data`. (Optimum `main_export` is the front door per #7; `create_model` is the GenAI-builder fallback path. The pipeline picks per the #7 toolchain decision.)
4. **Normalize.** Run the #9 external-data normalization: force per-tensor external initializers for trainable MatMul weights (today `force_dequantize_external_and_save()` in `tools/onnx_builder.py` does the forcing) and split into a frozen `inference/base/` blob + trainable per-tensor `.bin` files **flat in `inference/`** (the legacy `inference/merged/` subdir is retired per the canonical external-initializer decision). Normalize KV-cache names and trainable-weight initializer names to match the engine's expectations. Emit `weight_handoff_map.json` (#8/07) recording the train-checkpoint-name → external-initializer-file mapping.
5. **Training artifacts (#9 + source-built ORT wheel).** After PEFT/MARS trainable modules are known, call `gen_artifacts(train_dir, artifact_dir=<build>/train, model_name=<quant train model>, training_config=...)` which calls `onnxruntime.training.artifacts.generate_artifacts(...)` to produce `training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint/`, and the extended `training_config.json`. **Must run in the `onnxruntime-training` env profile** (#2/03 dependency-profile isolation) — never the same env as `optimum-onnx`/`onnxruntime-genai`.
6. **Merger graphs.** Emit on-device mergers via the unified `build_merger_model(spec, output_path)` (`00_code_plans/09`): iterate `09.resolve_merger(peft_method, quant_in, quant_out)` for the package's PEFT method and emit one merger per resolved `MergerSpec`, writing to the descriptive filename recorded in the handoff map. (Registry data preserves today's `convert_pipeline` gating — e.g. the non-quantized MARS merger only for `optimization_level <= 1` — as a spec condition, not a hand-coded `if`.)
7. **Tokenizer / chat-template / generation / genai config.** `export_tokenizer_config(--model, build_dir, HF_TOKEN)` → `tokenizer/` + `ortmobile_tokenizer_config.json`. Split out `chat_template.jinja` from `tokenizer.chat_template`. Write `inference/generation_config.json` (`{"type":"native",...}`). If variant supports GenAI, `Model.make_genai_config(...)` → `inference/genai_config.json`.
8. **Optional embedding / RAG.** If `--include-rag`: export/copy embedding model + tokenizer + `rag_config.json` into `variants/<id>/embedding/` (reuse `convert_pipeline`'s `gen_rag_config` branch + `get_all_metadata_from_onnx` for `embeddingDimension`).
9. **Reshape into #14 package layout.** Move the tmp `build/{train,inference,tokenizer,embedding}` outputs into `variants/<id>/...` + `shared/...` per #14's mapping helper.
10. **Emit manifest + checksums + reports.** Call `build_manifest(package_dir, ...)` (#14) → `mobiletransformers_manifest.json` (incl. `downloadPlan`, `fileSizes`, `sha256`, `weightHandoff`, engine/runtime/version fields) + per-variant `checksums.json`. Write `optimum/export_report.json` (versions, task, timings, normalization summary), `optimum/supported_tasks.json`, `optimum/optimum_config.json`.
11. **Optional desktop smoke (`--validate`).** Run `convert_pipeline`'s training check (`onnx_checktrain`, one train step + optimizer step) and a 1-token inference generation against `inference/base/model.onnx`. Record pass/fail into `export_report.json`. Smoke runs in the ORT-training profile; generation may run in a separate profile if package conflicts (matches the existing comment in `convert_pipeline` that skips generation testing in the artifact script).

`run_push()`:

1. Validate the package against the #13 manifest validator (fail before any upload).
2. `render_model_card(manifest, package_dir)` → README.md: base model, exact base-model license + framework Apache-2.0, support matrix (inference/train/rag readiness), ORT-training/GenAI/optimum/transformers versions, `minimumAndroidApi`, `recommendedDeviceMemoryMb`, known device limits, variant table.
3. `huggingface_hub.upload_folder(folder_path=package_dir, repo_id=--repo-id, repo_type="model", token=..., revision=--revision, commit_message=...)` under stable variant paths; `create_repo(exist_ok=True, private=--private)` first.

## Interactions

- **#7 (optimum/TasksManager):** task discovery + inference export front door (step 2–3).
- **#9 (unified merger + external-data):** owns the `inference/base`+`inference/merged` split and the per-tensor external-initializer normalization (step 4) and informs the handoff map (#8/07).
- **#2/03 (dependency profiles):** export must shell out to / run training-artifact generation in the `onnxruntime-training` env; the CLI must not import `optimum-onnx` and `onnxruntime-training` in one process.
- **#13 (manifest + cache bridge):** validator gate for `push`; manifest schema producer (`build_manifest`).
- **#14 (package format):** the emitted directory shape and manifest fields.
- **#21 (pull):** consumes `downloadPlan`/`sha256` this CLI writes; uses the tiny fixture this CLI can also regenerate.
- **#22 (adapter push-back):** reuses `model_card.py` and the `huggingface_hub.upload_folder` push path.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- CLI arg-parse unit (`pytest tests/cli/test_export.py`): `--peft mars-opt1` → `(train_method="mars", optimization_level=1)`; `--quant qint8` → dynamic QInt8; invalid `--peft mars-opt9` rejected.
- Config-overlay test: `config.yml` defaults + CLI flags → effective `TRAIN_BUILDER/INFERENCE_BUILDER/ARTIFACT_BUILDER` dicts (uses #4 layering).
- Task-discovery test (#7 wrapper) for a supported model_type (qwen2) and an unsupported one (asserts the "provide --task" error).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- Dry-run test from `config.yml`: `export_package(dry_run=True)` resolves task, plans output paths, and prints the manifest skeleton without writing large ONNX.
- Export smoke for the #14 fixture stub: produces the full `variants/<id>` tree, emits `mobiletransformers_manifest.json` + `checksums.json` + `weight_handoff_map.json` + `optimum/export_report.json`, and the result validates against #13.
- Handoff-map test: every entry's merged-initializer name resolves to a real file under `inference/merged/`.
- `push` dry-run: `render_model_card` produces a card naming base model, both licenses, version pins, and the variant table; `upload_folder` invoked with the expected `repo_id`/paths (mocked Hub).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Real export for one tiny model (needs the source-built `onnxruntime-training` wheel in its env profile): `mobiletransformers export --model <tiny HF id> ... --validate` produces the package and runs the desktop smoke. Skipped in CI when the wheel is unavailable.
- `--validate` smoke: one train step + 1-token generation pass recorded in `export_report.json` (runs in the ORT-training profile; skippable in CI if the source-built ORT wheel is unavailable).

**Workflow (end-to-end)** — *(CHECKPOINT #15, Python export E2E)* one `mobiletransformers export` command on a tiny model produces a validated device-ready package: the `variants/<id>/{train,inference,embedding}` + `shared/` + `optimum/` tree with inference graph (`inference/base/model.onnx(+.data)`) + ORT training artifacts (`training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint/`) + on-device merger graphs + `mobiletransformers_manifest.json` + `checksums.json` + `weight_handoff_map.json`. Assert the package **validates** against the #13 manifest validator (and, with `--validate`, that the recorded desktop smoke passed). Over the #14 fixture stub this leg is automated; the real-tiny-model run is Manual (ORT-training wheel).

**Definition of done** — `mobiletransformers export` turns a single HF id into the #14 package layout in one command, delegating to the existing builder stages (Optimum #7, unified merger #9, manifest #13/#14) without reimplementing them; the emitted manifest carries `downloadPlan`/`fileSizes`/`sha256`/`weightHandoff` + engine/runtime/version fields; the package validates against #13; `mobiletransformers push` gates on that validation and renders a model card with both licenses + version pins before `upload_folder`; and the fixture-stub export E2E above is green in CI.

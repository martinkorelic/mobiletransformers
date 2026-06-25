# Adapter Push-Back

**Priority #22 (last) | Prerequisites: #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`), #14 (`02_code_plans/03_hub_model_package_format.md`) | Blocks: nothing (terminal)**

## Purpose

After on-device fine-tuning, get the *trained adapter* (the per-tensor updated weights + metadata) back onto the Hub so it can be shared, audited, or re-applied. **Python-first**: export the adapter from the Android cache, convert to a PEFT-compatible layout where the math allows, write a model card with the mandatory privacy + license disclosures, and upload with `huggingface_hub`. Android direct upload is optional and gated behind auth + security review.

Hard caveat baked into the design: **MARS merge semantics may not map cleanly to standard PEFT / `.onnx_adapter`.** The exporter must *gate* — only emit a "PEFT-compatible" adapter when the handoff metadata proves a clean LoRA-shaped decomposition exists; otherwise emit a MobileTransformers-native adapter package and say so in the card.

## Touched / new files

New (Python pkg root; reuses #15's `model_card.py` + push path):

- `mobiletransformers/adapter/export.py` — `export_adapter_from_cache(cache_repo_dir) -> AdapterPackage` (reads the materialized Android cache layout).
- `mobiletransformers/adapter/convert.py` — `to_peft_layout(adapter_pkg) -> PeftAdapter | None` (None when MARS can't be cleanly mapped → gate).
- `mobiletransformers/adapter/model_card.py` — thin wrapper over `mobiletransformers/export/model_card.py` (#15) adding adapter-specific sections.
- `mobiletransformers/cli/push_adapter.py` — `mobiletransformers push-adapter` subcommand (registered in `cli/__main__.py`).

Android (optional, gated — later milestone):

- `hub/AdapterUploader.kt` — direct upload from device; **disabled by default**, behind an explicit auth + `BuildConfig` security flag.

Reads from the on-device cache shape (produced by #21, owned by #13):

- `<cacheDir>/<sanitizedRepoId>/train/checkpoint/` — ORT `CheckpointState` with the trained parameters.
- `<cacheDir>/<sanitizedRepoId>/train/training_config.json` — `requires_grad`, `peft_mapping`, `rank`, `alpha`, `peft_target`, `trainable_parameter_count`, `peftMethod`, `modelId`.
- `<cacheDir>/<sanitizedRepoId>/train/weight_handoff_map.json` — train→infer tensor contract (#8/07).
- `<cacheDir>/<sanitizedRepoId>/inference/merged/<per-tensor files>` — the per-tensor merged external initializers written on-device by `ORTTrainerNative.mergeExportSessionWeights()` (→ `inference/merged`).

## Data contracts / interfaces

### `AdapterPackage` (intermediate)

```jsonc
{
  "baseModelId": "Qwen/Qwen2-0.5B",        // from training_config.modelId
  "peftMethod": "mars",                      // training_config.peftMethod: "mars" | "lora"
  "marsOptimizationLevel": 1,                // null for lora; from the MARS config / training_config
  "rank": 8,
  "alpha": 64,
  "peftTarget": ["q_proj", "v_proj"],        // training_config.peft_target
  "trainableParameterCount": 1234567,
  "tensors": [                                // one entry per trained tensor
    {
      "trainingCheckpointName": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight",
      "mergedInitializerFile": "model.layers.0.attn.q_proj.MatMul.weight",  // file under inference/merged/
      "dtype": "float16", "shape": [896, 896]
    }
  ],
  "handoffMode": "external_initializer",      // from weight_handoff_map.json
  "source": { "device": "android", "exportedAt": "..." }
}
```

Two export modes, chosen by `convert.py` gate:

1. **PEFT-compatible (`lora`, or `mars` proven LoRA-decomposable):** emit standard PEFT files — `adapter_config.json` (PEFT type, r, alpha, target_modules, base_model_name_or_path) + `adapter_model.safetensors` (the A/B low-rank factors). Requires the handoff map to expose the LoRA factors (not just merged weights); if only merged weights are present, decomposition is lossy → fall to mode 2.
2. **MobileTransformers-native adapter (gate fallback, esp. MARS):** emit the per-tensor merged initializers + `weight_handoff_map.json` + `training_config.json` under a `variants/<id>/train/` style subtree (#14), with the model card stating it is **not** a drop-in PEFT adapter and must be re-applied via MobileTransformers.

`to_peft_layout()` returns `None` (→ mode 2) when: `peftMethod == "mars"` and `marsOptimizationLevel` involves quantization/fusion that breaks a clean `W + BA` recovery, or when the handoff map only carries merged tensors. Otherwise returns the PEFT layout.

### Model card mandatory sections

- Base model id + **exact upstream weight license** (restate, not "see base"); framework Apache-2.0.
- PEFT type (`lora` / `mars`) + MARS optimization level + rank/alpha + target modules.
- Dataset notes (task name; the bundled `<task>.jsonl` is *not* uploaded unless the user opts in).
- **Privacy warning:** adapters trained on-device may encode private user data; uploading is a deliberate publication. Bold, near the top.
- Re-apply instructions (PEFT path vs MobileTransformers-native path).

## Implementation steps

1. `export_adapter_from_cache(cache_repo_dir)`:
   - Read `train/training_config.json` → fill `baseModelId`/`peftMethod`/`rank`/`alpha`/`peftTarget`/`trainableParameterCount`/`marsOptimizationLevel`.
   - Read `train/weight_handoff_map.json` → `handoffMode` + per-tensor name mapping.
   - Enumerate `inference/merged/` files; for each, cross-reference the handoff map to build `tensors[]`. (Optionally read `train/checkpoint/` via ORT `CheckpointState` to recover requires_grad params directly if `inference/merged` is absent.)
   - Return `AdapterPackage`.
2. `to_peft_layout(adapter_pkg)`:
   - If `peftMethod == "lora"` and handoff exposes A/B factors → build `adapter_config.json` + `adapter_model.safetensors`; return `PeftAdapter`.
   - If `peftMethod == "mars"` → attempt decomposition only when the handoff map flags it as cleanly recoverable; else return `None`.
3. `build adapter package dir`:
   - Mode 1: write PEFT files at repo root.
   - Mode 2: write the native adapter subtree (merged tensors + handoff map + training_config) and a `mobiletransformers_adapter.json` header noting mode + base model + handoff pointer.
4. `model_card`: call `render_model_card` (#15) extended with the adapter sections above; assert privacy + license blocks present (fail upload if missing).
5. `push-adapter` CLI: `--cache-repo <dir> --repo-id mobiletransformers/<name>-adapter [--private] [--peft-only]`. `--peft-only` errors instead of falling back to mode 2 (for users who specifically need a PEFT adapter). Upload via `create_repo(exist_ok=True, private=...)` + `huggingface_hub.upload_folder(...)`.
6. **Android direct upload (gated, optional):** `AdapterUploader.kt` only compiled/enabled behind a security flag; mirrors the Python flow but uploads from device. Default product path is: device → desktop sync → Python `push-adapter`. Do not enable device upload until auth handling + the privacy gate are security-reviewed.

## Interactions

- **#9 (unified merger + external-data):** defines the `inference/merged/` per-tensor files and the merge semantics this exporter reads; the MARS-vs-PEFT gate hinges on what #9's handoff map records.
- **#14 (package format):** mode-2 native adapters reuse the `variants/<id>/train/` subtree shape and the `weight_handoff_map.json` location.
- **#8/07 (handoff map):** authoritative source for `handoffMode` and per-tensor name mapping; the gate decision reads its flags.
- **#15 (export CLI):** shares `model_card.py` and the `huggingface_hub.upload_folder` push path.
- **`ORTTrainerNative.mergeExportSessionWeights()`:** the on-device producer of `inference/merged` that this exporter consumes.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `to_peft_layout` gate (`pytest tests/adapter/test_convert.py`): LoRA fixture → returns `PeftAdapter` with valid `adapter_config.json`; MARS-opt4 (quantized) fixture → returns `None` and the CLI falls back to mode 2.
- `--peft-only` on a MARS fixture → hard error (no silent mode-2 fallback).
- Model-card test: rendered card contains the bold privacy warning, the exact base-model license string, PEFT type + MARS level + rank/alpha.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `export_adapter_from_cache` over a fixture cache dir (LoRA): builds `AdapterPackage` with correct `tensors[]` cross-referenced to the handoff map.
- Push dry-run (mocked Hub): `upload_folder` called with the adapter repo id; PEFT mode uploads `adapter_config.json` + `adapter_model.safetensors`; native mode uploads the merged-tensor subtree + `mobiletransformers_adapter.json`.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Round-trip (optional, gated): export PEFT adapter → reload with `PeftModel.from_pretrained` (already used in `inference/builder.py`) to confirm the layout loads (needs torch/peft).

**Definition of done** — `mobiletransformers push-adapter` reads the materialized Android cache (`train/checkpoint`, `training_config.json`, `weight_handoff_map.json`, `inference/merged/`), builds an `AdapterPackage`, and emits either a PEFT-compatible layout (`adapter_config.json` + `adapter_model.safetensors`) when `to_peft_layout` proves a clean LoRA decomposition or a MobileTransformers-native adapter subtree otherwise; the gate falls back honestly (and `--peft-only` errors instead of falling back); every rendered model card carries the bold privacy warning + exact base-model license + PEFT/MARS metadata (upload fails if missing); and uploads go through the shared `huggingface_hub.upload_folder` path with the expected repo id.

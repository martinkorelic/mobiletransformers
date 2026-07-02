# Unified Merger And External-Data Export

**Priority #9 | Prerequisites: #6 (`00_code_plans/09_typed_models_enums_and_registries.md`, merger registry + `build_merger_model`), #7 (`01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`), #8 (`00_code_plans/07_weight_handoff_map_and_tensor_codec.md`) | Blocks: #10 (`02_genai_external_data_swap_spike.md`), #11 (`03_inference_engine_abstraction_native_and_genai.md`), #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`)**

> **Consumes `00_code_plans/09` (merger registry + full unification).** This plan no longer leaves the merger-construction math untouched. The four near-duplicate factories — `create_lora_merger_model` (`artifact/merger.py:240`), `create_lora_merger_model_2` (`:351`), `create_mars_merger_model_2` (`:599`), `create_mars_merger_model` (`:824`) — are collapsed by 09 into a single parameterized `build_merger_model(MergerSpec)`. The "merger 1/2" naming and the `_2` suffixes are **removed**; output filenames are descriptive and recorded in the handoff map. This plan owns the **on-disk handoff-filename contract and the device merge path**; 09 owns the **registry/spec/builder**. The hard-coded merger dispatch (`onnx_builder.py:628-641` `if peft_method == "lora" … elif "mars"`) and the C++ literal branches (`weight_merger.cpp:476-499, 526-712`) are replaced by registry/handoff-map lookups (see Touched files + step 6.5).

## Purpose

Today there are **two overlapping, partially-divergent inference-export paths** plus **two merge implementations**, and the device side rebuilds tensor names from hard-coded string rules. This plan unifies them into **one inference builder** and **one weight-handoff contract** so both engines (Native ORT, GenAI) read the *same* folder and the *same* per-tensor external files.

Concrete goals:

1. Collapse `inference/builder.py` (GenAI-style graph + `genai_config.json`) and `artifact/onnx_builder.py::gen_genai(...)` into a single export entry point.
2. Emit trainable/merged weights as **ONNX external initializers, one file per tensor** (`<name>.bin`) inside `<cacheDir>/<model>/inference/`. The frozen quantized base stays a **separate immutable external blob**, never overwritten.
3. Emit `weight_handoff_map.json` (schema owned by `00_code_plans/07`) as the single source of truth for tensor identity, replacing the hard-coded rewrites in `weight_merger.cpp:904` (`WeightMerger::inference_name`) and the Python `backbone.model` -> `model` replacements.
4. Support a selectable `handoff_mode`: `external_initializer` (default, dual-engine, no graph rewrite), `model_input` (fallback only — ports the legacy `weight_input=True` graph-input conversion), `adapter` (stub, later).
5. Make **both** mergers — `WeightMerger` (device C++) and `artifact/merger.py` (offline Python) — write merged tensors to the **exact external filenames named in the handoff map**, with atomic rename + checksum.

The load-bearing invariant: *the name/shape/dtype/quant-metadata a merger writes must exactly equal what the inference graph consumes as an external initializer.* No string-munging on device.

## Touched / new files

Python (export side):
- `inference/builder.py` — becomes the **single** inference builder. Keep `make_genai_config(...)` (lines ~325-391), `make_external_tensor(...)` (lines ~559-574, already emits `<name>.bin` one-file-per-tensor), `make_inputs_and_outputs(...)` KV-cache naming (lines ~627-638). **Add**: handoff-map emission, base/trainable file separation, `handoff_mode`, ported `model_input` conversion, `session_options.config_entries` emission.
- `artifact/onnx_builder.py` — **demote** `gen_genai(...)` (lines 384-498). Its `weight_input=True` block (lines 410-445) and `force_dequantize_external_and_save(...)` (lines 184-257) move into the unified builder as named helpers (`_convert_initializers_to_inputs`, `_force_external`). `gen_genai` becomes a thin deprecated shim that calls the unified path, or is deleted once callers migrate.
- `artifact/merger.py` — the four merger-model factories (`create_lora_merger_model` `:240`, `create_lora_merger_model_2` `:351`, `create_mars_merger_model_2` `:599`, `create_mars_merger_model` `:824`) are **collapsed by `00_code_plans/09` into one `build_merger_model(MergerSpec, output_path)`** (graph math preserved per-variant, but expressed once and parameterized by `(peft_method, quant_in, quant_out)`). The **offline merge driver** calls `build_merger_model` per required `MergerSpec` and looks up output filenames from the handoff map instead of inventing `_2`/`qmerger` names. `artifact/onnx_builder.py:628-641` (`if peft_method == "lora" … elif "mars" … else raise "Unsupported PEFT method."`) and its `_2`-factory imports (`:31`) are deleted in favor of iterating `09.resolve_merger(...)` specs.
- Handoff-map builder/loader/validator: **import `mobiletransformers.artifacts.handoff_map` (#8's module — #8 is order 8, already merged when this plan runs).** Do NOT create a second `handoff_map.py`; this plan only *produces* the `weight_handoff_map.json` file through #8's `TrainableTensorCodec`/`HandoffMap` APIs.
- NEW `inference/export_inference_package.py` — top-level orchestrator: take Optimum/torch.onnx graph -> normalize -> split base vs trainable externals -> emit handoff map + genai_config -> validate.

C++ (device merge side):
- `android/ORTransformer/ORTransformersMobile/src/main/cpp/weight_merger.cpp` / `.h` — replace `WeightMerger::inference_name(...)` (lines 904-925) and the filename logic in `save_merged_parameters(...)` (lines 815-901) with a **handoff-map lookup**. Add a handoff-map loader (`load_handoff_map(path)`), atomic write (`temp + rename`), and checksum. **Also replace the merger-variant string dispatch:** `get_merger_type` (`:476-499`, which returns the literals `"mars_q"/"lora_q"/"lora"`) resolves the `MergerVariant` from the handoff map (the variant is recorded per-entry by the exporter); `run_merger_model`'s `if (merger_type == "lora") … else if "lora_q" … else if "mars_q"` branches (`:526, :562, :613, :687, :707`) select session + IO names from a registry-built variant→session map; the hard-coded merger ONNX paths (`:448-464`: `/lora_merger_model.onnx`, `/lora_qmerger_model.onnx`, `/mars_qmerger_model.onnx`) come from the handoff map's descriptive filenames. Closes `weight_merger.cpp:494` (`// TODO: Custom merger model?`).

Config:
- `config/config.yml` (moved there by `00_code_plans/02`) — `ARTIFACT_BUILDER.inference_export_config` already has `weight_input` (line ~183) and `force_external_initializers` (line ~195); `INFERENCE_BUILDER.export_genai_config` (line ~141); `inference_config.loadMergedWeights` (line ~208). **Add** `handoff_mode: external_initializer` and keep `weight_input` as the `model_input`-mode toggle (or fold `weight_input: true` -> `handoff_mode: model_input`).

## Data contracts / interfaces

### On-disk layout (`<cacheDir>/<model>/inference/`)

```
inference/
  model.onnx                      # graph; initializers are EXTERNAL refs only
  genai_config.json               # GenAI engine config (see below)
  weight_handoff_map.json         # SINGLE SOURCE OF TRUTH (schema: 00_code_plans/07)
  frozen_base.onnx.data           # frozen quantized base — IMMUTABLE, mmap'd read-only (flat; matches the map's frozenBaseBlob)
  <trainable_tensor_a>.bin        # one file per trainable tensor (overwritten by merge)
  <trainable_tensor_a>.bin.sha256
  <trainable_tensor_b>.bin
  ...
```

- Frozen base initializers point (`set_external_data location=...`) at `frozen_base.onnx.data` (flat in `inference/` — the canonical layout; **no `base/` subdir, no `merged/` subdir**). They are **never** written by a merger.
- Trainable initializers point at their own `<name>.bin` (one-file-per-tensor) so the device merge can overwrite a single file atomically without touching the base blob. This is what `make_external_tensor` already does for `<name>.bin`; extend it to *route* base tensors into the shared blob and trainable tensors into per-tensor files.

### `weight_handoff_map.json` (consumed here; schema authored in `00_code_plans/07` — that camelCase `entries[]` schema is the ONLY vocabulary)

Each entry maps one logical trainable weight across the train -> merge -> infer chain. This plan **must populate**, per entry (field names exactly as in #8): `trainingBaseLayerName`, `checkpointNames`, `mergerOutputNames` (e.g. `{"weight_quantized": "merged_weight_quantized"}` — the merger graph's output tensor per role), `mergedTensorNames`, `inferenceInitializerNames`, `externalDataLocation` (per-role `.bin` filename), `sha256` (per-role, of the bytes actually written), `genaiInputNames` (populated only in `model_input` mode), `dtype`, `shape`, `quantization` (role names for `weight_quantized`/`scale`/`zero_point`), `transposePolicy`. Top-level: `handoffMode`, `frozenBaseBlob` (`frozen_base.onnx.data`), `externalDataLayout`, `engines`, and `mergerModels` (resolved `MergerVariant` → descriptive merger ONNX filename from 09's `build_merger_model`). A worked instance lives in #8 — do not restate a divergent shape here.

Note the quantized triple comes straight from the current code: `WeightMerger::save_merged_parameters` writes `weight_quantized.tensor`, `weight_zero_point.tensor`, `weight_scale.tensor`; `artifact/merger.py` outputs `merged_weight_quantized` / `merged_scale` / `merged_zero_point`; `force_dequantize_external_and_save` keys on `MatMul.weight`, `MatMul.weight_zero_point`, `MatMul.weight_scale`. **The map reconciles all three vocabularies into the single `inferenceInitializerNames[role]` + `externalDataLocation[role]`.** This is also where Tier-0 finding #10 (the `safe_name` vs `base_layer_name + ".weight_scale"` inconsistency) is resolved: the scale filename becomes data in the map, with a regression test.

### `genai_config.json` additions

In `make_genai_config(...)`, add under `model.decoder.session_options.config_entries` (belt-and-suspenders so GenAI resolves the per-tensor externals from the package dir):

```
["session.model_external_initializers_file_folder_path", "<absolute inference dir at runtime>"]
```

This is a real documented ORT key. On a file-path load (`OgaCreateModel(<inference dir>)`) ORT already resolves external initializers by the `location/offset/length` recorded in `model.onnx`; the config entry is the fallback for memory-buffer loads. Do **not** rely on it as the primary mechanism.

### `handoff_mode` semantics

| Mode | Graph shape | Who consumes weights | Engines | Status |
| --- | --- | --- | --- | --- |
| `external_initializer` (default) | trainable tensors stay as external initializers | Native: `AddExternalInitializers`/file resolution; GenAI: file resolution from package dir | Native + GenAI | v1 |
| `model_input` (fallback) | trainable initializers removed, re-added as graph inputs | GenAI `set_model_input` / `OgaGeneratorParamsSetModelInput` | GenAI only | fallback |
| `adapter` | LoRA deltas as `.onnx_adapter` | GenAI adapter API | GenAI only | later (stub) |

`model_input` is the *port* of `gen_genai`'s `weight_input=True` path. It **disables ORT MatMul prepacking/constant-folding** on those tensors (verified: GenAI `ExtraInputs::Add` only feeds declared graph inputs), so it is strictly a fallback, never the default.

**HandoffMode (F7).** `handoff_mode` is backed by a `HandoffMode` enum that **enumerates all three** values (`external_initializer`, `model_input`, `adapter`), but **only `external_initializer` is supported in v1**. `model_input` and `adapter` are **registry stubs that fail closed**: selecting either raises a clear `NotImplementedError` ("handoff mode 'adapter' not supported in v1") rather than silently degrading. Keeping them as enumerated stubs (rather than deleting them) means a future plan adds the implementation behind the existing key — no new enum member, no caller changes — while v1 readers reject them deterministically.

## Implementation steps

1. **Lift the legacy conversions into named helpers** (no behavior change yet). Move `gen_genai`'s `weight_input=True` block (onnx_builder.py:410-445) into `inference/builder.py` as `_convert_initializers_to_inputs(model, requires_grad_names, opset_version)`. Move `force_dequantize_external_and_save` (onnx_builder.py:184-257) into `_force_external(model, path)`. Keep the existing `force_external or training_config` guard semantics (onnx_builder.py:476-480).

2. **Split base vs trainable externals.** In `inference/builder.py`, change initializer routing so `make_external_tensor` (builder.py:559-574) takes a `kind` ("base" | "trainable"). Base tensors append into one shared `frozen_base.onnx.data` flat in `inference/` (`onnx.save(..., save_as_external_data=True, location="frozen_base.onnx.data", size_threshold=0)` semantics, matching the current single-`.data` flow). Trainable tensors keep one `<name>.bin` each. Drive the split from `training_config["requires_grad"]` (the same list already read by `gen_genai` at onnx_builder.py:412 and by `generator_genai.py`).

3. **Emit `weight_handoff_map.json`.** Build one entry per `requires_grad` base layer, using #8's field names exactly: `trainingBaseLayerName`/`checkpointNames`, `mergerOutputNames`, `inferenceInitializerNames`, `externalDataLocation`, `dtype`, `shape`, `quantization` (resolve scale/zero-point names here — fix the `safe_name` vs `.weight_scale` inconsistency), `transposePolicy`. Set `genaiInputNames` only in `model_input` mode. Write the per-role `sha256` of the bytes actually written for each trainable `.bin`.

4. **Add `handoff_mode` switching.** Read from `config.yml`. In `external_initializer` mode: keep initializers external (call `_force_external`), do NOT convert to inputs. In `model_input` mode: call `_convert_initializers_to_inputs` and set `genaiInputNames[role] = inferenceInitializerNames[role]` in the map, and stop forcing those particular tensors external (they are now inputs). `adapter` mode: emit a TODO stub + clear NotImplemented error.

5. **Emit genai_config session_options.config_entries.** Extend `make_genai_config` to write the `session.model_external_initializers_file_folder_path` entry plus the harmless ORT entries from Tier-0 (`use_env_allocators`, `qdq_matmulnbits_accuracy_level`, etc.). Gate behind `export_genai_config`.

6. **Offline merge driver -> handoff filenames** (`artifact/merger.py` caller). Build each required merger via `09.build_merger_model(spec, output_path)` (iterating `09.resolve_merger(peft_method, quant_in, quant_out)` — **not** the deleted `create_*_merger_model{,_2}` factories). After running the merger ONNX, do NOT invent `output_path`; look up `externalDataLocation[role]` from the handoff map for each `inferenceInitializerNames[role]`, write to a temp file, `fsync`, atomic `os.replace`, then write `<file>.sha256` and update the entry's `sha256[role]`. Quantized outputs (`merged_weight_quantized`/`merged_scale`/`merged_zero_point`) route to the three filenames named in `quantization`.

7. **Device merge -> handoff lookup** (`weight_merger.cpp`). Add `load_handoff_map(path)` returning a `trainingBaseLayerName -> HandoffEntry{externalDataLocation, quantization names, dtype, shape, transposePolicy}` map. In `save_merged_parameters` (lines 815-901), replace the `inference_name(...)` call (line ~826) and the constructed `.tensor` filenames with the entry's `externalDataLocation[role]` (quantized roles from `quantization`). **Delete** `inference_name` (lines 904-925) or keep it only behind a `#ifdef LEGACY_NAME_REWRITE` fallback that logs a deprecation warning. Write through a temp path + `rename(2)` (atomic on same filesystem) and emit a checksum file.

8. **Fail closed on mismatch.** Both mergers: if a `requires_grad`/PEFT base layer has no handoff entry, or dtype/shape/transpose disagree, abort with an explicit error (do not silently fall back to string rewrites). On device, surface as a load error consumed by `ORTGeneratorNative`.

9. **Migrate callers.** Point the trainer's `mergeExportWeights` (ORTTrainerNative.kt:588 -> JNI -> `WeightMerger::merge_and_export_weights`) at the new per-tensor `inference/` layout. Note: the device currently writes to `inference/merged`; under this plan, merged trainable tensors overwrite the per-tensor `.bin` files **directly in `inference/`**, so the separate `merged` subdir becomes unnecessary (see Interactions — coordinate with the `loadMergedWeights` check).

10. **Deprecate `gen_genai`.** Once `export_inference_package.py` is the single entry, make `gen_genai` raise a deprecation warning and delegate, or remove and update imports.

## Interactions

- **`00_code_plans/09`** owns the merger registry + `build_merger_model`; this plan consumes them and owns the on-disk/device merge filename contract. The merger ONNX construction lives in 09; never reintroduce `create_*_merger_model{,_2}`.
- **`00_code_plans/07`** owns the `weight_handoff_map.json` schema + tensor codec; this plan *writes* the file and *reads* the codec. If field names here differ from 07, 07 wins — align.
- **File #10 (GenAI spike)** consumes the exact package this builder produces (per-tensor externals + genai_config). The spike's pass/fail depends on step 2 (file split) and step 5 (config entry).
- **File #11 (engine abstraction)** reads `genai_config.json`'s engine candidate and the handoff map; both engines point at the same `inference/` dir.
- **`ORTGeneratorNative` `loadMergedWeights`**: currently checks `<cacheDir>/<repoName>/inference/merged` (ORTGeneratorNative.kt:41-48). Under the new layout merged tensors live directly in `inference/` as the canonical `.bin` files, so this check must move to "handoff map present and every `externalDataLocation[role]` file exists with matching checksums" rather than a `merged/` dir probe. Coordinate this rename in File #11 / `00_code_plans/06`.
- **`session_cache.h`**: `AddExternalInitializers` (line ~702) and `session.use_ort_model_bytes_for_initializers=0` (line ~717) stay the native injection path; they now read the handoff map's `externalDataLocation`/`inferenceInitializerNames` instead of the old `<layer>/<tensor>.tensor` convention. Native still works even if GenAI never loads.
- **Optimum export (#7)** feeds the raw inference graph into `export_inference_package.py`.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/inference/test_handoff_map.py` — `handoff_map.py` builder/loader/validator round-trips one entry and `check_compat()` accepts a matching `inferenceInitializerNames`/`externalDataLocation`/dtype/shape.
- **Quantized naming regression** (`pytest tests/inference/test_quant_names.py`): assert merged `weight_quantized` / `weight_zero_point` / `weight_scale` filenames in the map equal what the inference graph references; specifically pin the scale filename to close the `safe_name` vs `.weight_scale` inconsistency.
- **HandoffMode fail-closed unit**: selecting `model_input` or `adapter` raises the explicit `NotImplementedError` (F7 stub), while `external_initializer` resolves.
- C++ change site **compiles**: `./gradlew :MobileTransformers:compileDebugKotlin` after the `weight_merger.cpp` handoff-map lookup edits (full link/run is Manual below).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Handoff-map validation smoke**: export a tiny model; every `requires_grad`/PEFT base layer maps to exactly one `inferenceInitializerNames[weight]` + `externalDataLocation[weight]`, dtype/shape/transpose/quant fields match the actual ONNX initializer. Fail closed otherwise.
- **Export-mode parity smoke**: export the same tiny model (SmolLM2-360M) in `external_initializer` and `model_input` modes; assert (a) external mode keeps trainable tensors as external initializers and base in one blob, (b) model_input mode removes them and adds matching graph inputs + sets `genaiInputNames`.
- **Base/trainable separation smoke**: assert no merger ever writes into `frozen_base.onnx.data`; assert each trainable `.bin` has a sibling `.sha256` after the offline merge.
- **Atomic-overwrite smoke (Python)**: kill the offline `artifact/merger.py` driver mid-write; assert either the old valid file or the new valid file is present (never a truncated `.bin`), verified by checksum.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Atomic-overwrite smoke (C++/device)**: kill the on-device merge mid-write; assert the same never-truncated invariant on device via `rename(2)` + checksum.
- **Offline-vs-device merge parity**: run `artifact/merger.py` offline and `WeightMerger` on device on the same checkpoint; assert byte-identical `.bin` outputs (or within documented quant tolerance) and identical filenames.
- **Native load smoke (device)**: `ORTGeneratorNative` loads the unified `inference/` package with `loadMergedWeights` and generates one token (the guaranteed-path regression guard before File #10 introduces GenAI).

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- A single `export_inference_package.py` entry produces `<cacheDir>/<model>/inference/` with `model.onnx` (external refs only), `genai_config.json`, `weight_handoff_map.json`, one immutable `frozen_base.onnx.data` (flat), and per-tensor `<name>.bin` (+ `.sha256`).
- Offline (`artifact/merger.py`) and device (`WeightMerger`) both write merged tensors to the **exact** filenames in the handoff map (no `_2`/string-rewrite names); both fail closed on any missing/mismatched entry.
- The hard-coded merger dispatch (`onnx_builder.py:628-641`) and C++ literal branches (`weight_merger.cpp:476-499, 526-712`) are gone, replaced by registry/handoff-map lookups; `gen_genai` is a deprecated shim or removed.

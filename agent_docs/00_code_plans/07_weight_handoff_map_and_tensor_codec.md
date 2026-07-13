# Weight Handoff Map & Trainable Tensor Codec

**Priority (global #):** 8  |  **Prerequisites:** `00_code_plans/02_config_layering_settings_constants.md` (#4), `00_code_plans/09_typed_models_enums_and_registries.md` (#6)  |  **Blocks:** `01_code_plans/01_unified_merger_and_external_data_export.md` (#9), `00_code_plans/06_manifest_first_package_and_cache_bridge.md` (#13)

> This plan **OWNS** the `weight_handoff_map.json` schema and the `TrainableTensorCodec`. Every other plan (#9, #13, #17, #18, #19, #22) references the contract defined here and must not redefine it.
>
> **Consumes `00_code_plans/09`.** The codec's tensor naming is no longer hand-rolled here: the adapter **component schema** (which roles exist per PEFT method and how to find them) comes from 09's `PEFTMethodSpec.component_schema`, and the per-architecture name-rewrite data (the `self_attn`→`attn` / `base_layer`→`MatMul` rules, attention-module name) comes from 09's architecture registry. This plan defines *how the map is built/validated*; 09 defines *the vocabulary the codec reads*.

---

## Purpose

Today the train→infer name translation is hard-coded in C++ (`WeightMerger::inference_name`, `weight_merger.cpp:904`): it strips `backbone.`, rewrites `self_attn`→`attn`, and `base_layer`→`MatMul`. That mapping is duplicated implicitly in three other places that must agree byte-for-byte but currently do not:

- **Save side (merge):** `WeightMerger::save_merged_parameters` (`weight_merger.cpp:814`) writes regular weights flat as `output_directory/<safe_name>.tensor` with the embedded TensorProto name set to `base_layer_name` (`weight_merger.cpp:882`), but writes quantized triplets into a **subdirectory** `<safe_name>/{weight_quantized,weight_zero_point,weight_scale}.tensor` with embedded names `safe_name + ".weight_quantized"`, `safe_name + ".weight_zero_point"`, but the scale as `base_layer_name + ".weight_scale"` (`weight_merger.cpp:855`). **This is the documented inconsistency: scale uses `base_layer_name`, the other two use `safe_name`.**
- **Load side (inference):** `WeightSessionCache::init` (`session_cache.h:54`) ignores the embedded TensorProto name entirely and reconstructs the initializer name from the filesystem as `layer_name + "." + tensor_filename` (directory name + file stem; `session_cache.h:71`), then feeds those names to `OrtSessionOptions::AddExternalInitializers` (`session_cache.h:702`).
- **Inference graph:** `inference/builder.py` emits MatMul weight initializers named `name[1:].replace("/", ".") + ".weight"` (`inference/builder.py:842/814`) — e.g. `model.layers.0.attn.q_proj.MatMul.weight` — plus int4 siblings `.qweight`, `.scales`, `.qzeros` (`inference/builder.py:866-874`).

The invariant that must hold: **the name the inference model consumes (initializer name in `model.onnx`, OR graph input name for GenAI) MUST equal the name produced on the save side, with matching dtype, shape, transpose state, and quantization role.** Right now nothing enforces it; a rename in `inference/builder.py` silently breaks merged-weight loading at runtime with no build-time signal.

This plan replaces the implicit four-way agreement with **one declarative artifact** emitted by the Python package builder and consumed (data-driven) by the device merge writer and both inference engines. Fail closed: if any consumer cannot satisfy an entry, it errors before a session is created rather than producing a silently-wrong model.

---

## Touched / new files

**Python (export toolkit — new):**
- `src/mobiletransformers/artifacts/handoff_map.py` — `TrainableTensorCodec`, `HandoffEntry`, `HandoffMap` dataclasses; emit + validate.
- `src/mobiletransformers/artifacts/manifest.py` — already planned in #13; imports the handoff dataclasses (do not duplicate).
- `tests/unit/test_handoff_map.py`, `tests/unit/test_tensor_codec.py`.

**Python (export toolkit — edit):**
- `inference/builder.py` (→ `src/mobiletransformers/export/inference_export.py`): after building the inference graph, record the actual emitted initializer names/dtypes/shapes per trainable MatMul so the codec uses **observed** names, not re-derived ones.
- `trainer/builder.py` (→ `src/mobiletransformers/export/training_export.py`): already emits `requires_grad`, `frozen_params`, `peft_mapping` into `training_config.json` (`trainer/builder.py:382-391`); feed `peft_mapping` + `requires_grad` into the codec as the checkpoint-name source of truth.

**Android C++ (edit):**
- `weight_merger.cpp` / `weight_merger.h`: add a handoff-map loader; route `save_merged_parameters` and `inference_name` through it. Keep `inference_name` as a deprecated fallback only when no map is present.
- `session_cache.h`: `WeightSessionCache::init` gains a map-driven path that reads the embedded TensorProto name (or the map's `inferenceInitializerNames`) instead of reconstructing from the filesystem.

**Android Kotlin (edit):**
- `ORTTrainerNative.mergeExportSessionWeights` (`ORTTrainerNative.kt:587`) and `mergeExportWeights` JNI signature (`ORTTrainerNative.kt:605`): pass the handoff-map path alongside `peftMappingPath`.

---

## Data contract — `weight_handoff_map.json` (CANONICAL, schemaVersion 1)

Lives at `<package>/.../train/weight_handoff_map.json` and is installed into `<cacheDir>/<sanitizedRepoId>/inference/weight_handoff_map.json` (co-located with the model it describes; see #13 for install). This is the single source of truth that **replaces `weight_merger.cpp:904`**.

```json
{
  "schemaVersion": "1.0",
  "minReaderVersion": "1.0",
  "handoffMode": "external_initializer",
  "engines": ["native", "genai"],
  "externalDataLayout": "one_file_per_tensor",
  "frozenBaseBlob": "frozen_base.onnx.data",
  "mergerModels": {"mars_q": "merger_mars_qin_qout.onnx", "lora": "merger_lora_fp.onnx"},
  "entries": [
    {
      "trainingBaseLayerName": "backbone.model.layers.0.self_attn.q_proj.base_layer",
      "checkpointNames": {"weight": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight"},
      "mergerOutputNames": {"weight": "merged_weight"},
      "mergedTensorNames": {"weight": "model.layers.0.attn.q_proj.MatMul.weight"},
      "inferenceInitializerNames": {"weight": "model.layers.0.attn.q_proj.MatMul.weight"},
      "externalDataLocation": {"weight": "model.layers.0.attn.q_proj.MatMul.weight.bin"},
      "sha256": {"weight": "<hex of last-written bytes, updated by each merge>"},
      "genaiInputNames": {},
      "dtype": "float16",
      "shape": [4096, 4096],
      "quantization": {"weightQuantizedName": "...", "scaleName": "...", "zeroPointName": "..."},
      "transposePolicy": "already_transposed_for_inference"
    }
  ]
}
```

> **This camelCase `entries[]` shape is the only handoff-map vocabulary.** An earlier draft of `01_code_plans/01` sketched a snake_case `tensors[]` shape (`training_param_names`, `external_file`, …); that sketch is superseded — #9, the #10 spike, #22, and #23 all read/write **this** schema. When any plan mentions "the external file for a role," it means `externalDataLocation[role]`.

Field semantics:

| Field | Meaning |
| --- | --- |
| `schemaVersion` / `minReaderVersion` | `"MAJOR.MINOR"` strings (F1). Readers preserve unknown fields and fail closed only when `major` exceeds support or their version is below `minReaderVersion`; one `check_compat()` helper, mirrored Python↔Kotlin/C++ — **exact algorithm below**. |

### `check_compat()` — the ONE canonical algorithm (F1; every schema-versioned contract uses exactly this)

This plan owns the schema-versioning contract, so the precise semantics live here once. Every reader of a versioned JSON (`weight_handoff_map.json`, `mobiletransformers_manifest.json`, `model_support_matrix.json`, `FederatedAdapterRecord`, generated `schemas/*.schema.json`) declares a compile-time constant `READER_SCHEMA_VERSION: "MAJOR.MINOR"` **per contract** (e.g. `HANDOFF_MAP_READER_VERSION = "1.0"` in Python, the same value in the Kotlin/C++ mirror — bumped only when the reader actually learns a new schema). Then:

```text
check_compat(docSchemaVersion, docMinReaderVersion, READER_SCHEMA_VERSION) -> accept | fail
  (docMajor, docMinor) = parse(docSchemaVersion)        # split on "."; both non-negative ints; malformed => fail closed
  (reqMajor, reqMinor) = parse(docMinReaderVersion)
  (rdrMajor, rdrMinor) = parse(READER_SCHEMA_VERSION)
  if docMajor >  rdrMajor:                       fail "document schema vX.Y needs a newer SDK (reader supports major rdrMajor)"
  if (rdrMajor, rdrMinor) < (reqMajor, reqMinor): fail "document requires reader >= minReaderVersion; this SDK is READER_SCHEMA_VERSION"
  accept                                          # docMajor < rdrMajor is fine (older doc); docMinor > rdrMinor is fine (additive fields, ignored)
```

Rules an implementer must not improvise around: comparison is on the **(major, minor) integer tuple**, never string comparison; a doc with a *lower* major than the reader is accepted (readers keep old-major compatibility until a deliberate drop, which is itself a major reader bump); unknown fields never influence the result; the two `fail` messages above are the required "needs newer SDK" wording family. Both mirrors are covered by a shared table-driven test fixture (`check_compat_cases.json`: doc version × min-reader × reader version → expected accept/fail).
| `handoffMode` | `HandoffMode` value (F7): `external_initializer` (the **only** mode supported in v1; both engines), or `model_input` (GenAI graph-input path) / `adapter` — both registry **stubs that fail closed** until implemented. Controls which name field must resolve. |
| `engines` | Engines this map is valid for. A consumer not listed must refuse to use it. |
| `externalDataLayout` | `one_file_per_tensor` — matches `WeightSessionCache` per-tensor `.tensor`/`.bin` files. |
| `frozenBaseBlob` | Filename of the immutable quantized base external blob (canonical: `frozen_base.onnx.data`, flat in `inference/`); never overwritten on-device merge. |
| `mergerModels` | Resolved `MergerVariant` → descriptive merger ONNX filename (emitted by 09's `build_merger_model`, e.g. `merger_mars_qin_qout.onnx` — no `_2` names). The C++ side builds its variant→session map from this instead of the hard-coded paths at `weight_merger.cpp:448-464`. |
| `entries[].trainingBaseLayerName` | Key into `peft_mapping` from `training_config.json` (`trainer/builder.py:386`); the PEFT base-layer path. |
| `entries[].checkpointNames` | Role→name as the tensor appears in the ORT `CheckpointState` (what `WeightMerger::extract_base_layer_params`, `weight_merger.cpp:269`, reads). |
| `entries[].mergerOutputNames` | Role→output tensor name of the merger ONNX graph (e.g. `merged_weight_quantized`); how the merge driver routes merger outputs to roles. |
| `entries[].mergedTensorNames` | Role→name the merge writer stamps into the saved TensorProto. **MUST equal `inferenceInitializerNames`** for `external_initializer` mode. |
| `entries[].inferenceInitializerNames` | Role→name the inference ONNX graph consumes (observed from `inference/builder.py`, not re-derived). |
| `entries[].externalDataLocation` | Role→on-disk per-tensor filename inside `inference/`. |
| `entries[].sha256` | Role→SHA-256 of the last-written bytes of that role's external file; refreshed by every merge (offline and on-device) and checked by the load-side precondition (#23). |
| `entries[].genaiInputNames` | Role→graph-input name; required & non-empty iff `handoffMode == model_input`. |
| `entries[].quantization` | Present iff the entry is quantized. Names for `weight_quantized`/`scale`/`zero_point`. **Resolves the scale-naming bug (see invariant below).** |
| `entries[].transposePolicy` | `already_transposed_for_inference` (writer must transpose), `no_transpose`, or `transpose_on_load`. Mirrors `force_transpose_inputs` branch in `inference/builder.py:813`. |

### The quantized-name invariant (resolves the `base_layer_name` vs `safe_name` bug)

`weight_merger.cpp:837-855` saves the quantized triplet with **mixed** naming: quantized weight + zero-point use `safe_name = inference_name(base_layer_name)`, but scale uses `base_layer_name + ".weight_scale"`. Meanwhile `WeightSessionCache::init` reconstructs names from `<dirname>.<filestem>` = `<safe_name>.weight_scale`. So the embedded scale name (`base_layer_name.weight_scale`) is **discarded** today and only works by accident because the loader ignores it. The handoff map makes all three explicit and identical to the inference-graph names:

- `quantization.scaleName`, `quantization.zeroPointName`, `quantization.weightQuantizedName` are **all** derived from the inference-graph observed names (the `.scales`/`.qzeros`/`.qweight` initializers from `inference/builder.py:866-874`), never from `base_layer_name`.
- The merge writer stamps these exact names into the TensorProto AND uses them to compute the external-data filename, so the loader-reconstructed name == map name == inference-graph name.
- **Validation (`HandoffMap.validate`, fail closed):** for every entry, for every role, assert `mergedTensorNames[role] == inferenceInitializerNames[role]` (when `external_initializer`); assert the quantized role names match the observed inference-graph initializer set; assert no two entries claim the same `externalDataLocation` or `inferenceInitializerNames`.

---

## Data contract — `TrainableTensorCodec` (Python)

Deterministic, round-trippable description of one trainable tensor. Lives in `handoff_map.py`.

```python
@dataclass(frozen=True)
class TensorSpec:
    name: str                 # canonical inference name
    dtype: str                # "float16" | "float32" | "int8" | "uint8" | "int4"
    shape: tuple[int, ...]
    role: str                 # "weight" | "weight_quantized" | "scale" | "zero_point"
    transpose_policy: str
    aggregation_role: str     # "merged_base_plus_adapter" | "frozen" | "adapter_only"
```

`TrainableTensorCodec` responsibilities (pure, no I/O):
- `from_peft_mapping(peft_mapping, requires_grad, observed_inference_inits, peft_spec, arch_spec) -> list[HandoffEntry]` — joins the three sources by base-layer name and produces one entry per trainable MatMul. The `peft_spec` (09's `PEFTMethodSpec.component_schema`) supplies which adapter roles/components exist instead of hardcoding `shared_A`/`adapter_A`; the `arch_spec` (09's `ArchitectureSpec`) supplies the name-rewrite rules.
- `canonical_inference_name(base_layer_name, arch_spec) -> str` — the **single** Python implementation of the `weight_merger.cpp:904` rules: strip leading `backbone.`, `self_attn`→`attn` (using `arch_spec.attention_module_name`), `base_layer`→`MatMul`. The rewrite rules are **data from 09's architecture registry**, not literals baked here. Used only to *seed* a lookup against observed names; the observed name wins on conflict (and a conflict raises, so drift is caught at build time, not runtime).
- `to_dict()` / `from_dict()` — stable key order, sorted entries by `inferenceInitializerNames["weight"]`, so the JSON is byte-deterministic for checksums (#13).
- `validate()` — enforces the invariants above; raises `HandoffMapError` with the offending entry.

Dtype/shape/order are deterministic: entries sorted by canonical weight name; roles within an entry ordered `weight, weight_quantized, scale, zero_point`.

---

## Implementation steps

### Python (build side)
1. Add `TensorSpec`, `HandoffEntry`, `HandoffMap`, `TrainableTensorCodec`, `HandoffMapError` to `artifacts/handoff_map.py`.
2. In `inference_export.py` (`inference/builder.py`), accumulate an `observed_inference_inits` list at each `make_external_tensor` call site for trainable MatMuls (`make_matmul_fp16_or_fp32`, `make_matmul_int4`): record `(name, dtype, shape, role, transposed?)`. Surface it from `build`.
3. In `training_export.py` (`trainer/builder.py`), after `create_lora_mapping`/`create_mars_adapter_mapping` (`trainer/builder.py:344-346`), pass `peft_mapping` + `requires_grad` into `TrainableTensorCodec.from_peft_mapping` together with `observed_inference_inits`.
4. Emit `weight_handoff_map.json` next to `training_config.json`; record its relative path so the manifest's `weightHandoff` pointer (#13) resolves.
5. Run `HandoffMap.validate()` during export; fail the export on any mismatch.

### Android C++ (merge/save side)
6. Add `WeightMerger::load_handoff_map(path)` storing `unordered_map<trainingBaseLayerName, HandoffEntry>`.
7. In `save_merged_parameters` (`weight_merger.cpp:814`): replace the `inference_name(base_layer_name)` calls and the hard-coded `+ ".weight_scale"` / `+ ".weight_quantized"` / `+ ".weight_zero_point"` suffixes (`weight_merger.cpp:838/847/855`) with `entry.mergedTensorNames[role]` for the embedded TensorProto name and `entry.externalDataLocation[role]` for the filename. Write directly into `inference/` (flat per-tensor), not the legacy `inference/merged/<safe_name>/` subdir layout.
8. Keep `inference_name` (`weight_merger.cpp:904`) compiled but only used when `load_handoff_map` fails — log a loud deprecation warning.

### Android C++ (load side)
9. In `WeightSessionCache::init` (`session_cache.h:54`): when a handoff map is present, iterate `entries[].externalDataLocation` and load each file, taking the initializer name from `inferenceInitializerNames[role]` (do **not** reconstruct from `<dirname>.<filestem>`, `session_cache.h:71`). Verify the loaded TensorProto's dtype/shape match the entry; abort the session on mismatch (fail closed). Then `AddExternalInitializers` (`session_cache.h:702`) as today.

### Android Kotlin
10. Thread the map path through `ORTTrainerNative.mergeExportSessionWeights` (`ORTTrainerNative.kt:587`) and `mergeExportWeights` JNI (`ORTTrainerNative.kt:605`).

---

## Interactions

- **#9 (unified merger / external-data export):** consumes the Python emitter from this plan; the merger's on-device external-data output layout is defined by `externalDataLayout` + `externalDataLocation` here.
- **#13 (manifest + cache bridge):** `MobileTransformersManifest.weightHandoff` points at this map; `ChecksumVerifier` hashes it; `ModelPackageInstaller` copies it into `inference/`.
- **#11 (engine abstraction) / GenAI:** the `model_input` mode + `genaiInputNames` define how the GenAI engine receives merged tensors as graph inputs vs initializers.
- **#18 (training lifecycle):** merge runs at end of training (`ORTTrainerNative.startTraining`, `mergeWeightsAtEnd`, `ORTTrainerNative.kt:369-387`); the lifecycle reports merge progress but the contract is owned here.

---

## Worked example

A minimal `weight_handoff_map.json` (illustrative — one non-quantized entry, `external_initializer` mode):

```json
{
  "schemaVersion": "1.0",
  "minReaderVersion": "1.0",
  "handoffMode": "external_initializer",
  "entries": [
    {
      "trainingBaseLayerName": "backbone.model.layers.0.self_attn.q_proj.base_layer",
      "mergedTensorNames": {"weight": "model.layers.0.attn.q_proj.MatMul.weight"},
      "inferenceInitializerNames": {"weight": "model.layers.0.attn.q_proj.MatMul.weight"},
      "dtype": "float16",
      "shape": [4096, 4096]
    }
  ]
}
```

Under `external_initializer`, `mergedTensorNames["weight"] == inferenceInitializerNames["weight"]` (the validated invariant), so the merge writer stamps exactly the name the inference graph consumes.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/unit/test_tensor_codec.py`: round-trip `TensorSpec`/`HandoffEntry` through `to_dict`/`from_dict`; assert byte-identical JSON across runs (deterministic order); assert dtype/shape/role/transpose preserved.
- `pytest tests/unit/test_handoff_map.py`:
  - `validate()` rejects an entry where `mergedTensorNames != inferenceInitializerNames` under `external_initializer`.
  - `validate()` rejects a quantized entry whose `scaleName` is derived from `base_layer_name` instead of the observed inference initializer (the documented bug fixture).
  - `validate()` rejects duplicate `externalDataLocation` / duplicate inference names.
  - `model_input`/`adapter` modes fail closed in v1 (F7); `model_input` additionally requires non-empty `genaiInputNames`.
  - `check_compat()` rejects a `schemaVersion` major beyond support or an unmet `minReaderVersion` (F1).
  - `from_peft_mapping` raises when canonical-derived name disagrees with observed inference init (drift detection).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **C++ smoke:** generate a tiny 2-layer fixture map; run `save_merged_parameters` map-driven, then `WeightSessionCache::init` map-driven; assert every `inferenceInitializerNames[weight]` loads and the loader does not fall back to filesystem reconstruction.
- **Cross-language golden:** a checked-in `weight_handoff_map.json` fixture parsed by both the Python `HandoffMap.from_dict` and the C++ loader; assert identical entry count, names, and dtypes.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- On a device, run an end-to-end merge (`ORTTrainerNative.mergeExportWeights` with the handoff-map path) and load the resulting `inference/` package; confirm generation works with map-driven external-initializer loading (no filesystem-reconstruction fallback). Full lifecycle owned by #18.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- `weight_handoff_map.json` (schemaVersion `"1.0"`, `external_initializer`) is the single source of truth that replaces `weight_merger.cpp:904`; emitted by the Python builder next to `training_config.json` and validated (`HandoffMap.validate`, fail closed) during export.
- The quantized-name invariant holds: `weight_quantized`/`scale`/`zero_point` names come from the observed inference-graph initializers, never from `base_layer_name`; the documented scale-naming bug is resolved.
- C++ save (`save_merged_parameters`) and load (`WeightSessionCache::init`) are map-driven; `inference_name` (`weight_merger.cpp:904`) remains only as a loud-deprecation fallback when no map is present.
- `HandoffMode` enumerates `external_initializer`/`model_input`/`adapter`, with `model_input`/`adapter` as fail-closed stubs in v1 (F7).

---

## Implementation notes — Python owner layer done (2026-07-13)

The **Python owner layer is implemented, tested, and green** (`make check`: 108 passed, 6 skipped). The
cross-boundary consumers (C++ merge/save, C++ load, Kotlin JNI) are deferred to their integration plans,
exactly as this plan's own Interactions section directs — nothing here was skipped, only sequenced.

### What landed (and where — the file inventory drifted from the plan header)
- **`src/mobiletransformers/artifacts/handoff_map.py`** — owns the schema + codec. Public surface:
  - `TensorSpec` (frozen) — as specified.
  - `HandoffEntry` (mutable dataclass) — snake_case fields; `to_dict()`/`from_dict()` map to/from the
    canonical **camelCase** wire keys (`trainingBaseLayerName`, `inferenceInitializerNames`, …). Extra
    helpers: `.roles`, `.is_quantized`, `.tensor_specs()`.
  - `HandoffMap` — `to_dict()`/`from_dict()`, `to_json()` (**byte-deterministic**: entries sorted by
    canonical weight name + `sort_keys`), `load(path)` (parse → `check_compat` → `validate`),
    `save(path)`, `validate()`.
  - `TrainableTensorCodec` — `canonical_inference_name(base_layer_name, arch_spec)` (the single Python
    impl of the `weight_merger.cpp:904` rewrite, driven by `ArchitectureSpec.attention_module_name`) and
    `from_peft_mapping(peft_mapping, requires_grad, observed_inference_inits, peft_spec, arch_spec)`.
- **`src/mobiletransformers/artifacts/versioning.py`** — the plan pinned the `check_compat` algorithm
  inline but did not name a module; it was **extracted to its own module** (`check_compat`,
  `parse_version`, `SchemaVersionError`) so every versioned contract (#13 manifest, #20 support matrix,
  #35 federated) imports one implementation. `HANDOFF_MAP_READER_VERSION = "1.0"` lives in `handoff_map.py`.
- **`tests/fixtures/check_compat_cases.json`** — the shared cross-language table fixture (the plan asked
  for it); consumed by `test_handoff_map.py` and to be mirrored by the Kotlin/C++ readers.
- **`tests/unit/test_handoff_map.py`** + **`tests/unit/test_tensor_codec.py`** (26 tests).

### Contracts the plan under-specified (now pinned by the implementation)
- **`ObservedInit`** (new frozen dataclass, `handoff_map.py`) is the codec's ground-truth input for a
  single emitted inference initializer: `(name, dtype, shape, role, transposed)`. The plan spoke of an
  `observed_inference_inits` "list of `(name, dtype, shape, role, transposed?)`"; this formalizes it. The
  inference-export accumulation (deferred) must produce a list of these.
- **`INFERENCE_SUFFIX_TO_ROLE`** (`handoff_map.py`) pins the inference-graph suffix → handoff role map:
  `weight→weight`, `qweight→weight_quantized`, `scales→scale`, `qzeros→zero_point`. The deferred
  accumulation uses this to tag each `ObservedInit.role`; recorded here so both sides share one mapping.
- **`trainingBaseLayerName` convention:** `from_peft_mapping` derives it as
  `peft_mapping_key + ".base_layer"` (the `peft_mapping` key is the LoRA/MARS module path without
  `.base_layer`; the base weight lives at `<key>.base_layer.weight`). This matches the plan's worked
  example. Revisit if a real device merge shows a different PEFT wrap path — the codec is the only place
  to change it.
- **`HandoffMapError`** in the plan text == **`HandoffError`** from `exceptions.py` (the established typed
  hierarchy). No new exception type was introduced; `validate()` raises `HandoffError`, version gating
  raises `SchemaVersionError` (both under `MobileTransformersError`).
- **`model_input`/`adapter` fail-closed:** `validate()` rejects both with "not supported in this version".
  The `model_input`-requires-`genaiInputNames` structural rule is moot in v1 (the mode can never be
  used) and lands with the GenAI model-input path (#11).

### Deferred to owning plans (not lost — tracked)
- **Build-side emit wiring:** accumulate `ObservedInit`s in the inference-graph builder
  (`make_matmul_fp16_or_fp32`/`make_matmul_int4` sites) and feed `peft_mapping`/`requires_grad` at export
  time, then `HandoffMap(...).save()` next to `training_config.json`. Blocked on the inference-builder
  migration (gated by the Optimum-vs-GenAI decision, same gate as #7's `inference/builder.py` rewrite).
- **C++ save/load** (`weight_merger.cpp` `save_merged_parameters` + `load_handoff_map`;
  `session_cache.h` map-driven `init`): rides with **#9** (merge/save) and **#23** (native load).
- **Kotlin JNI** thread-through (`ORTTrainerNative.mergeExportSessionWeights`/`mergeExportWeights`): **#18/#19**.
- **Integration tests** (C++ smoke; cross-language golden parsed by Python + C++): land with the C++
  consumers above. `check_compat_cases.json` is ready for the C++/Kotlin mirror side.
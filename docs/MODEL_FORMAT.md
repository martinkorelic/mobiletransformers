# Model Format

A MobileTransformers package is a self-describing directory: a top-level manifest plus one subtree per
**variant** (ABI × quantization × feature set). One package feeds **both** inference engines (the native
ORT runtime and the ONNX Runtime GenAI engine) — the engine is a selection over the same files, never a
separate download. This page is sourced from the manifest owner
(`src/mobiletransformers/hub/package_format.py` + `artifacts/manifest.py`) and the weight-handoff
owner (`src/mobiletransformers/artifacts/handoff_map.py`).

Two JSON contracts govern the on-disk shape: **`mobiletransformers_manifest.json`** (what the package
contains and how to select/download it) and **`weight_handoff_map.json`** (the single source of tensor
identity across train → merge → inference). Both are versioned and read fail-closed.

## Versioning contract (both files)

Every cross-boundary JSON carries a `schemaVersion` and `minReaderVersion` (`"MAJOR.MINOR"`) and is gated
by one shared `check_compat()` helper (`artifacts/versioning.py`):

- A reader **tolerates** unknown fields and minor bumps (additive changes are non-breaking).
- A reader **rejects** (raising `SchemaVersionError`) when the file's `major` exceeds the reader's support,
  or when the file's `minReaderVersion` is newer than the reader.

Current reader versions: manifest `MANIFEST_READER_VERSION = "1.0"`; handoff map
`HANDOFF_MAP_READER_VERSION = "1.0"`; package format `SCHEMA_VERSION = "1.0"` /
`ARTIFACT_FORMAT_VERSION = 1`. Devices enforce the contract via typed fail-closed parsing (they do **not**
run runtime JSON-schema validation); the `schemas/*.schema.json` files are CI parity artifacts.

## Package layout

```
<package>/
├── mobiletransformers_manifest.json      # the manifest (below)
├── tokenizer files (shared)
└── variants/
    └── <variant-id>/                      # e.g. cpu-int4
        ├── train/                         # training artifacts (training/eval/optimizer models, checkpoint)
        │   └── weight_handoff_map.json
        ├── inference/
        │   ├── model.onnx                 # inference graph (external initializers)
        │   ├── frozen_base.onnx.data      # frozen base weights blob
        │   ├── <tensor-name>.bin          # one flat file per trainable/merged tensor
        │   ├── <tensor-name>.bin.sha256   # integrity sidecar per tensor
        │   └── weight_handoff_map.json
        └── embedding/                     # optional RAG embedding subtree
```

On device the installed cache mirrors this per repo: `<cacheDir>/<sanitizedRepoId>/{train,inference,embedding}/…`.
`sanitize_repo_id()` maps a Hub repo id to that directory name; the installer writes into exactly this shape
so the runtime repositories discover models unchanged.

## `mobiletransformers_manifest.json`

Built deterministically by `build_manifest()` from the on-disk tree. Top-level fields:

| Field | Meaning |
| --- | --- |
| `schemaVersion` / `minReaderVersion` | version gate (above). |
| `baseModelId` | source HF repo id. |
| `exportedAt` / `mobiletransformersVersion` / `artifactFormatVersion` | provenance. |
| `architectures` / `supportedTasks` / `selectedTask` / `trustRemoteCode` | model provenance from export. |
| `optimumOnnxVersion` / `transformersVersion` / `onnxRuntimeTrainingVersion` / `onnxRuntimeGenAIVersion` | toolchain pins. |
| `peftMethods` / `quantization` | realized capabilities of this package. |
| `defaultVariant` | variant id chosen when the caller requests none. |
| `variants[]` | per-variant descriptors (below). |
| `downloadPlan` | per-variant, per-feature repo-relative glob patterns — lets a client size + fetch selectively before touching large ONNX blobs. |
| `requiredFiles` | files that must exist for the package to be usable (includes the default variant's `inference/model.onnx`). |
| `fileSizes` / `sha256` | stream-hashed integrity for every file. |
| `weightHandoff` | path to the default variant's `weight_handoff_map.json`. |
| `androidRuntime` | `{ minimumAndroidApi, recommendedDeviceMemoryMb, requiredAbis }`. |
| `license` | `{ framework, baseModelWeights, noticeFile }`. |

Each `variants[]` entry:

| Field | Meaning |
| --- | --- |
| `id` | variant id (e.g. `cpu-int4`) — `cpu-<quantization>`. **Names the training-side quantization; see the note below.** |
| `executionProvider` | `cpu` \| `xnnpack` \| `nnapi`. |
| `quantization` | `QInt8` \| `QUInt8` \| `int4`. The **requested** setting. |
| `supportedEngines` | subset of `["native", "genai"]`. |
| `abi` | target Android ABI. |
| `features` | feature groups present (`inference`, `training`, `rag`, `embedding`, …). |
| `minimumAndroidApi` / `recommendedDeviceMemoryMb` | device requirements. |
| `weightHandoff` | this variant's handoff-map path. |
| `paths` | per-feature subtree paths. |

### The variant id names the TRAINING quantization, not the inference graph's

A variant id is `cpu-<quantization>`, and `--quant` drives the **training** stage: the training graph is
weight-quantized (`quant_model.onnx`, dynamic per-channel) before `generate_artifacts` runs. The
**inference** export does not quantize — it ships whatever precision optimum exported, in practice fp32.

So a variant named `cpu-int4` legitimately contains a **uint8-quantized training graph beside an fp32
inference graph**. That is the design, not a packaging bug, but nothing declared it and the directory
name was the only (misleading) signal. Two things now make it explicit:

- `inference/optimum_config.json` carries **`inferenceGraphPrecision`**, *measured from the graph that
  shipped* (`artifacts/parameter_budget.py::describe_graph_precision`) — never inferred from the id.
- The export gates the gap numerically: `verify_train_inference_parity` runs identical tokens through
  both graphs and fails the export if the cross-entropy differs by more than 1.5 nats. Weight-only
  uint8 quantization moves it ~0.4 nats; a graph that lost its weights moves it to the uniform floor.

**Do not read precision off the variant id.** Read `inferenceGraphPrecision` for the inference half and
`quantization` for the training half.

### Validation and variant selection

- `MobileTransformersManifest.validate(package_dir)` version-gates the manifest, then asserts the selected
  variant's declared files (the `inference/` group and `weight_handoff_map.json`) exist on disk — fail-closed,
  naming the missing file, **before** any load.
- `select_variant(execution_provider, quantization, total_mem_mb, requested_engine)` hard-filters variants by
  requested ABI/EP/engine/memory and returns a `SelectedVariant`. (The soft preference layer — quant
  preference, download-size tie-break, storage-budget ceiling — lives in `hub/variant_select.py`.)

## `weight_handoff_map.json`

The single source of tensor identity. It replaces the implicit name-agreement between the merge writer,
the native load side, and the inference graph with one declarative artifact. Owned by
`artifacts/handoff_map.py`; every consumer — the merger, the manifest, the native load path and the federated exporter — reads
this shape and none may redefine it.

Document-level fields:

| Field | Value |
| --- | --- |
| `schemaVersion` / `minReaderVersion` | `"1.0"` (version gate above). |
| `handoffMode` | `external_initializer` — the **only** supported mode in v1 (`model_input` / `adapter` are fail-closed stubs). |
| `engines` | `["native", "genai"]` — the engines this layout serves. |
| `externalDataLayout` | `one_file_per_tensor`. |
| `frozenBaseBlob` | `frozen_base.onnx.data`. |
| `mergerModels` | resolved `MergerVariant` → merger ONNX filename. |
| `entries[]` | one entry per trainable MatMul (below). |

Each `entries[]` element is one trainable layer's full identity:

| Field | Meaning |
| --- | --- |
| `trainingBaseLayerName` | the training-side base layer (e.g. `backbone.…q_proj.base_layer`). |
| `dtype` / `shape` | the **weight-like** role's dtype (`float16`/`float32`/`int8`/`uint8`/`int4`) and shape. |
| `tensorDtypes` / `tensorShapes` | role → that role's **own** on-disk dtype/shape — see "Per-role dtype and shape". |
| `checkpointNames` | role → ORT-checkpoint tensor name (the frozen `weight` + the adapter A/B factors). |
| `mergerOutputNames` | role → the merger graph's output name. |
| `mergedTensorNames` | role → the name the on-device merger stamps. |
| `inferenceInitializerNames` | role → the initializer name in `inference/model.onnx`. |
| `externalDataLocation` | role → the flat per-tensor `.bin` filename in `inference/`. |
| `sha256` | role → integrity hash of the **shipped** (pre-merge) bytes — see "Checksum precedence". |
| `genaiInputNames` | role → GenAI input name (when GenAI consumes the tensor as an input). |
| `quantization` | optional `{ weightQuantizedName, scaleName, zeroPointName }`. |
| `transposePolicy` | `no_transpose` \| `already_transposed_for_inference` — how the on-disk weight is oriented relative to the training checkpoint. See "Weight orientation" below; **do not honour it without reading that section.** |

Roles are drawn from the fixed order `("weight", "weight_quantized", "scale", "zero_point")`.

### Invariants (enforced by `HandoffMap.validate()`, fail-closed)

- **Merged name equals inference name** for every role: `mergedTensorNames[role] == inferenceInitializerNames[role]`
  (the external-initializer contract — the writer stamps exactly the name the inference graph reads).
- **Quantized names come from the observed inference initializers**, never derived from `trainingBaseLayerName`
  (`quantization.weightQuantizedName/scaleName/zeroPointName` must equal the observed initializer names).
- **No two entries** may claim the same `externalDataLocation` file or the same inference initializer name.
- Serialization is byte-deterministic (`to_json()` sorts keys and sorts entries by canonical weight name), so
  the manifest's `sha256` over this file is stable.

### Per-role dtype and shape

Each `<name>.bin` is an **ONNX external-data blob: raw tensor bytes with no header**. Nothing in the
file describes its element type or dimensions, so the map is the device loader's only source — and it
must describe every role separately, because a quantized entry's roles do not share a layout:

| Role | Typical dtype | Shape |
| --- | --- | --- |
| `weight` | `float16` | the logical `[out, in]` |
| `weight_quantized` | `uint8` (int4 packed two-per-byte) | packed, narrower than the logical shape |
| `scale` | `float16` | one per quantization group |
| `zero_point` | `uint8` | one per quantization group |

`tensorDtypes[role]` / `tensorShapes[role]` carry each role's own pair, taken from the initializers as
actually observed at export (`_classify_initializers`). The entry-level `dtype`/`shape` describe the
weight-like role only and remain as the fallback for maps written before these fields existed — sound
for a single non-quantized role, which is why `validate()` rejects a quantized entry that omits them.

The device loader (`session_cache.h::load_tensor_raw`) **constructs** the tensor from this declaration
rather than checking a parsed header against it, and fails closed when the file size is not exactly
`numel × element_size`.

### Weight orientation

Two conventions meet at the merge, and they disagree:

| side | convention |
| --- | --- |
| training checkpoint (`base_layer.weight`, a PyTorch `nn.Linear`) | `[out_features, in_features]` |
| inference initializer (an ONNX `MatMul` right-hand side) | `[in_features, out_features]` |

`transposePolicy` records which one the on-disk `.bin` uses, relative to the checkpoint. It is
**observed, not declared**: the merger computes `base + scale · (adapter_B @ adapter_A)`, whose shape
is `(adapter_B.rows, adapter_A.cols)`. If that equals the weight's on-disk shape the two agree
(`no_transpose`); if it equals the reverse, the on-disk tensor is the transpose
(`already_transposed_for_inference`); anything else describes a delta that cannot be added to its own
weight and is refused at export.

**A square weight cannot decide its own orientation** — `[576,576]` satisfies both readings — so the
policy is resolved **package-wide** from the entries that are not square. One export uses one
convention throughout, and mixed conventions fail closed.

> ⚠️ **Consumers: derive the orientation, do not trust this field yet.** Every package exported before
> 2026-08-14 declares `no_transpose` unconditionally, because the producing side never assigned it —
> and that is wrong for all of them. The on-device merger deliberately observes orientation from the
> tensors themselves rather than reading this field, so that packages already in the wild keep working.
> The field became meaningful for packages exported after that date. A declaration is only safe to
> honour once no artifact in circulation carries a wrong value for it.

### Checksum precedence: the sidecar wins

Each `<name>.bin` has **two** possible integrity sources, and they answer different questions:

| Source | Written by | Covers |
| --- | --- | --- |
| `<name>.bin.sha256` (sidecar) | the device merger, `weight_merger.cpp::write_raw_tensor_atomic`, atomically on **every** merge | the **live** bytes currently on disk |
| `entries[].sha256[role]` (in the map) | the exporter, once, at package build | the **shipped** bytes as published |

**A reader must prefer the sidecar and fall back to the map.** After an on-device train→merge the
`.bin` and its sidecar are both rewritten, but the map is not — C++ only ever *reads*
`weight_handoff_map.json`. A reader that preferred the map would therefore compare post-merge bytes
against the pre-merge shipped digest and reject a perfectly correct merge.

Absence of both is fail-closed: the load gate throws rather than skipping verification. A stale
*sidecar* still fails, so the precedence never weakens the gate — it only picks the authority that is
actually kept current. Enforced by `HandoffPrecondition.loadMergedWeightsReady`
(`internal/runtime/HandoffPrecondition.kt`).

The map is produced offline by `TrainableTensorCodec.from_peft_mapping(...)` (which joins the training-side
`peft_mapping` with the *observed* inference initializers — a naming drift raises at build time, never
silently at runtime) and consumed on device by the map-driven load path.

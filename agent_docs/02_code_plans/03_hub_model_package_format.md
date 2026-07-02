# Hub Model Package Format

**Priority #14 | Prerequisites: #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`) | Blocks: #15 (one-command export CLI), #21 (hub pull + cache flow), #22 (adapter push-back)**

## Purpose

Define the on-Hub layout of a "MobileTransformers-ready" HuggingFace model repository, and the exact schema of `mobiletransformers_manifest.json` that drives it. The package is **manifest-first**: any consumer (Python CLI, Android downloader, sample app) fetches one small JSON before touching large ONNX blobs, then resolves exactly which files to pull from a `downloadPlan` keyed by feature.

This plan owns the *Hub repo shape* and the *manifest field list*. It does **not** own:
- the manifest validator / fixture or the cache-bridge mapping — those live in #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`);
- the `weight_handoff_map.json` schema — owned by #8 (`00_code_plans/07_weight_handoff_map_and_tensor_codec.md`);
- the per-tensor external-initializer merge contract — owned by #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`).

This plan **references** those contracts and pins them into the repo layout.

### Canonical inheritance

- **One shared package, dual engine.** The same package directory and the same materialized Android cache folder are consumable by **both** the native ORT engine and the ONNX Runtime GenAI engine. The *variant* (not the package) declares which engines it supports. GenAI is selectable; Native is the guaranteed path.
- **Per-tensor external initializers, FLAT layout.** Trainable/merged weights are ONNX external initializers, one-file-per-tensor, living **flat** in `inference/` beside `model.onnx`. The frozen quantized base is the separate immutable blob `inference/frozen_base.onnx.data`. On-device merge overwrites the per-tensor files (atomic rename + checksum). (Today's `ORTTrainerNative.mergeExportSessionWeights()` writes into `<repo>/inference/merged` — that is the **legacy** behavior #9/#23 retire; there is no `base/` or `merged/` subdir in this format.)
- **Apache-2.0** is the framework license. Base-model weights keep their upstream license, declared per package in the manifest `license` field and restated in the model card.

## Touched / new files

New (authored by this plan; emitted by the export CLI in #15, validated by #13):

- `docs/HUB_PACKAGE_FORMAT.md` — human-facing spec mirror of this plan (added in Tier 1 step 14).
- `agent_docs/fixtures/tiny_package/` — tiny on-disk fixture repo used by #13 validator and #21 pull smoke. Contains a manifest, stub configs, and zero-byte/placeholder ONNX files sized to match `fileSizes`.
- `mobiletransformers/hub/package_format.py` (Python pkg root scaffolded by #1) — module of constants: `SCHEMA_VERSION`, `REQUIRED_TOP_LEVEL_FILES`, `FEATURE_GROUPS = ("core","inference","train","rag","genai","checksums")`, `VARIANT_SUBDIRS = ("train","inference","embedding")`, `sanitize_repo_id(repo_id)` — **the single canonical definition of the algorithm**, precisely: (1) replace every `/` with `__` (double underscore); (2) replace every remaining character NOT in `[A-Za-z0-9._-]` with a single `_`; (3) no trimming, no case-folding, no length cap (HF repo ids are already length-bounded). Deterministic and order-independent — e.g. `mobiletransformers/Qwen2-0.5B-mobile` → `mobiletransformers__Qwen2-0.5B-mobile`. The Kotlin sanitizer (`00_code_plans/06`/#21) must match byte-for-byte; the shared `sanitize_repo_id_cases.json` fixture (checked in with edge cases: unicode, spaces, `..`, leading dots) is the parity oracle.

Existing files that define the contract this format must stay compatible with:

- `artifact/onnx_builder.py` → `convert_pipeline()` already emits `build/train/{training_model.onnx,eval_model.onnx,optimizer_model.onnx,checkpoint,training_config.json}`, `build/inference/{generation_config.json, model + .onnx.data}`, `build/tokenizer/`, `build/embedding/rag_config.json`. The package layout is a reorganization of these outputs.
- `inference/builder.py` → `make_genai_config()` emits `genai_config.json`; `create_model()` emits `model.onnx` + `model.onnx.data`.
- Android `LLMRepository` expects, under `<cacheDir>/<repoName>/`: `tokenizer/`, `train/training_config.json`, `inference/generation_config.json`, `embedding/rag_config.json` (see `LLMRepository.updatePaths()` and `ORTTrainerNative` paths).

## Data contracts / interfaces

### Repository shape (on the Hub)

```text
mobiletransformers/Qwen2-0.5B-mobile/
  README.md                         # model card: base model, license, support matrix, ORT/GenAI versions, device limits
  mobiletransformers_manifest.json  # MANIFEST-FIRST entrypoint; fetched before anything large
  licenses/
    BASE_MODEL_LICENSE              # upstream weight license (e.g. Apache-2.0 / Qwen license)
    FRAMEWORK_LICENSE               # Apache-2.0 (MobileTransformers)
  optimum/
    export_report.json             # provenance: optimum-onnx version, transformers version, task, timings
    supported_tasks.json           # TasksManager output for this model_type
    optimum_config.json            # export knobs actually used (precision, opset, force_unpacked_matmul, etc.)
  shared/                          # files identical across variants → declared once, downloaded once
    tokenizer/                     # tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json, ortmobile_tokenizer_config.json
    chat_template.jinja            # chat template (split out so Android can render without the full tokenizer)
    generation_config.json         # HF generation defaults (sampling, eos/pad) — variant inference/ may override
    config.json                    # HF AutoConfig dump (model_type, n_layers, heads, etc.)
  default/                         # the simplest known-good package; a symlink-style alias of one variant id
    train/        ...              # (see variant layout below)
    inference/    ...
    embedding/    ...
    checksums.json
  variants/
    cpu-int4/                      # mobile default: int4 weights, acc-level-4
      train/
        training_model.onnx
        eval_model.onnx
        optimizer_model.onnx
        checkpoint/                # ORT CheckpointState dir
        training_config.json       # requires_grad, peft_mapping, rank, alpha, peft_target, trainable_parameter_count, peftMethod, modelId
        trainable_parameters.json  # list of per-tensor trainable initializer names (one-file-per-tensor manifest)
        weight_handoff_map.json    # train→infer tensor contract (schema owned by #8/07)
        merger_*.onnx              # on-device merger graphs — descriptive registry filenames from 09's
                                   # build_merger_model (e.g. merger_mars_qin_qout.onnx), recorded in the
                                   # handoff map's mergerModels; NO legacy lora_/mars_(q)merger_model names
        <task>.jsonl               # optional bundled fine-tune dataset (export_dataset)
      inference/                   # FLAT canonical layout — no base/ or merged/ subdirs
        model.onnx                 # graph; initializers are external refs only
        frozen_base.onnx.data      # frozen quantized base — immutable external blob
        model.layers.0.attn.q_proj.MatMul.weight.bin        # per-tensor trainable/merged externals
        model.layers.0.attn.q_proj.MatMul.weight.bin.sha256 # (+ sibling checksums; one pair per tensor)
        weight_handoff_map.json    # installed copy the engines read (source copy under train/)
        generation_config.json     # MobileTransformers inference config: { "type": "native" | "genai", ... }
        genai_config.json          # present only if variant supports the GenAI engine
        session_options.json       # ORT SessionOptions hints (threads, graph opt level)
      embedding/                   # present only on rag-capable variants
        embedding_model.onnx
        tokenizer/
        rag_config.json            # embeddingDimension, etc.
      checksums.json               # sha256 per file for THIS variant (mirror of manifest.sha256 subset)
    cpu-fp16/                      # desktop-smoke / higher-accuracy variant
      train/ inference/ embedding/ checksums.json
```

Notes:
- `default/` is the manifest-declared `defaultVariant`. Emit it either as a real copy or as a manifest alias (`downloadPlan` paths point at `variants/<defaultVariant>/...`). Prefer alias to avoid doubling repo size; only materialize a real `default/` copy if Xet dedup is unavailable.
- The flat `inference/` dir realizes the per-tensor-external + frozen-base split: Android merge overwrites the per-tensor `.bin` files in place (atomic rename + checksum); `frozen_base.onnx.data` is never rewritten on device.
- Variant ids are free-form but should encode `<ep>-<quant>` and optionally engine, e.g. `cpu-int4`, `cpu-fp16`, `cpu-int4-genai`.

### `mobiletransformers_manifest.json` — full field list

```jsonc
{
  "schemaVersion": "1.0",                    // "MAJOR.MINOR"; additive minor bumps are non-breaking
  "minReaderVersion": "1.0",                 // oldest SDK/CLI that can safely read this manifest
  "baseModelId": "Qwen/Qwen2-0.5B",         // upstream HF id the package was exported from
  "exportedAt": "2026-06-24T10:00:00Z",
  "mobiletransformersVersion": "0.1.0",
  "artifactFormatVersion": 1,                // bump when train/inference dir layout changes

  // --- support / provenance (mirrors optimum/export_report.json) ---
  "architectures": ["Qwen2ForCausalLM"],
  "supportedTasks": ["text-generation", "text-generation-with-past"],
  "selectedTask": "text-generation-with-past",
  "trustRemoteCode": false,
  "optimumOnnxVersion": "0.1.0",
  "transformersVersion": "4.x.y",
  "onnxRuntimeTrainingVersion": "1.x.y",     // source-built ORT-training wheel used for train artifacts
  "onnxRuntimeGenAIVersion": "0.x.y",        // null if no GenAI variant
  "peftMethods": ["mars", "lora"],
  "quantization": ["int4", "fp16"],

  // --- variant catalogue ---
  "defaultVariant": "cpu-int4",
  "variants": [
    {
      "id": "cpu-int4",
      "executionProvider": "cpu",
      "quantization": "int4",
      "supportedEngines": ["native", "genai"],   // ONE package; variant declares engines
      "abi": ["arm64-v8a"],                        // device ABIs this variant's blobs target (null = any)
      "features": ["core", "inference", "train", "rag", "genai"],
      "minimumAndroidApi": 28,
      "recommendedDeviceMemoryMb": 3072,
      "weightHandoff": "variants/cpu-int4/inference/weight_handoff_map.json",
      "paths": { "train": "variants/cpu-int4/train", "inference": "variants/cpu-int4/inference", "embedding": "variants/cpu-int4/embedding", "tokenizer": "shared/tokenizer" }
    },
    {
      "id": "cpu-fp16",
      "executionProvider": "cpu",
      "quantization": "fp16",
      "supportedEngines": ["native"],
      "abi": null,
      "features": ["core", "inference", "train"],
      "minimumAndroidApi": 28,
      "recommendedDeviceMemoryMb": 6144,
      "weightHandoff": "variants/cpu-fp16/inference/weight_handoff_map.json",
      "paths": { "train": "variants/cpu-fp16/train", "inference": "variants/cpu-fp16/inference", "tokenizer": "shared/tokenizer" }
    }
  ],

  // --- per-variant download plan, grouped by feature ---
  // Keyed by variant id. Each value is the FEATURE_GROUPS map. Paths are repo-relative
  // glob patterns suitable for huggingface_hub.snapshot_download(allow_patterns=...).
  "downloadPlan": {
    "cpu-int4": {
      "core":      ["mobiletransformers_manifest.json", "shared/tokenizer/**", "shared/chat_template.jinja", "shared/config.json", "shared/generation_config.json"],
      "inference": ["variants/cpu-int4/inference/**"],
      "train":     ["variants/cpu-int4/train/**"],
      "rag":       ["variants/cpu-int4/embedding/**"],
      "genai":     ["variants/cpu-int4/inference/genai_config.json"],
      "checksums": ["variants/cpu-int4/checksums.json"]
    }
  },

  // --- integrity / sizing (keyed by repo-relative path) ---
  "requiredFiles": ["mobiletransformers_manifest.json", "shared/tokenizer/tokenizer.json", "variants/cpu-int4/inference/model.onnx", "..."],
  "fileSizes": { "variants/cpu-int4/inference/frozen_base.onnx.data": 412000000, "...": 0 },
  "sha256":    { "variants/cpu-int4/inference/frozen_base.onnx.data": "ab12...", "...": "" },
  "etag":      { "variants/cpu-int4/inference/frozen_base.onnx.data": "\"ab12...\"" },  // optional, from Hub HEAD

  // --- top-level pointer to the active handoff (default variant) ---
  "weightHandoff": "variants/cpu-int4/inference/weight_handoff_map.json",

  // --- android runtime requirements (top-level summary; per-variant overrides win) ---
  "androidRuntime": {
    "minimumAndroidApi": 28,
    "recommendedDeviceMemoryMb": 3072,
    "requiredAbis": ["arm64-v8a"]
  },

  "license": {
    "framework": "Apache-2.0",
    "baseModelWeights": "Apache-2.0",          // restate exact upstream license string per model
    "noticeFile": "licenses/BASE_MODEL_LICENSE"
  }
}
```

Field semantics that matter for consumers:
- `schemaVersion` / `minReaderVersion` (**F1 schema versioning**): `schemaVersion` is `"MAJOR.MINOR"`; readers preserve unknown fields, so additive minor bumps (a new manifest key, a new variant attribute) are non-breaking. A reader fails closed with a "needs newer SDK" message only when the manifest's `major` exceeds what it supports or its own version is below `minReaderVersion`. One `check_compat()` helper is mirrored Python↔Kotlin so the CLI, Python pull, and Android downloader all gate identically (exact algorithm + `READER_SCHEMA_VERSION` convention pinned in `00_code_plans/07`).
- `downloadPlan[variant][group]` is the **only** thing a downloader needs to convert "I want inference + rag" into an `allow_patterns` list — no path knowledge baked into clients.
- `requiredFiles` is the minimal set whose absence fails validation regardless of requested features (manifest + tokenizer + base inference graph).
- `sha256` / `fileSizes` are the integrity source of truth; `checksums.json` per variant is a redundant copy so a variant subtree is self-validating after extraction.
- `weightHandoff` (top-level) always points at the **default variant's** map; per-variant `weightHandoff` is authoritative when a non-default variant is selected.

### `weight_handoff_map.json` (referenced, schema owned by #8/07)

The copy consumers resolve lives at `variants/<id>/inference/weight_handoff_map.json` (the `weightHandoff` pointer targets this one, and it ships in the `inference` download group so inference-only pulls carry it); the exporter also emits the source copy next to `training_config.json` under `train/`. It is the contract that replaces hard-coded name rewrites (e.g. `weight_merger.cpp:904`). For `handoffMode = "external_initializer"` (the canonical dual-engine mode), each entry maps a training checkpoint tensor name → the per-tensor external initializer `.bin` file flat in `inference/` (`externalDataLocation[role]`). Android consults it during merge instead of string-replacing names. This plan only pins its **location** and requires that every `externalDataLocation` it references resolves to a real file in the variant's `inference/` dir (validated by #13).

## Implementation steps

1. Add `mobiletransformers/hub/package_format.py` with the constants, `FEATURE_GROUPS`, `VARIANT_SUBDIRS`, and `sanitize_repo_id()`. Keep `sanitize_repo_id` byte-identical to the Kotlin sanitizer used in #21/cache bridge (single source: document the algorithm here and in #13).
2. Write a `build_manifest(package_dir, variants, base_model_id, export_report) -> dict` helper that walks the package tree, computes `fileSizes` + `sha256` (stream-hash, 1 MiB chunks), and assembles `downloadPlan` from `FEATURE_GROUPS` × variant subtrees. The export CLI (#15) calls this as its final emit step.
3. Define the mapping from `convert_pipeline()`/`create_model()` outputs into the variant tree: `build/train/*` → `variants/<id>/train/`, the inference graph → `variants/<id>/inference/model.onnx` with the frozen base as `frozen_base.onnx.data` and per-tensor trainable `.bin` files flat beside it (#9's layout), `build/tokenizer` → `shared/tokenizer`, `build/embedding` → `variants/<id>/embedding`. Pull `chat_template.jinja` out of the tokenizer (use `tokenizer.chat_template`).
4. Emit `optimum/export_report.json`, `optimum/supported_tasks.json`, `optimum/optimum_config.json` from the export run's metadata (TasksManager output + the resolved `inference_export_config`).
5. Emit per-variant `checksums.json` (subset of manifest `sha256` scoped to that variant subtree) so a downloaded variant is independently verifiable.
6. Build the fixture `agent_docs/fixtures/tiny_package/`: a `cpu-int4` variant with placeholder ONNX files whose byte sizes match `fileSizes`, a real tiny tokenizer, and a real (tiny) `weight_handoff_map.json`. This is the shared fixture for #13 validator tests and #21 pull/downloader smokes.
7. Document `default/` aliasing policy in `package_format.py` docstring and `HUB_PACKAGE_FORMAT.md`.

## Interactions

- **#13 (manifest-first package + cache bridge):** owns the validator that asserts this schema; owns the path mapping that materializes `variants/<id>/{train,inference,embedding}` → `<cacheDir>/<sanitizedRepoId>/{train,inference,embedding}` plus copied `mobiletransformers_manifest.json` and `checksums.json`. The Android `LLMRepository` then sees its expected `train/training_config.json`, `inference/generation_config.json`, `embedding/rag_config.json` unchanged.
- **#15 (export CLI):** produces a package in exactly this shape and calls `build_manifest()`; emits `optimum/export_report.json`.
- **#21 (hub pull + cache flow):** Python pull derives `allow_patterns` from `downloadPlan`; Android downloader fetches manifest first, selects a variant by `abi`/`quantization`/`supportedEngines`/`features`/`recommendedDeviceMemoryMb`, then downloads the variant's grouped files.
- **#22 (adapter push-back):** an exported adapter is published into a `variants/<id>/train/` subtree (or a dedicated adapter repo) and references the same `weight_handoff_map.json` contract.
- **#8/07:** owns `weight_handoff_map.json` schema; this plan pins its location and the resolvability invariant.
- **#9/01:** owns the frozen-blob (`frozen_base.onnx.data`) + per-tensor `.bin` flat split; this plan pins where those land in the repo.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `sanitize_repo_id()` parity test (`pytest tests/hub/test_package_format.py`): a table of repo ids (`mobiletransformers/Qwen2-0.5B-mobile`, `SmolLM2-360M-mobile`, `TinyLlama-1.1B-mobile`, `all-MiniLM-L6-v2-embedding-mobile`) → expected sanitized dir names; must match the Kotlin sanitizer (assert against a checked-in `sanitize_repo_id_cases.json` shared with #21).
- Variant-selection invariants (pure-Python helper shared with #21): given device constraints, the chosen variant's `features` cover the requested feature set and `abi` is compatible.
- Schema-version test (**F1**): a manifest with `schemaVersion="1.0"` + an unknown extra field reads cleanly (unknown field preserved); `check_compat()` fails closed with a "needs newer SDK" message when `major` exceeds support or `minReaderVersion` is unmet.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `build_manifest()` round-trip: build over the tiny fixture → assert `requiredFiles` present, every `sha256`/`fileSizes` key exists on disk, `downloadPlan` groups reference only existing paths.
- Handoff resolvability: every `externalDataLocation` in `weight_handoff_map.json` resolves to a file under `variants/<id>/inference/` (this assertion is owned by #13 but the fixture supports it here).
- Dual-engine sanity: a variant advertising `supportedEngines: ["native","genai"]` has both `inference/generation_config.json` (`type: native`) and `inference/genai_config.json`; a `["native"]`-only variant omits `genai_config.json` and its `downloadPlan.genai` group is empty.
- Fixture lint: `agent_docs/fixtures/tiny_package/` validates clean against the #13 validator in CI.

**Definition of done** — the Hub repo shape and the full `mobiletransformers_manifest.json` field list are specified and mirrored in `docs/HUB_PACKAGE_FORMAT.md`; `package_format.py` exposes the constants + a `sanitize_repo_id` byte-identical to the Kotlin sanitizer; `build_manifest()` over the tiny fixture emits a manifest whose `requiredFiles`/`sha256`/`fileSizes`/`downloadPlan` all reference real on-disk paths and carries `schemaVersion`/`minReaderVersion`; and `agent_docs/fixtures/tiny_package/` validates clean against the #13 validator.

# Manifest-First Package Model & Cache-Install Bridge

**Priority (global #):** 12  |  **Prerequisites:** `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` (#7), `01_code_plans/01_unified_merger_and_external_data_export.md` (#8)  |  **Blocks:** `02_code_plans/03_hub_model_package_format.md` (#13), `02_code_plans/05_one_command_export_cli.md` (#14), `00_code_plans/05_android_facade_foundation.md` (#16), `02_code_plans/04_hub_pull_and_cache_flow.md` (#20)

---

## Purpose

`LLMRepository` (`LLMRepository.kt:76`) already discovers usable models purely by **filesystem convention**: `availableModels` lists every directory under `cacheDir` (`LLMRepository.kt:97-101`), and `updatePaths()` (`LLMRepository.kt:182`) probes for exactly these files per model:

- `"$cacheDir/$modelName/train/training_config.json"` → sets `isTrainingAvailable`
- `"$cacheDir/$modelName/inference/generation_config.json"` → sets `isGenerationAvailable`
- `"$cacheDir/$modelName/embedding/rag_config.json"` → sets `isRagAvailable`
- `"$cacheDir/$modelName/tokenizer"` → tokenizer dir

The repository never reads a manifest. So the goal here is **not** to change `LLMRepository`: it is to add a manifest-first package format plus an installer that **materializes a downloaded/installed package INTO exactly this cache shape**, so the existing repository classes keep working unchanged. The manifest lets the SDK understand, select, and validate a package *before* downloading large files; the installer is the bridge that lands the right files at the conventional paths.

Canonical cache/package layout (both inference engines read the **same** folder):

```
<cacheDir>/<sanitizedRepoId>/
  train/          training_model.onnx, eval_model.onnx, optimizer_model.onnx,
                  checkpoint, training_config.json, training_state.json (post-train),
                  <merger>.onnx, weight_handoff_map.json (source copy)
  inference/      model.onnx, genai_config.json, generation_config.json,
                  frozen_base.onnx.data (immutable), per-tensor trainable *.bin,
                  weight_handoff_map.json
  embedding/      rag_config.json, embedding model files
  tokenizer/      tokenizer.json, tokenizer_config.json, ...
  mobiletransformers_manifest.json
  checksums.json
```

`<sanitizedRepoId>` is the repo id with `/`→`_` (and other FS-unsafe chars stripped), so it is a single directory name under `cacheDir` — i.e. the `modelName` that `LLMRepository.availableModels` returns.

---

## Touched / new files

**Android Kotlin (new), package `com.martinkorelic.ortmobile.package`:**
- `MobileTransformersManifest.kt` — data class + Gson (de)serialization (project already uses Gson, see `ORTTrainerNative.kt:7`).
- `ManifestValidator.kt` — schema + required-file presence + version-compat checks.
- `VariantSelector.kt` — capability/variant selection.
- `ChecksumVerifier.kt` — SHA-256 over `requiredFiles` vs `checksums.json`.
- `ModelPackageInstaller.kt` — atomic staging → cache-shape materialization.
- `CacheIndex.kt` — enumerate installed packages + manifest metadata + storage usage.

**Android Kotlin (light edit):**
- `LLMRepository.kt` — **no behavioral change required**. Optional: a secondary ctor / companion that, given a `sanitizedRepoId`, sets `modelName` (uses existing `modelName` setter validation, `LLMRepository.kt:83-92`). Do not move the filesystem probing.

**Python (export toolkit, new — shared schema source of truth):**
- `src/mobiletransformers/artifacts/manifest.py` — `MobileTransformersManifest` dataclass + emitter + validator (imports handoff dataclasses from #7's `handoff_map.py`).
- `tests/unit/test_manifest.py`.

---

## Data contract — `mobiletransformers_manifest.json`

Builds on the field sketch in `02_tier1_hf_integrated_core.md:281-356`. Required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `schemaVersion` | int | Validator rejects unknown majors. |
| `baseModelId` | string | HF repo id of the base model. |
| `variants` | list\<Variant\> | See below. |
| `defaultVariant` | string | Must be a `variants[].id`. |
| `downloadPlan` | object | `{ "groups": { "core": [...], "inference": [...], "train": [...], "rag": [...], "genai": [...], "checksums": [...] } }` (glob list per feature; `02_tier1...:312-325`). |
| `requiredFiles` | list\<string\> | Relative paths that MUST exist post-install for the package to be valid. |
| `sha256` | map\<path,hex\> | Per-file digest; mirrored into `checksums.json`. |
| `weightHandoff` | object | Pointer `{ "path": "variants/<id>/inference/weight_handoff_map.json", "handoffMode": "external_initializer" }`. Map schema **owned by #7**, not redefined here. |
| `androidRuntime` | object | `{ "minimumAndroidApi": int, "abis": ["arm64-v8a", ...], "recommendedDeviceMemoryMb": int }`. |
| `onnxRuntimeTrainingVersion` | string | Train wheel provenance. |
| `onnxRuntimeGenAIVersion` | string? | Present iff a GenAI variant exists. |
| `supportedTasks` | list\<string\> | e.g. `["text-generation","feature-extraction"]` (matches `trainer/builder.py` `task_type`). |
| `selectedTask` | string | Default task. |
| `trustRemoteCode` | bool | Mirrors `trust_remote_code=True` used in `trainer/builder.py:255-258`. |
| `license` | string | SPDX; framework Apache-2.0, weights keep upstream license. |

**Variant** object:
```json
{
  "id": "cpu-int4",
  "abi": ["arm64-v8a"],
  "quantization": "int4",
  "minMemoryMb": 2048,
  "features": ["inference", "train", "rag"],
  "engines": ["native", "genai"],
  "paths": { "inference": "variants/cpu-int4/inference", "train": "variants/cpu-int4/train", "embedding": "variants/cpu-int4/embedding", "tokenizer": "shared/tokenizer" }
}
```

`checksums.json`: `{ "<relativePath>": "<sha256hex>", ... }` over every `requiredFile` of the selected variant + core.

---

## Implementation steps

### Selection & validation (no download yet)
1. `MobileTransformersManifest.parse(json)` (Gson) → typed object; `ManifestValidator.validate` checks `schemaVersion` major, `defaultVariant ∈ variants`, every `weightHandoff.path` resolvable, and that each variant's `paths` cover the features it claims.
2. `VariantSelector.select(manifest, deviceCaps, requestedFeatures, requestedEngine)`:
   - Filter variants whose `abi` intersects `Build.SUPPORTED_ABIS`.
   - Filter by `quantization` acceptable + `minMemoryMb <= ActivityManager.MemoryInfo.totalMem` (the trainer already reads memory via `ActivityManager`, `ORTTrainerNative.kt:3/496`).
   - Filter by `features ⊇ requestedFeatures` and `engines ∋ requestedEngine` (default `native`, the guaranteed path).
   - Tie-break: smallest `minMemoryMb`, then `defaultVariant`. Return a `SelectedVariant` or a typed `NoCompatibleVariant` error (fail closed).

### Download plan → staging
3. From `SelectedVariant`, compute the file set = `downloadPlan.groups.core` + groups for each requested feature + `checksums`. (The actual network fetch is #20; this plan defines the plan + the install bridge and works against an already-staged dir for testing.)
4. Download/copy into a staging dir `<cacheDir>/.staging/<sanitizedRepoId>/` (sibling of the final dir so rename is same-filesystem and atomic).

### Verify
5. `ChecksumVerifier.verify(stagingDir, manifest.sha256 ∩ selectedVariant files)` — SHA-256 each `requiredFile`; abort + delete staging on any mismatch. Verify the handoff map's own checksum (it gates correctness per #7).

### Materialize into cache shape (the bridge)
6. `ModelPackageInstaller.install`: map variant `paths` → conventional layout:
   - `variants/<id>/train/**` → `<dir>/train/**`
   - `variants/<id>/inference/**` → `<dir>/inference/**` (includes `model.onnx`, `genai_config.json`, `generation_config.json`, `frozen_base.onnx.data`, per-tensor `*.bin`, `weight_handoff_map.json`)
   - `variants/<id>/embedding/**` → `<dir>/embedding/**`
   - `shared/tokenizer/**` → `<dir>/tokenizer/**`
   - manifest + `checksums.json` → `<dir>/` root
7. Assert the resulting dir satisfies `LLMRepository.updatePaths()` probes (`train/training_config.json`, `inference/generation_config.json`, `embedding/rag_config.json`, `tokenizer/`) for the features the variant claims. If a claimed feature's probe file is missing, fail and roll back.
8. Atomic publish: `File.renameTo` from staging to `<cacheDir>/<sanitizedRepoId>`. On any prior partial dir, delete first. After rename, the package is instantly visible to `LLMRepository.availableModels`.
9. `CacheIndex.list()` walks `cacheDir`, reads each `mobiletransformers_manifest.json` (tolerating legacy dirs without one — those still load via convention), returns id, base model, variants, installed features, byte size.

---

## Interactions

- **`LLMRepository` (unchanged):** consumes the materialized convention layout; `modelName` setter validates against `availableModels` (`LLMRepository.kt:83-101`).
- **#7 handoff map:** installer places `weight_handoff_map.json` in `inference/`; verifier checksums it; both engines read it from there.
- **#8 merger/external-data:** the per-tensor `*.bin` + `frozen_base.onnx.data` layout the installer lands is produced by #8 and described by #7.
- **#13 hub package format:** defines the HF-repo on-disk layout (`variants/`, `shared/`, `default/`) the manifest's `paths`/`downloadPlan` reference.
- **#14 export CLI / #20 hub pull:** the CLI emits `mobiletransformers_manifest.json` + `checksums.json`; the Android downloader fetches manifest-first, then the selected variant only, then calls this installer.
- **#16 facade:** `MobileTransformers.fromPretrained` orchestrates select→download→verify→install→hand to `LLMRepository`.

---

## Tests & smokes

**Python (`test_manifest.py`):**
- Round-trip manifest dataclass ↔ JSON; deterministic key order for checksumming.
- Validator rejects: missing `defaultVariant`, `defaultVariant` not in variants, unresolvable `weightHandoff.path`, variant claiming a feature with no `paths` entry, unknown `schemaVersion` major.

**Android (unit / Robolectric):**
- `ManifestValidator`: missing required file, version mismatch (`onnxRuntimeTrainingVersion`), bad schema.
- `VariantSelector`: ABI filter (no arm64 variant on an arm64-only device → `NoCompatibleVariant`); memory filter (variant `minMemoryMb` > device); feature filter (request `rag` but variant lacks it); engine filter (request `genai`, only `native` available); tie-break to `defaultVariant`.
- `ChecksumVerifier`: corrupt one staged byte → verify fails, staging deleted.
- `ModelPackageInstaller`: from a tiny fixture package, install and assert the four `updatePaths()` probe files exist at the conventional paths; assert `LLMRepository(cacheDir).availableModels` contains the `sanitizedRepoId` and `isGenerationAvailable`/`isTrainingAvailable`/`isRagAvailable` flip true.
- Atomicity: kill install mid-copy (throw before rename) → no partial dir under `cacheDir`, only `.staging` residue which a retry cleans.

**Smoke:** export a tiny model via #14, point a local fixture server (or `file://`) at it, run select→verify→install, then load through `LLMRepository` + run one generation step.

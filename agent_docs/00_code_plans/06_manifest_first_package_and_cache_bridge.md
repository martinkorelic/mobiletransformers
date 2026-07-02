# Manifest-First Package Model & Cache-Install Bridge

**Priority (global #):** 13  |  **Prerequisites:** `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` (#8), `01_code_plans/01_unified_merger_and_external_data_export.md` (#9)  |  **Blocks:** `02_code_plans/03_hub_model_package_format.md` (#14), `02_code_plans/05_one_command_export_cli.md` (#15), `00_code_plans/05_android_facade_foundation.md` (#17), `02_code_plans/04_hub_pull_and_cache_flow.md` (#21)

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

`<sanitizedRepoId>` is the repo id sanitized per the **single algorithm defined in `02_code_plans/03`** (`/` → `__` double underscore, other FS-unsafe chars stripped; e.g. `mobiletransformers/Qwen2-0.5B-mobile` → `mobiletransformers__Qwen2-0.5B-mobile`), so it is a single directory name under `cacheDir` — i.e. the `modelName` that `LLMRepository.availableModels` returns. The Kotlin sanitizer here must be byte-identical to the Python one (shared `sanitize_repo_id_cases.json` parity fixture).

---

## Touched / new files

**Android Kotlin (new), package `com.martinkorelic.ortmobile.packages`** (NOT `.package` — `package` is a Kotlin hard keyword and illegal as a package segment; `.packages` also matches the facade layout in `00_code_plans/05`):
- `MobileTransformersManifest.kt` — data class + Gson (de)serialization (project already uses Gson, see `ORTTrainerNative.kt:7`; Gson is the single JSON library for all manifest/config parsing on Android — do not introduce kotlinx.serialization for this).
- `ManifestValidator.kt` — schema + required-file presence + version-compat checks.
- `VariantSelector.kt` — capability/variant selection.
- `ChecksumVerifier.kt` — SHA-256 over `requiredFiles` vs `checksums.json`.
- `ModelPackageInstaller.kt` — atomic staging → cache-shape materialization.
- `CacheIndex.kt` — enumerate installed packages + manifest metadata + storage usage.

**Android Kotlin (light edit):**
- `LLMRepository.kt` — **no behavioral change required**. Optional: a secondary ctor / companion that, given a `sanitizedRepoId`, sets `modelName` (uses existing `modelName` setter validation, `LLMRepository.kt:83-92`). Do not move the filesystem probing.

**Python (export toolkit, new — shared schema source of truth):**
- `src/mobiletransformers/artifacts/manifest.py` — `MobileTransformersManifest` dataclass + emitter + validator (imports handoff dataclasses from #8's `handoff_map.py`).
- `tests/unit/test_manifest.py`.

---

## Data contract — `mobiletransformers_manifest.json`

> **The full field list is owned by `02_code_plans/03` (#14) — this plan does NOT redefine it.** #14 is the single definition site of the schema (camelCase wire names, `downloadPlan` keyed `[variantId][group]`, variant fields `supportedEngines` / `recommendedDeviceMemoryMb` / `paths`, integrity maps `sha256` / `fileSizes` / `etag`, provenance fields). This plan owns the **validator, variant-selection, and cache-install semantics** over that schema. The fields the components below consume:

| Field (per #14) | Consumed by | Notes |
| --- | --- | --- |
| `schemaVersion` / `minReaderVersion` | `ManifestValidator.check_compat` | `"MAJOR.MINOR"`; see the schema-versioning contract below. |
| `defaultVariant`, `variants[]` | `VariantSelector` | Variant fields: `id`, `abi`, `quantization`, `supportedEngines`, `features`, `recommendedDeviceMemoryMb`, `minimumAndroidApi`, `weightHandoff`, `paths`. |
| `variants[].paths` | `ModelPackageInstaller` | Map of `train`/`inference`/`embedding`/`tokenizer` → repo-relative subtree; the cache-bridge mapping source. |
| `downloadPlan[variantId][group]` | file-set computation (step 3) | Groups: `core`, `inference`, `train`, `rag`, `genai`, `checksums`. |
| `requiredFiles`, `sha256`, `fileSizes` | `ChecksumVerifier` | Per-file integrity; `sha256` mirrored into per-variant `checksums.json`. |
| `weightHandoff` (top-level + per-variant) | installer + engines | Pointer to `weight_handoff_map.json`; map schema **owned by #8**. Per-variant pointer wins over top-level. |
| `androidRuntime`, `onnxRuntimeTrainingVersion`, `onnxRuntimeGenAIVersion`, `supportedTasks`, `selectedTask`, `trustRemoteCode`, `license` | validator / `CacheIndex` metadata | Provenance + runtime requirements. |

`checksums.json`: `{ "<relativePath>": "<sha256hex>", ... }` over every `requiredFile` of the selected variant + core.

### Schema versioning (F1)

The manifest carries `schemaVersion` (`"MAJOR.MINOR"`) and `minReaderVersion`. Readers **tolerate unknown fields** (readers never rewrite the manifest — only the Python exporter writes it — so "preserve" means parse-and-ignore, not round-trip): an additive minor bump (new optional field) is non-breaking and parses fine on an older reader. A reader fails closed with a "needs newer SDK" message only when the manifest's `major` exceeds what it supports **or** its own version is below `minReaderVersion`. One `check_compat()` helper encodes this rule and is mirrored Python↔Kotlin (`ManifestValidator.check_compat` on the Android side, the same logic in `manifest.py`), so both ends agree on accept/reject — **implement exactly the algorithm pinned in `00_code_plans/07` ("check_compat — the ONE canonical algorithm")**, including the per-contract `READER_SCHEMA_VERSION` constant (here: `MANIFEST_READER_VERSION`) and the shared `check_compat_cases.json` test fixture. The handoff map (`#8`) and generated config schemas (`#6`) follow the same `{"schemaVersion":"1.2","minReaderVersion":"1.0", …}` field block.

---

## Implementation steps

### Selection & validation (no download yet)
1. `MobileTransformersManifest.parse(json)` (Gson) → typed object (unknown fields tolerated/ignored — Gson skips them by default); `ManifestValidator.validate` runs `check_compat` (`schemaVersion` major + `minReaderVersion`, F1), then `defaultVariant ∈ variants`, every `weightHandoff.path` resolvable, and that each variant's `paths` cover the features it claims.
2. `VariantSelector.select(manifest, deviceCaps, requestedFeatures, requestedEngine)`:
   - Filter variants whose `abi` intersects `Build.SUPPORTED_ABIS`.
   - Filter by `quantization` acceptable + `recommendedDeviceMemoryMb <= ActivityManager.MemoryInfo.totalMem` (the trainer already reads memory via `ActivityManager`, `ORTTrainerNative.kt:3/496`).
   - Filter by `features ⊇ requestedFeatures` and `supportedEngines ∋ requestedEngine` (default `native`, the guaranteed path).
   - Tie-break: smallest `recommendedDeviceMemoryMb`, then `defaultVariant`. Return a `SelectedVariant` or a typed `NoCompatibleVariant` error (fail closed).

### Download plan → staging
3. From `SelectedVariant`, compute the file set = `downloadPlan.groups.core` + groups for each requested feature + `checksums`. (The actual network fetch is #21; this plan defines the plan + the install bridge and works against an already-staged dir for testing.)
4. Download/copy into a staging dir `<cacheDir>/.staging/<sanitizedRepoId>/` (sibling of the final dir so rename is same-filesystem and atomic).

### Verify
5. `ChecksumVerifier.verify(stagingDir, manifest.sha256 ∩ selectedVariant files)` — SHA-256 each `requiredFile`; abort + delete staging on any mismatch. Verify the handoff map's own checksum (it gates correctness per #8).

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
- **#8 handoff map:** installer places `weight_handoff_map.json` in `inference/`; verifier checksums it; both engines read it from there.
- **#9 merger/external-data:** the per-tensor `*.bin` + `frozen_base.onnx.data` layout the installer lands is produced by #9 and described by #8.
- **#14 hub package format:** defines the HF-repo on-disk layout (`variants/`, `shared/`, `default/`) the manifest's `paths`/`downloadPlan` reference.
- **#15 export CLI / #21 hub pull:** the CLI emits `mobiletransformers_manifest.json` + `checksums.json`; the Android downloader fetches manifest-first, then the selected variant only, then calls this installer.
- **#17 facade:** `MobileTransformers.fromPretrained` orchestrates select→download→verify→install→hand to `LLMRepository`.

---

## Worked example

A minimal `mobiletransformers_manifest.json` (illustrative — one variant, one feature group; an instance of the #14 canonical schema):

```json
{
  "schemaVersion": "1.0",
  "minReaderVersion": "1.0",
  "baseModelId": "google/gemma-2-2b",
  "defaultVariant": "cpu-int4",
  "variants": [
    {
      "id": "cpu-int4",
      "abi": ["arm64-v8a"],
      "quantization": "int4",
      "recommendedDeviceMemoryMb": 2048,
      "features": ["inference"],
      "supportedEngines": ["native"],
      "paths": { "inference": "variants/cpu-int4/inference", "tokenizer": "shared/tokenizer" }
    }
  ],
  "downloadPlan": {
    "cpu-int4": {
      "inference": [
        "variants/cpu-int4/inference/model.onnx",
        "variants/cpu-int4/inference/generation_config.json",
        "variants/cpu-int4/inference/frozen_base.onnx.data",
        "variants/cpu-int4/inference/weight_handoff_map.json"
      ]
    }
  },
  "sha256": {
    "variants/cpu-int4/inference/model.onnx": "9f2b…c1",
    "variants/cpu-int4/inference/weight_handoff_map.json": "4ad0…7e"
  },
  "weightHandoff": {
    "path": "variants/cpu-int4/inference/weight_handoff_map.json",
    "handoffMode": "external_initializer"
  }
}
```

After install, `variants/cpu-int4/inference/**` lands at `<cacheDir>/<sanitizedRepoId>/inference/**` and `shared/tokenizer/**` at `…/tokenizer/**`, so `LLMRepository.updatePaths()` discovers the model with no change.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- Python `test_manifest.py`: round-trip manifest dataclass ↔ JSON with deterministic key order for checksumming; validator rejects missing `defaultVariant`, `defaultVariant` not in variants, unresolvable `weightHandoff.path`, a variant claiming a feature with no `paths` entry, and a `schemaVersion` major beyond support / `minReaderVersion` unmet (`check_compat`, F1).
- Android JVM/Robolectric `ManifestValidator`: missing required file, version mismatch (`onnxRuntimeTrainingVersion`), bad schema.
- Android `VariantSelector`: ABI filter (no arm64 variant on an arm64-only device → `NoCompatibleVariant`); memory filter (`recommendedDeviceMemoryMb` > device); feature filter (request `rag` but variant lacks it); engine filter (request `genai`, only `native` available); tie-break to `defaultVariant`.
- Android `ChecksumVerifier`: corrupt one staged byte → verify fails, staging deleted.
- Android atomicity: kill install mid-copy (throw before rename) → no partial dir under `cacheDir`, only `.staging` residue which a retry cleans.
- Module compiles: `./gradlew :MobileTransformers:compileDebugKotlin`.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `ModelPackageInstaller` (Robolectric, no device): from a tiny fixture package, install and assert the four `updatePaths()` probe files exist at the conventional paths; assert `LLMRepository(cacheDir).availableModels` contains the `sanitizedRepoId` and `isGenerationAvailable`/`isTrainingAvailable`/`isRagAvailable` flip true for the features the variant claims.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Export a tiny model via #15, point a local fixture server (or `file://`) at it, run select→verify→install, then load through `LLMRepository` + run one generation step on a device/emulator.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- `mobiletransformers_manifest.json` parses to a typed object on both sides (Python dataclass + Kotlin Gson), preserves unknown fields, and passes `check_compat` (F1).
- `VariantSelector` returns a `SelectedVariant` or a fail-closed `NoCompatibleVariant`; `ChecksumVerifier` aborts + cleans staging on any mismatch.
- `ModelPackageInstaller` materializes a package into the exact `<cacheDir>/<sanitizedRepoId>/{train,inference,embedding,tokenizer}` shape via atomic rename, so `LLMRepository` discovers it with **zero** behavioral change.
- `CacheIndex.list()` enumerates installed packages (tolerating legacy dirs without a manifest).

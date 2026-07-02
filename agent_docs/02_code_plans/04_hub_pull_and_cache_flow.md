# Hub Pull and Cache Flow

**Priority #21 | Prerequisites: #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`), #14 (`02_code_plans/03_hub_model_package_format.md`) | Blocks: sample-app facade load (#19), starter-zoo loading**

## Purpose

Get a MobileTransformers-ready Hub package onto a device's cache in the exact layout `LLMRepository` expects, by downloading **only the files the user actually needs**, validated by sha256, with crash-safe staging.

Two surfaces, shipped in order:

1. **Python `mobiletransformers pull` / `install-package` (ship first).** Uses `huggingface_hub.snapshot_download(..., allow_patterns=...)` with patterns derived from the manifest `downloadPlan`, then materializes the selected variant into the Android cache layout (push to emulator/device or a local dir). This is the robust v1 path — `huggingface_hub` already handles revisions, ETags, Xet, retries, and auth.
2. **Android manifest-first downloader (next milestone, deferral allowed).** A small OkHttp + WorkManager resolver that fetches the manifest first, selects a variant from device constraints, downloads grouped files into `.partial`, sha256-validates, atomically renames, and registers with `LLMRepository`. **NOT** an embedding of the Python SDK.

If the Android downloader slips, v1 ships with Python pull preparing the cache; direct device download is the follow-up.

## Touched / new files

Python (pkg root scaffolded by #1):

- `mobiletransformers/hub/pull.py` — `pull_package()` + `install_package()`; the `pull` / `install-package` CLI subcommands (registered alongside `export`/`push` from #15).
- `mobiletransformers/hub/variant_select.py` — pure-function `select_variant(manifest, constraints) -> variant_id`; shared logic with Android (algorithm documented, reimplemented in Kotlin).
- Reuses `mobiletransformers/hub/package_format.py` (#14): `FEATURE_GROUPS`, `sanitize_repo_id`, `build_manifest` round-trip helpers.

Android (`com.martinkorelic.mobiletransformers` SDK module, post-rename per `00_repository_restructure_plan.md`):

- `hub/HubResolver.kt` — builds Hub resolve URLs (`https://huggingface.co/<repoId>/resolve/<revision>/<path>`), adds `Authorization: Bearer <token>` when present.
- `hub/ManifestClient.kt` — fetches + parses `mobiletransformers_manifest.json` first.
- `hub/VariantSelector.kt` — Kotlin port of `select_variant`.
- `hub/PackageDownloadWorker.kt` — `CoroutineWorker` (WorkManager) doing the grouped download into `.partial` with OkHttp, sha256, atomic rename.
- `hub/PackageInstaller.kt` — registers the materialized package and refreshes `LLMRepository.availableModels`.
- Reuses #13's cache-bridge mapper (variant subtree → cache layout).

## Data contracts / interfaces

### Cache layout (target, owned by #13; restated)

```text
<cacheDir>/<sanitizedRepoId>/
  train/         training_model.onnx, eval_model.onnx, optimizer_model.onnx, checkpoint/,
                 training_config.json, trainable_parameters.json, weight_handoff_map.json,
                 merger_*.onnx (descriptive registry filenames from 09), <task>.jsonl
  inference/     model.onnx, frozen_base.onnx.data (immutable), per-tensor <name>.bin (+ .sha256) FLAT,
                 weight_handoff_map.json, generation_config.json, genai_config.json?, session_options.json
  embedding/     embedding_model.onnx, tokenizer/, rag_config.json          (only if rag pulled)
  tokenizer/     tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json, ortmobile_tokenizer_config.json, chat_template.jinja
  mobiletransformers_manifest.json
  checksums.json
```

This is exactly what `LLMRepository.updatePaths()` and `ORTTrainerNative` probe (`<cacheDir>/<repoName>/train/...`, `inference/...`, `embedding/rag_config.json`, `tokenizer`). `<sanitizedRepoId>` becomes `LLMRepository.modelName` / `ORTTrainingConfig.repoName`.

> Mapping note: the Hub repo splits tokenizer under `shared/tokenizer/` and chat template under `shared/chat_template.jinja`; the cache bridge flattens these into the single `<repo>/tokenizer/` dir the repository expects. `inference/` is copied through as-is — it is already the flat canonical layout (#9/#13); there are **no** `base/` or `merged/` subdirs.

### Variant selection (`select_variant`)

Inputs (`constraints`):
- `abi` (Android `Build.SUPPORTED_ABIS`, e.g. `arm64-v8a`)
- `preferredQuantization` (default `int4`)
- `engine` (`native` default | `genai`)
- `requestedFeatures` (subset of `core`/`inference`/`train`/`rag`/`genai`; always includes `core`+`inference`)
- `availableStorageBytes`
- `deviceMemoryMb` (`ActivityManager.MemoryInfo.totalMem`)

Algorithm (deterministic; identical in Python and Kotlin):
1. Filter variants whose `abi` is null or intersects device `abi`.
2. Filter variants whose `supportedEngines` contains `engine`.
3. Filter variants whose `features` ⊇ `requestedFeatures`.
4. Filter variants whose `recommendedDeviceMemoryMb` ≤ `deviceMemoryMb` (soft: keep best-effort fallback if none qualify, surface a warning).
5. Among survivors, prefer `quantization == preferredQuantization`; tie-break by smaller estimated download size (sum of `fileSizes` over the requested groups).
6. If none survive hard filters → fall back to `manifest.defaultVariant` and emit a constraint-mismatch warning.
7. Estimated download size must be ≤ `availableStorageBytes × 0.9` or fail with an actionable error.

### Download file list

```
files = []
for group in requestedFeatures + ["core", "checksums"]:
    files += manifest.downloadPlan[variantId][group]   # repo-relative globs
expand globs against manifest.requiredFiles/fileSizes keys → concrete paths
```

Python passes the glob list straight to `snapshot_download(allow_patterns=...)`. Android expands globs against `manifest.fileSizes` keys (the manifest enumerates every file, so no remote listing is needed).

## Implementation steps

### A. Python pull (ship first)

1. `pull_package(repo_id, *, revision="main", variant=None, features=("inference",), token=None, dest=None)`:
   - `snapshot_download(repo_id, revision=revision, allow_patterns=["mobiletransformers_manifest.json"], token=token)` → read manifest.
   - `variant = variant or select_variant(manifest, default_desktop_constraints())` (desktop constraints: abi=any, engine=native, memory=large).
   - Build `allow_patterns` from `downloadPlan[variant]` over `features ∪ {core, checksums, genai?}`.
   - `snapshot_download(repo_id, revision=revision, allow_patterns=patterns, token=token, local_dir=<staging>)`.
   - Verify sha256 of every downloaded file against `manifest.sha256`; fail loudly on mismatch.
2. `install_package(staging, cache_root, repo_id)`:
   - Call #13's cache-bridge mapper to transform `variants/<variant>/...` + `shared/...` → `<cache_root>/<sanitize_repo_id(repo_id)>/{train,inference,embedding,tokenizer}` + copy `mobiletransformers_manifest.json` + `checksums.json`.
   - Atomic: build under `<cache_root>/.partial/<sanitized>` then `os.replace` onto the final dir.
3. CLI wiring: `mobiletransformers pull --repo-id ... [--revision] [--variant] [--features inference,train,rag] [--out <dir>]` and `mobiletransformers install-package --repo-id ... --cache-root <android-cache-dir>`. `--out` defaults to HF cache; `install-package` additionally runs the bridge. Support pushing to a device dir or `adb push` target documented in help text (no adb dependency in code).

### B. Android downloader (next milestone)

4. `ManifestClient.fetchManifest(repoId, revision, token)` → GET resolve URL for `mobiletransformers_manifest.json`, parse with #13's Gson-based `MobileTransformersManifest` (one JSON library on Android — do not add kotlinx.serialization).
5. `VariantSelector.select(manifest, DeviceConstraints)` → Kotlin port of `select_variant`; `DeviceConstraints` filled from `Build.SUPPORTED_ABIS`, `ActivityManager`, `StatFs` on cacheDir, plus caller's `features`/`engine`.
6. `PackageDownloadWorker` (WorkManager `CoroutineWorker`, foreground for large pulls, `Constraints` = unmetered + storage-not-low):
   - For each file in the computed list: HEAD (size/etag) → cross-check against `manifest.fileSizes`/`etag` → GET via OkHttp streaming into `<cacheDir>/.partial/<sanitizedRepoId>/<repoRelativePath>`.
   - Support HTTP Range resume when the CDN returns `Accept-Ranges: bytes`; resume from existing `.partial` length.
   - Stream sha256 while writing; compare to `manifest.sha256[path]`. On mismatch delete the file and retry up to N times, then fail the work.
   - Emit `setProgress` per-file and aggregate bytes for UI.
7. On all-files-complete: run the Kotlin cache-bridge mapper (#13) inside the worker to reshape `.partial` into the final cache layout, then atomic `File.renameTo` (same filesystem) of `.partial/<sanitizedRepoId>` → `<cacheDir>/<sanitizedRepoId>`. Write `mobiletransformers_manifest.json` + `checksums.json` last.
8. `PackageInstaller.register(repoId)` → set `LLMRepository.modelName = sanitizedRepoId` (it appears in `availableModels` because it's now a dir under `cacheDir`); `updatePaths()` then loads configs through existing parsers.
9. Cancellation/failure: never touch the live `<cacheDir>/<sanitizedRepoId>`; only `.partial` is mutated until the final rename. On cancel, leave `.partial` for resume or GC it on next run.
10. Surface as `MobileTransformers.fromPretrained(context, repoId, revision, variant?, features, cacheDir)` (facade in #19) → enqueue worker → await → return `MobileTransformerModel`.

## Interactions

- **#14 (package format):** source of `downloadPlan`, `sha256`, `fileSizes`, `etag`, `variants`, `defaultVariant`; `sanitize_repo_id` parity.
- **#13 (cache bridge):** owns the variant-subtree → cache-layout mapper used by both `install_package` (Python) and `PackageDownloadWorker` (Kotlin), plus the manifest validator run before exposing the package.
- **#15 (export CLI):** produces the package this flow consumes; the tiny fixture from #14 backs the smokes here.
- **#19 (Kotlin facade):** `fromPretrained` is the public entry that drives the Android downloader.
- **`LLMRepository`:** unchanged — it just discovers the new dir under `cacheDir` and loads `train/training_config.json`, `inference/generation_config.json`, `embedding/rag_config.json`, `tokenizer/`.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `select_variant` unit table (`pytest tests/hub/test_variant_select.py`): device-constraint rows → expected variant id, incl. fallback-to-default and storage-exceeded failure.
- `sanitize_repo_id` Kotlin↔Python parity via the shared `sanitize_repo_id_cases.json` (from #14): a JVM `VariantSelectorTest`/parity test asserts the Kotlin sanitizer matches the Python table; the Android module **compiles** (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- Pull smoke against the tiny fixture served from a local dir / `huggingface_hub` offline cache: `pull_package(features=("inference",))` downloads only core+inference+checksums, sha256 all pass, `train/` and `embedding/` absent.
- `install_package` smoke: materializes the fixture into a temp `cache_root` with the exact `LLMRepository` layout; assert `train/training_config.json`, `inference/generation_config.json`, `tokenizer/tokenizer.json` exist and tokenizer/chat-template flattening happened.
- sha256-mismatch path: corrupt one fixture byte → pull fails with the offending path named.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Android instrumentation (MockWebServer serving the tiny fixture + manifest):
  - Manifest-first: only the manifest is fetched before variant selection; no large GETs precede it.
  - Grouped download: requesting `inference` only fetches core+inference+checksums files; `train` group untouched.
  - Checksum validation: a tampered served file → file deleted, retried, work fails cleanly; live cache dir never created.
  - Partial/resume: kill mid-download → `.partial` retained → re-run resumes via Range and completes.
  - Atomic rename: assert `<cacheDir>/<sanitizedRepoId>` appears only after full validation; interrupted run leaves no half-populated final dir.
  - Registration: post-install `LLMRepository(cacheDir, sanitizedRepoId)` reports `isGenerationAvailable == true` and the facade can load.

**Workflow (end-to-end)** — *(CHECKPOINT #21)* Python `pull_package(repo_id, features=("inference",))` → `install_package(staging, cache_root, repo_id)` materializes the variant into `<cache_root>/<sanitizedRepoId>/` (the exact `LLMRepository` layout) → the SDK loads that cache. The Python pull→install→sha256-verify legs are **automated** over the tiny fixture; the Android-load leg (`fromPretrained` over the materialized cache reporting `isGenerationAvailable`) is **Manual** (device/emulator).

**Definition of done** — `mobiletransformers pull` + `install-package` download only the files for the requested features (verified via `downloadPlan`), sha256-validate every file (failing loudly with the offending path named), and materialize a crash-safe (`.partial` → atomic replace) cache in the exact layout `LLMRepository.updatePaths()` probes; `select_variant` is deterministic and shared Python↔Kotlin (parity test green); and a materialized fixture loads through the facade on a device.

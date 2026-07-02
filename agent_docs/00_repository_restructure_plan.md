# Repository Restructure Plan

## Purpose

This plan defines the foundational repository shape for the next release. The goal is to turn the current research-oriented tree into a packageable Python export toolkit plus a first-class Android SDK/app workspace, without disrupting the working training, merge, inference, and RAG code paths during the first migration pass.

> Detailed per-feature code-implementation plans live in `agent_docs/00_code_plans/`, `01_code_plans/`, and `02_code_plans/`; the global implementation order lives in `agent_docs/IMPLEMENTATION_ORDER.md`.

## Current Repo Evidence

- Python code is currently split across root-level packages: `trainer/`, `artifact/`, `inference/`, `tools/`, `peft_models/`, `database/`, `evaluation/`, and `research/`.
- Runtime/export configuration lives at the repo root as `config.yml`, while secrets and experiment constants live in `config.py`.
- Several modules read environment variables directly, especially `HF_TOKEN`, `HF_CACHE`, Azure OpenAI variables, and `GEMINI_API_KEY`.
- There is no `pyproject.toml`, `setup.py`, `Makefile`, CI directory, changelog, or semantic version tag.
- Android is already a Gradle workspace under `android/ORTransformer/`, with `ORTransformersMobile` as the reusable library module and `app` as a sample UI.
- The current Android Gradle root is `ORTTransformer`, with modules `:ORTransformersMobile` and `:app`. The library namespace is `com.martinkorelic.ortmobile`; the app namespace is `com.martinkorelic.orttransformer`; the app `applicationId` is currently `com.martinkorelic.ortmobile`.
- The Android library currently pulls a local `onnxruntime-genai.aar`, links native C++ through CMake, and contains the active Kotlin/C++ training, merging, inference, tokenizer, and RAG code.
- Python dependencies are currently split into root `requirements-or.txt` and `requirements-ort.txt`, with different Optimum pins, GPU/ROCm packages, and a local/source-built `onnxruntime-training==1.23.0+cpu`.
- `requirements-ort.txt` combines training, evaluation, RAG, docs, and export dependencies in one file; `requirements-or.txt` combines notebook/visualization, ROCm, CUDA-related packages, and ORT/Torch packages.
- `agent_docs/` existed before this pass and was empty.

## External Research Summary

- PyPA recommends a `pyproject.toml` with `[build-system]` and `[project]` metadata for modern Python packaging, with extras and CLI scripts declared in the same file. Source: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- PyPA describes the `src/` layout as a way to avoid accidentally importing code directly from the project root during development. Source: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- `uv` supports standard `[project.dependencies]`, `[project.optional-dependencies]`, local `dependency-groups`, and `tool.uv.sources` for alternate development sources. Source: https://docs.astral.sh/uv/concepts/projects/dependencies/
- `uv` can import dependencies from existing requirements files, sync locked environments, and export lockfiles back to `requirements.txt`, `pylock.toml`, or CycloneDX SBOM formats. Sources: https://docs.astral.sh/uv/concepts/projects/dependencies/, https://docs.astral.sh/uv/concepts/projects/sync/, https://docs.astral.sh/uv/concepts/projects/export/
- `uv` path sources can point at local wheels, source distributions, or project directories, which matches this repo's local/source-built ONNX Runtime Training wheel problem. Source: https://docs.astral.sh/uv/concepts/projects/dependencies/
- Poetry supports project dependency specification and git/path-style dependencies, and remains a reasonable packaging option, but it is less convenient here because the repo needs pip-compatible exported lockfiles and multiple platform-specific ML profiles for Android/export workflows. Source: https://python-poetry.org/docs/dependency-specification/
- `optimum-onnx` is now the split-out ONNX export package and should be the primary export extra, with version `>=0.1.0` in the next dependency sketch. Source: https://pypi.org/project/optimum-onnx/
- Pydantic Settings is a mature option for env-driven typed settings, but the dependency should be deliberate because this repo already has a heavy export stack. Source: https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/

## Recommended Decision

Adopt a staged `src/mobiletransformers/` Python package and rename Android into a branded `android/MobileTransformers/` SDK/app workspace. Do not move all files at once. First add package/config/build scaffolding and compatibility wrappers, then migrate implementation modules subsystem by subsystem.

The first pass should prioritize import stability and reproducibility over a sweeping rename. Existing scripts should keep working while new CLI entry points call into the package.

Use `uv` as the primary Python project/dependency manager for this repository. Keep the `pyproject.toml` metadata standards-compliant so the package can still be built by normal Python tooling, but let `uv` own local environment sync, lock resolution, local wheel sources, dependency groups, and exported requirements files. Poetry is not the best fit for this repo's first packaging pass because MobileTransformers needs several incompatible dependency profiles, pip-compatible outputs for scripts/CI, and local source-built ORT artifacts.

Use `MobileTransformers` as the Android SDK/library module name and `MobileTransformersApp` as the sample app module name. Treat `ORTransformersMobile`, `ORTTransformer`, `ortmobile`, and `orttransformer` as legacy names that should be preserved only through compatibility aliases or wrappers during migration.

For ordering, treat `05_cross_cutting_release_modernization.md` as the master release spine. This restructure plan owns the target shape, but implementation should still follow the global sequence: package/config/dependency foundation, ORT/export provenance, manifest/cache contract, Android rename/facade, then feature work.

## Target Hierarchy

```text
mobiletransformers/
  pyproject.toml
  uv.lock
  README.md
  LICENSE.md
  CHANGELOG.md
  Makefile
  config/
    config.yml        # user-editable YAML only — NO Python in repo-root config/ (import-collision + wheel rules; see 00_code_plans/02)
    logging.yml
  src/mobiletransformers/
    __init__.py
    config/
      __init__.py
      settings.py     # secrets/env (ships in the wheel)
      constants.py    # shared constants + enums (00_code_plans/09)
      models.py       # Pydantic v2 typed config (00_code_plans/09)
      registry/       # PEFT / architecture / merger registries (00_code_plans/09)
    cli/
      __init__.py
      main.py
      export.py
      validate.py
      package_model.py
    export/
      __init__.py
      training_export.py
      inference_export.py
      tokenizer_export.py
      quantization.py
    artifacts/
      __init__.py
      builder.py
      merger_models.py
      manifest.py
      validation.py
    peft/
      __init__.py
      mars/
      lora_xs/
      ablation/
    training/
      __init__.py
      builders.py
      validators.py
      schedulers.py
    inference/
      __init__.py
      builders.py
      generator.py
      validation.py
    rag/
      __init__.py
      embeddings.py
      vector_store.py
      objectbox.py
      ingestion.py
    hub/
      __init__.py
      download.py
      upload.py
      model_card.py
    evaluation/
      __init__.py
      benchmarks/
      mobile/
      smoke/
    utils/
      __init__.py
      paths.py
      logging.py
      yaml.py
  tests/
    unit/
    integration/
    smoke/
  requirements/
    requirements-core.lock.txt
    requirements-export.lock.txt
    requirements-train-local.lock.txt
    requirements-rag.lock.txt
    sbom-cyclonedx.json
  scripts/
    export_model.sh
    build_ort_training_wheel.sh
    build_ort_training_android.sh
    android_build_aar.sh
    publish_local_maven.sh
  third_party/
    onnxruntime/
      BUILD.md
      manifest.json
    wheels/
      README.md
  android/MobileTransformers/
    settings.gradle.kts
    MobileTransformers/
      src/main/java/com/martinkorelic/mobiletransformers/
        MobileTransformers.kt
        MobileTransformerModel.kt
        config/
        hub/
        packages/
        runtime/
        training/
        inference/
        rag/
        extensions/
        internal/
      src/main/cpp/
    MobileTransformersApp/
      src/main/java/com/martinkorelic/mobiletransformers/app/
  docs/
  agent_docs/
  research/
```

## Migration Map

| Current path | Target package path | Notes |
| --- | --- | --- |
| `trainer/builder.py` | `src/mobiletransformers/export/training_export.py` | Keep a root `trainer/builder.py` shim initially. |
| `trainer/validator.py`, `trainer/merge_validator.py` | `src/mobiletransformers/training/validators.py` and `src/mobiletransformers/artifacts/validation.py` | Split model-training checks from artifact/merge checks. |
| `artifact/onnx_builder.py` | `src/mobiletransformers/artifacts/builder.py` | Owns model package assembly and JSON config emission. |
| `artifact/merger.py` | `src/mobiletransformers/artifacts/merger_models.py` | Owns generated ONNX merger graphs. |
| `inference/builder.py` | `src/mobiletransformers/export/inference_export.py` | Large file; migrate behind one stable CLI command first. |
| `inference/generator.py` | `src/mobiletransformers/inference/generator.py` | Python-side validation/generation helper. |
| `tools/tokenizer_export.py` | `src/mobiletransformers/export/tokenizer_export.py` | Shared by package builder and Hub package export. |
| `tools/parser_config.py` | `src/mobiletransformers/config/constants.py` (+ `src/mobiletransformers/utils/yaml.py` for the loader) | Constants become typed constants inside the package; parsing helper moves to utils. |
| `peft_models/mars/` | `src/mobiletransformers/peft/mars/` | Public differentiator; keep names stable. |
| `peft_models/lora_xs/` | `src/mobiletransformers/peft/lora_xs/` | Preserve import compatibility until tests move. |
| `database/` | `src/mobiletransformers/rag/` | Python-side RAG/database helpers; Android ObjectBox remains in Android module. |
| `evaluation/` | `src/mobiletransformers/evaluation/` plus `tests/` | Move reusable evaluators to package; actual tests to `tests/`. |
| `research/` | `research/` | Leave as non-package research scripts for v1. |

## Android Migration Map

| Current Android path/name | Target path/name | Notes |
| --- | --- | --- |
| `android/ORTransformer/` | `android/MobileTransformers/` | Rename Gradle workspace root and wrapper location as one isolated Android change. |
| Gradle root `ORTTransformer` | Gradle root `MobileTransformers` | Update `settings.gradle.kts`. |
| Module `:ORTransformersMobile` | Module `:MobileTransformers` | This is the Android SDK/AAR module. |
| Directory `android/ORTransformer/ORTransformersMobile/` | `android/MobileTransformers/MobileTransformers/` | Keep C++ layout under `src/main/cpp`. |
| Module `:app` | Module `:MobileTransformersApp` | This is the sample/dev app, not the SDK artifact. |
| Directory `android/ORTransformer/app/` | `android/MobileTransformers/MobileTransformersApp/` | Move sample app code after SDK module builds under the new name. |
| Library namespace `com.martinkorelic.ortmobile` | `com.martinkorelic.mobiletransformers` | Add temporary deprecated wrappers for old imports if practical. |
| App namespace `com.martinkorelic.orttransformer` | `com.martinkorelic.mobiletransformers.app` | Align source directories with package names. |
| App `applicationId = "com.martinkorelic.ortmobile"` | `com.martinkorelic.mobiletransformers.app` | Accept app reinstall/data reset because this is a sample app. |
| `implementation(project(":ORTransformersMobile"))` | `implementation(project(":MobileTransformers"))` | Update sample app dependency after module rename. |
| AAR artifact name | `mobiletransformers-android` | Use this as the first-release default; keep Maven coordinates separate from Gradle module name. |

## Config Strategy

Use three explicit configuration layers:

- `config/config.yml`: user-editable defaults for export, training, inference, artifact packaging, RAG, and smoke tests. This replaces root `config.yml` after compatibility shims are added. Repo-root `config/` holds **YAML only**.
- `src/mobiletransformers/config/settings.py`: typed runtime settings that read environment variables. This owns secrets and machine-specific values: `HF_TOKEN`, `HF_CACHE`, Azure OpenAI values, `GEMINI_API_KEY`, build cache directories, and optional Android SDK/NDK overrides. Lives **inside the package** so installed wheels are self-contained (and so the root `config.py` shim can import it without colliding on the `config` module name — see `00_code_plans/02`).
- `src/mobiletransformers/config/constants.py`: non-secret constants shared across the repo, such as config section names, default artifact filenames, supported PEFT method names, supported embedding dimensions, and canonical package manifest filenames.

Recommended environment precedence:

1. CLI flag.
2. Environment variable via `settings.py`.
3. `config/config.yml`.
4. Package default from `constants.py`.

Recommended concrete settings model:

- Use `pydantic-settings` if it is already accepted as a dependency in the packaging spike.
- Otherwise use a small standard-library dataclass loader with `os.environ`, `pathlib.Path`, and explicit validation.
- Never read secrets directly in export/training modules after this migration. Pass a `Settings` object or call a central `get_settings()`.

## Dependency And Packaging Strategy

### Decision: Use `uv` Over Poetry

Use `uv` for the next release. Keep Poetry out of the critical path unless a future contributor strongly prefers it for local development.

Reasons:

- The repo has at least four incompatible install surfaces: lightweight package/runtime, export/training artifact generation, local/source-built ORT training, and RAG/evaluation.
- `onnxruntime-training==1.23.0+cpu` should be treated as a source-built local wheel rather than a normal public PyPI dependency. `uv` can declare this through `tool.uv.sources` pointing at a local wheel or private index while keeping the published dependency metadata clean.
- Existing requirements files can be imported during migration, and exported lockfiles can keep shell scripts, Dockerfiles, and CI jobs pip-compatible.
- Dependency groups are a clean place for local-only profiles such as `dev`, `ort-training-local`, `android-build`, and `rocm-export`, while optional dependencies remain the public install surface.
- A single `uv.lock` plus exported requirements gives both reproducible local development and compatibility with tools that do not understand `uv`.

Poetry would still work for a conventional Python app/library. It is not the preferred tool here because its lock workflow is less convenient for generating multiple pip-compatible outputs and because this repo needs local wheel sources and platform-specific ML dependency profiles as first-class concepts.

### `pyproject.toml` Shape

Create `pyproject.toml` with:

- `[build-system]` using `hatchling` or `setuptools`; choose `hatchling` unless compiled Python extensions are added.
- `[project]` with `name = "mobiletransformers"`, `readme = "README.md"`, `requires-python`, authors, URLs, and license expression after the licensing decision.
- `[project.scripts]` with `mobiletransformers = "mobiletransformers.cli.main:main"`.
- `[project.optional-dependencies]` extras for public install surfaces: `export`, `train`, `rag`, `eval`, and `hub`.
- `[dependency-groups]` for local-only workflows: `dev`, `docs`, `smoke`, `ort-training-local`, `android-build`, `export-cpu`, and `export-rocm`.
- `[tool.uv.sources]` for local/source-built wheels, PyTorch indexes, or private artifact indexes.

Recommended starting sketch:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "mobiletransformers"
version = "0.1.0"
description = "Export and Android runtime tooling for on-device transformer training and inference."
readme = "README.md"
requires-python = ">=3.10,<3.14"  # optimum-onnx 0.1.0 supports Python 3.13
dependencies = [
  "huggingface-hub>=0.34",
  "numpy>=1.26",
  "onnx>=1.16",
  "pyyaml>=6.0",
  "tokenizers>=0.20",
  # optimum-onnx 0.1.0 ceilings transformers; keep this in sync with the export extra.
  "transformers>=4.45,<4.58",
]

[project.scripts]
mobiletransformers = "mobiletransformers.cli.main:main"

[project.optional-dependencies]
export = [
  # Pin optimum-onnx exactly; it transitively pins optimum (~=2.1). Do NOT install its
  # [onnxruntime] extra when co-located with the source-built training wheel (see note below).
  "optimum-onnx==0.1.0",
  # torch is currently UNPINNED anywhere in the repo; it must be explicit, pinned, and CI-verified.
  "torch==2.5.1",
]
genai = [
  # Currently absent from requirements files; only CMake-linked on Android today.
  "onnxruntime-genai>=0.14",
]
train = [
  "onnxscript>=0.3",
  "peft>=0.13",
  # torch is currently UNPINNED anywhere in the repo; it must be explicit, pinned, and CI-verified.
  "torch==2.5.1",
]
rag = [
  "langchain-community>=0.3",
  "langchain-huggingface>=0.3",
  "langchain-objectbox>=0.1",
  "sentence-transformers>=5",
]
eval = [
  "deepeval>=3",
  "matplotlib>=3.10",
]

[dependency-groups]
dev = [
  "pytest>=8",
  "ruff>=0.6",
  "mypy>=1.11",
]
docs = ["mkdocs>=1.6"]
ort-training-local = ["onnxruntime-training==1.23.0+cpu"]
export-rocm = ["onnxruntime-rocm==1.18.0"]

[tool.uv.sources]
# Generated by scripts/build_ort_training_wheel.sh; wheel path is ignored by git.
onnxruntime-training = { path = "third_party/wheels/onnxruntime_training-1.23.0+cpu-local.whl" }
```

> **Profile isolation correctness (ONNX Runtime import collision).** The public packages
> `onnxruntime`, `onnxruntime-gpu`, `onnxruntime-genai`, and `onnxruntime-training` all collide on
> the `onnxruntime` import namespace and must be isolated per profile — never resolved into the same
> environment. In particular, the `export` extra must install `optimum-onnx` **without** its
> `[onnxruntime]` extra whenever it is co-located with the source-built training wheel, because the
> `ort-training-local` wheel already **provides** the `onnxruntime` import. Installing
> `optimum-onnx[onnxruntime]` alongside the training wheel would pull a second, conflicting
> `onnxruntime` distribution over the same import path. Keep `optimum` itself transitively pinned
> through `optimum-onnx==0.1.0` (~=2.1) rather than pinning it directly.

Keep lockfiles separate from loose package metadata:

- `uv.lock`
- `requirements/requirements-export.lock.txt`
- `requirements/requirements-train-local.lock.txt`
- `requirements/requirements-rag.lock.txt`
- `requirements/requirements-dev.lock.txt`
- `requirements/sbom-cyclonedx.json`

Generate the requirements files from `uv.lock` rather than hand-editing them:

```bash
uv sync --group dev --extra export
uv sync --group ort-training-local --extra train
uv export --format requirements.txt --extra export --output-file requirements/requirements-export.lock.txt
uv export --format requirements.txt --group ort-training-local --extra train --output-file requirements/requirements-train-local.lock.txt
uv export --format cyclonedx1.5 --output-file requirements/sbom-cyclonedx.json
```

During migration, import the existing files only as seed data:

```bash
uv add -r requirements-ort.txt --group ort-training-local
uv add -r requirements-or.txt --group export-rocm
```

Then split the resulting dependency set by purpose. Do not preserve the current mixed requirements shape as a long-term contract.

### Dependency Profile Boundaries

Recommended profiles:

| Profile | Purpose | Notes |
| --- | --- | --- |
| Core project dependencies | Import package, parse config, download/upload Hub packages, lightweight validation | Must stay small and platform-neutral. |
| `export` extra | Build ONNX/GenAI/mobile packages on a desktop machine | Contains Optimum ONNX or direct ONNX export dependencies. |
| `train` extra | Python-side training artifact generation and PEFT export helpers | Does not imply local ORT training wheel by itself. |
| `ort-training-local` group | Source-built ONNX Runtime Training wheel used for artifact generation compatibility | Local-only; never assume public PyPI has the exact wheel. |
| `export-rocm` group | Existing ROCm/Torch-ORT experiment surface | Keep out of default sync. |
| `rag` extra | Embeddings, ObjectBox/LangChain helpers, ingestion | Useful but not required for export. |
| `eval` extra | Benchmarks and quality checks | Keep out of Android/export minimal paths. |
| `android-build` group | Python tools needed by AAR packaging scripts | Should not pull training/eval stacks unless required. |

### Source-Built ORT Training Artifact Strategy

Treat ONNX Runtime Training as a local build product with metadata:

- Add `scripts/build_ort_training_wheel.sh` for the Python wheel with `--enable_training_apis` and `--build_wheel`.
- Add `scripts/build_ort_training_android.sh` for the Android AAR/header/native libraries with matching ORT commit, NDK, Android API level, ABIs, and build type.
- Record the exact ORT git SHA/tag, build flags, Python version, CMake version, Android NDK version, ABI list, and SHA256 checksums in `third_party/onnxruntime/manifest.json`.
- Document human-readable build steps in `third_party/onnxruntime/BUILD.md`.
- Keep actual `.whl`, `.aar`, `.so`, and extracted build outputs ignored by git unless the project intentionally creates a release artifact repository.
- Use `tool.uv.sources` path entries for local development and a private package/index source if the team later publishes internal wheels.

This avoids pretending that the public `onnxruntime-training` package can satisfy the repo's current `1.23.0+cpu` requirement while still giving contributors a reproducible build story.

## Android Structure

Rename the Android workspace as part of the foundation restructuring:

- Gradle root: `android/MobileTransformers`, `rootProject.name = "MobileTransformers"`.
- SDK module: `:MobileTransformers`, directory `android/MobileTransformers/MobileTransformers`.
- Sample app module: `:MobileTransformersApp`, directory `android/MobileTransformers/MobileTransformersApp`.
- SDK namespace: `com.martinkorelic.mobiletransformers`.
- Sample app namespace/application ID: `com.martinkorelic.mobiletransformers.app`.
- Public facade package: `com.martinkorelic.mobiletransformers`.

Recommended Android additions:

- Add Gradle publishing metadata to `MobileTransformers`.
- Publish the AAR as `mobiletransformers-android` unless the organization namespace decision forces a different coordinate before the first public release.
- Add `scripts/android_build_aar.sh` and `scripts/publish_local_maven.sh` using the new Gradle paths.
- Keep C++ implementation under `MobileTransformers/src/main/cpp`.
- Add new Kotlin facade APIs under `MobileTransformers/src/main/java/com/martinkorelic/mobiletransformers`.
- Keep existing repository/JNI classes available during migration through deprecated wrappers or typealiases where possible.
- Add public API docs that treat JNI/C++ types as internal implementation details.

Do not move Android code into `src/mobiletransformers/`; that package is Python-only.

## Mobile Foundation Architecture

The Android restructure should create a strong public API foundation without replacing the engine. The guiding rule is: wrap the current working repositories first, then move implementation details behind stable MobileTransformers contracts.

### Mobile Module Package Layout

Use logical Kotlin packages inside the renamed `MobileTransformers` SDK module:

```text
com.martinkorelic.mobiletransformers/
  MobileTransformers.kt
  MobileTransformerModel.kt
  config/
    TrainConfig.kt
    GenerationConfig.kt
    RagConfig.kt
    PeftConfig.kt
    DeviceConfig.kt
    HubConfig.kt
    TrainingScheduleConfig.kt
  hub/
    HuggingFaceHubClient.kt
    HubDownloadRequest.kt
    HubAuthProvider.kt
  packages/
    MobileTransformersManifest.kt
    ModelVariant.kt
    ModelFeature.kt
    ModelPackageValidator.kt
    ModelPackageInstaller.kt
    ChecksumVerifier.kt
    CacheIndex.kt
  runtime/
    ModelSession.kt                 # facade-level whole-model contract (the name ModelRuntime is reserved for the engine boundary, 01_code_plans/03)
    RepositoryBackedModelSession.kt
    InferenceEngine.kt              # declared by 01_code_plans/03; reused here
    RuntimeCapabilities.kt          # model-level flags (engine-level = EngineCapabilities, 01_code_plans/03)
  training/
    DatasetConfig.kt   # single dataset type everywhere (no separate DatasetSource)
    TrainingJob.kt
    TrainingProgress.kt
    AdapterPackage.kt
    TrainableTensorCodec.kt
    ScheduledTrainingManager.kt
  inference/
    GenerationResult.kt
    TokenStreamCallback.kt
  rag/
    RetrievalResult.kt
    VectorStoreConfig.kt
  extensions/
    FederatedRoundConfig.kt
    FederatedAdapterRecord.kt
    ExtensionCapability.kt
  internal/
    legacy/
    jni/
    repositories/
```

This is a namespace plan, not a demand to physically move every class in the first commit. Existing classes such as `LLMRepository`, `TrainingRepository`, `InferenceRepository`, `RagRepository`, `ORTTrainerNative`, and `ORTGeneratorNative` should initially live behind `internal/` adapters or deprecated compatibility wrappers. The public API should stop exposing `ORT*` names even if the native implementation still uses ORT internally.

### Required Foundation Features

These features should land with the mobile restructure so later HF, background-training, RAG, and federated extensions have a stable base:

| Feature | Why it belongs in the restructure | Conservative first implementation |
| --- | --- | --- |
| Public facade | Gives users one HF-style entrypoint and hides repository/JNI details. | `MobileTransformers.fromPretrained(...)` returning `MobileTransformerModel` backed by the current `LLMRepository`. |
| Manifest-first package model | Lets Android understand HF-hosted packages before downloading large files. | Parse `mobiletransformers_manifest.json`, select a variant, validate expected `train/`, `inference/`, and `embedding/` paths. |
| Feature-group downloads | Keeps inference-only, training, RAG, and GenAI/manual packages separable. | `ModelFeature.Inference`, `Training`, `Rag`, `Embedding`, `GenAI`, `ManualInference`, `Adapter`. |
| Cache bridge | Avoids rewriting current model discovery. | Install packages into the existing cache shape used by `LLMRepository`, plus a `CacheIndex` for manifest metadata and storage usage. |
| Variant/capability selection | Lets one HF repo host multiple mobile-ready builds. | Choose by ABI, quantization, memory estimate, requested features, ORT/GenAI support, and default variant. |
| Typed user configs | Replaces leaking `ORT*` details in public APIs. | Public `TrainConfig`, `GenerationConfig`, `RagConfig`, `PeftConfig`, `DeviceConfig`, `HubConfig`; map internally to existing config classes. |
| Dataset abstraction | Needed by local training, scheduled training, and federated rounds. | `DatasetConfig` for local JSONL/text/tokenized datasets, mapped to the existing dataset loader. |
| Training lifecycle API | Needed for foreground use, WorkManager scheduling, and retries. | `TrainingJob` with `start`, `cancel`, progress callbacks, checkpoint path, and final status. |
| Checkpoint/resume contract | Makes sleep/charging-cycle training possible later. | Expose checkpoint metadata already saved in `training_state.json`; do not change native checkpoint format yet. |
| Adapter package contract | Needed for adapter export, Hub push-back, and Flower-style federation. | `AdapterPackage` metadata plus file refs; start with LoRA/MARS trainable tensors already produced by the current pipeline. |
| Weight handoff map | Needed to avoid fragile training-to-inference tensor name rewrites. | `weight_handoff_map.json` maps checkpoint names, merger outputs, inference initializers, GenAI inputs, dtype, shape, quantization fields, and transpose policy. |
| Trainable tensor codec | Needed before any federated learning extension. | Define tensor names, shapes, dtype, order, and aggregation role from manifest metadata; implement round-trip tests before networking. |
| Progress and metrics events | Needed by app UI, scheduled training, evaluation, and federated metrics. | One `TrainingProgress`/`RuntimeEvent` stream backed by current loss, memory, step, and duration metrics. |
| Extension capability registry | Lets Tier 3 features check support without probing internals. | `RuntimeCapabilities`/`ExtensionCapability` values such as `supportsTraining`, `supportsMerge`, `supportsRag`, `supportsScheduledTraining`, `supportsAdapterTensorExport`. |
| Compatibility layer | Keeps existing demos and tests alive. | Deprecated wrappers/typealiases under `com.martinkorelic.ortmobile` for one release where practical. |

### Public API Surface To Stabilize

The first stable Android facade should be intentionally small:

```kotlin
val model = MobileTransformers.fromPretrained(
    context = context,
    repoId = "mobiletransformers/Qwen2-0.5B-mobile",
    revision = "main",
    variant = null,
    features = setOf(ModelFeature.Inference, ModelFeature.Training),
    cacheDir = context.filesDir,
    hubConfig = HubConfig(token = null)
)

model.train(dataset, TrainConfig(maxSteps = 10))
model.merge()
model.generate("Hello", GenerationConfig(maxNewTokens = 16))
model.retrieve("local query", RagConfig(topK = 5))
```

Keep lower-level repositories available for internal and advanced usage, but document the facade as the supported API. Future extensions should compose with `MobileTransformerModel` rather than reaching directly into `ORTTrainerNative` or cache internals.

### HF API And Hub Alignment

The Android package foundation should mirror Hugging Face concepts where they help, while keeping the on-device runtime explicit:

- Use `repoId`, `revision`, `token`, `cacheDir`, and `variant` terminology in public APIs.
- Treat `fromPretrained` as "download or load a MobileTransformers-ready package", not as arbitrary Transformers model loading on device.
- Require a MobileTransformers manifest for Android direct loading. Existing generic ONNX Hub repos can be converted by the Python CLI before Android use.
- Keep `snapshot_download` and `upload_folder` in the Python tooling, while Android implements only the small manifest-first resolver/downloader it needs.
- Store Optimum ONNX metadata, selected task, tokenizer metadata, ORT/GenAI versions, feature groups, checksums, and Android runtime requirements in the manifest.
- Store or reference the weight handoff map so Android never has to infer initializer names from string replacement rules.
- Preserve the current `<cacheDir>/<model>/train`, `<cacheDir>/<model>/inference`, and `<cacheDir>/<model>/embedding` layout behind a package installer so existing repository classes continue to work.

### Extension Readiness Requirements

The restructure should add the contracts that Tier 3 will need, but not implement the extensions yet:

- Sleep/charging-cycle training needs `TrainingScheduleConfig`, `TrainingJob`, foreground-progress events, cancel/resume behavior, and checkpoint metadata.
- Federated learning needs `AdapterPackage`, `TrainableTensorCodec`, deterministic trainable tensor ordering, and metrics export. It does not need Flower inside the Android SDK foundation.
- Encoder support needs `ModelFeature.Embedding`, `ModelFeature.Classification`, output-shape metadata, and task/capability fields in the manifest.
- Function-calling/mobile-actions work needs a future `StructuredGenerationConfig` or validation hook, but should not alter the core facade until GenAI/manual inference choices settle.

### What To Keep Internal For Now

To avoid diverting too far from the original code, keep these as implementation details during the first restructure:

- `System.loadLibrary("ortmobile")` and native shared-library naming, unless JNI smoke tests are already in place.
- `ORTTrainerNative`, `ORTGeneratorNative`, and C++ session/cache classes.
- Current `LLMRepository`, `TrainingRepository`, `InferenceRepository`, and `RagRepository` behavior.
- ObjectBox entity/package moves, unless generated-code tests are ready.
- Existing `training_config.json`, `generation_config.json`, and `rag_config.json` file formats. The manifest should reference and validate them before replacing them.

## Generated Artifact Boundary

Generated files must stay outside tracked source unless they are tiny fixtures:

- `build/`
- `dist/`
- `.cache/`
- `models/`
- `artifacts/`
- `third_party/wheels/*.whl`
- `third_party/onnxruntime/build/`
- `hf_cache/`
- Android app cache/model directories
- downloaded Hub snapshots
- generated starter-zoo packages, except curated manifest examples

Tracked examples should use `tests/fixtures/` or `docs/examples/` and must be small enough for CI.

## Implementation Sequence

1. Add `pyproject.toml`, `uv.lock`, `config/`, `src/mobiletransformers/__init__.py`, CLI stubs, tests directories, `requirements/`, `third_party/onnxruntime/`, and scripts directories.
2. Seed dependency groups from `requirements-ort.txt` and `requirements-or.txt`, then split them into core, export, train, ORT-training-local, RAG, eval, docs, and platform groups.
3. Add local ORT training wheel metadata and placeholder source path in `tool.uv.sources`; do not require the wheel for core install.
4. Add manifest-first package schema, validation fixtures, and cache-install bridge that materializes selected variants into the current `train/`, `inference/`, and `embedding/` cache layout.
5. Rename Android Gradle root/module/app from `ORTTransformer`/`ORTransformersMobile`/`app` to `MobileTransformers`/`MobileTransformers`/`MobileTransformersApp`.
6. Update Android namespaces, app `applicationId`, source package directories, Gradle project dependencies, scripts, and docs to the new MobileTransformers names.
7. Add the Android public facade namespace with `MobileTransformers`, `MobileTransformerModel`, public config classes, package manifest classes, and `RepositoryBackedModelSession` adapters around existing repositories.
8. Add Android-side package validation against the same manifest/cache contract before `fromPretrained` loads any package.
9. Add `TrainingJob`, progress events, checkpoint metadata access, and cancel/resume hooks around the existing training repository before adding WorkManager scheduling.
10. Add `AdapterPackage` and `TrainableTensorCodec` interfaces with no network implementation yet; use them only for local adapter/tensor export-import tests.
11. Move root `config.yml` to `config/config.yml`; leave root compatibility path or loader fallback for one release.
12. Move root `config.py` secrets/constants into `src/mobiletransformers/config/settings.py` and `src/mobiletransformers/config/constants.py`; root `config.py` becomes a deprecation shim importing from the package; update direct imports in evaluation and export code.
13. Add compatibility wrappers for current root packages while new imports stabilize.
14. Migrate `tools/parser_config.py` constants first because many modules depend on them.
15. Migrate export/artifact modules behind CLI entry points.
16. Migrate PEFT modules after export smoke tests are passing.
17. Move reusable evaluation code after the package API is stable; leave exploratory notebooks/scripts in `research/`.
18. Export requirements and SBOM artifacts from `uv.lock` for CI and release documentation.
19. Remove compatibility wrappers only after docs, tests, and user-facing examples all use the package imports.

## What Not To Move In The First Pass

- Do not rename Android internals opportunistically. Do the Gradle/module/app rename as one isolated migration with build verification, then move Kotlin package names and facade APIs deliberately.
- Do not rename public Kotlin classes yet (`LLMRepository`, `TrainingRepository`, `InferenceRepository`, `RagRepository`, `ORTTrainerNative`, `ORTGeneratorNative`).
- Do not rewrite the large inference builder while the Optimum/GenAI decision is unresolved.
- Do not collapse research scripts into the package.
- Do not remove root import compatibility before CI proves the new package path.
- Do not change license metadata in `pyproject.toml` until authors decide whether to keep CC-BY-NC-4.0 or relicense.
- Do not move generated ORT wheels/AARs into tracked source in the first pass.
- Do not force the local source-built `onnxruntime-training` wheel into default/core installs.

## Risks

- Import churn can break research scripts and old notebooks.
- Moving config may break Android artifact generation if JSON paths drift.
- Renaming Android modules and namespaces can break Gradle project dependencies, JNI bindings, ObjectBox generated classes, and sample-app data. It must be isolated and verified before functional changes.
- Optional dependencies can still resolve to incompatible ORT/Optimum combinations if not locked.
- A premature package boundary could hide the fact that training export depends on deprecated tooling.
- `tool.uv.sources` is `uv`-specific. The standards-compliant metadata must remain usable enough for build tools, while local development uses `uv`.
- Local ORT training wheel paths can make onboarding brittle unless the build script and manifest are kept current.

## Tests And Smokes

- `python -m mobiletransformers.cli.main --help`
- `python -m mobiletransformers.cli.export --config config/config.yml --dry-run`
- `uv sync --frozen --group dev --extra export`
- `uv sync --frozen --group ort-training-local --extra train`
- `uv export --format requirements.txt --extra export --output-file requirements/requirements-export.lock.txt`
- Unit tests for settings precedence: CLI overrides env, env overrides YAML, YAML overrides defaults.
- Import-compatibility tests for old module paths during migration.
- Existing Python artifact builder smoke against a tiny model or fixture.
- Android Gradle library assemble after rename: `./gradlew :MobileTransformers:assembleDebug`.
- Android sample app assemble after rename: `./gradlew :MobileTransformersApp:assembleDebug`.
- Android package/import compatibility smoke for old `com.martinkorelic.ortmobile` imports if wrappers/typealiases are kept for one release.
- Android facade compile smoke: import `com.martinkorelic.mobiletransformers.MobileTransformers` and call `fromPretrained` against a tiny local fixture package.
- Manifest/cache bridge smoke: install a fixture package into the existing cache shape and confirm `LLMRepository` can still discover it.
- Package validation tests for feature groups, variants, required files, checksums, model-feature flags, and unsupported extension capability errors.
- Training lifecycle smoke: start a one-step `TrainingJob`, emit progress, cancel or complete cleanly, and expose checkpoint metadata.
- Adapter/tensor codec round-trip smoke for one tiny LoRA/MARS fixture without any federated networking.

## Acceptance Criteria

- The repo has a documented target hierarchy and migration map.
- Config layering is explicit and secrets no longer require direct `os.environ` reads in business logic.
- `pyproject.toml` exposes install extras, dependency groups, local ORT source metadata, and CLI entry points.
- `uv.lock` is the source of truth for local reproducible environments, with exported requirements files only as generated integration artifacts.
- Android remains a first-class SDK workspace under `android/MobileTransformers/`.
- Android SDK module is named `MobileTransformers`, and the sample app module is named `MobileTransformersApp`.
- Android exposes a small HF-style public facade under `com.martinkorelic.mobiletransformers`.
- The facade wraps existing repositories instead of requiring an engine rewrite.
- Manifest-first package validation and cache installation are defined before direct Android Hub download is added.
- Public config objects map to existing `ORT*Config` internals without leaking ORT naming into the stable API.
- Training lifecycle, progress, checkpoint/resume, adapter package, and trainable tensor codec contracts exist as extension foundations.
- Generated artifacts are clearly separated from tracked source.
- Existing workflows remain runnable through compatibility wrappers during the transition.

## Source Links

- PyPA `pyproject.toml` guide: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- PyPA src layout discussion: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- uv dependency management: https://docs.astral.sh/uv/concepts/projects/dependencies/
- uv locking and syncing: https://docs.astral.sh/uv/concepts/projects/sync/
- uv lockfile export: https://docs.astral.sh/uv/concepts/projects/export/
- Poetry dependency specification: https://python-poetry.org/docs/dependency-specification/
- Optimum ONNX PyPI: https://pypi.org/project/optimum-onnx/
- ONNX Runtime training build: https://onnxruntime.ai/docs/build/training.html
- Pydantic Settings: https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/
- Hugging Face Hub download guide: https://huggingface.co/docs/huggingface_hub/guides/download
- Hugging Face Hub upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload
- Android WorkManager: https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started/define-work
- Android long-running workers: https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running
- Android library publishing: https://developer.android.com/build/publish-library
- Gradle Maven Publish Plugin: https://docs.gradle.org/current/userguide/publishing_maven.html
- ONNX Runtime on-device training: https://onnxruntime.ai/docs/get-started/training-on-device.html

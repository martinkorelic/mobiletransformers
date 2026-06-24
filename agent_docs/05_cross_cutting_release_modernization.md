# Cross-Cutting Release Modernization

> Detailed code-implementation plans for the build/CI/AAR/docs/release work live in `agent_docs/05_code_plans/`; global order in `agent_docs/IMPLEMENTATION_ORDER.md`.
>
> **Public contracts are typed.** The config enums, Pydantic models + generated JSON Schemas, and the PEFT/architecture/merger registries from `00_code_plans/09` are documented as public extension points (`docs/CONFIGURATION.md`); the compatibility matrix is enumerated from those registries so it cannot drift from the code.

## Purpose

This plan covers the work that runs alongside all tiers: build automation, CI, documentation, packaging, release/versioning, compatibility matrix, AAR publication, and final `v1.0` readiness. These tasks make the project installable, testable, citable, and maintainable after the research push.

## Current Repo Evidence

- There is no `Makefile`, `pyproject.toml`, `uv.lock`, `setup.py`, CI workflow, `scripts/` directory, changelog, or git tag (0 tags).
- The Android library module exists and can be the AAR/Maven artifact: `android/ORTransformer/ORTransformersMobile` (namespace `com.martinkorelic.ortmobile`). No `maven-publish`/publishing block exists.
- The Android sample app exists separately: `android/ORTransformer/app` (app namespace override `com.martinkorelic.orttransformer`, applicationId `com.martinkorelic.ortmobile`). Root Gradle project name is `ORTTransformer`.
- **`ORTransformersMobile/build.gradle.kts` declares `implementation(files("./src/main/aarLibs/onnxruntime-genai.aar"))` but that file/dir does not exist in the repo**, and the `srcDirs("libs")` JNI dir is also absent — both must be resolved before a clean AAR build/publish. ABIs configured: `arm64-v8a`, `x86_64`.
- Existing docs include `README.md`, `docs/mobile_evaluation.md`, and visual demo assets (`docs/*.gif`).
- `requirements-or.txt` (`optimum==1.23.2`, `torch-ort==1.19.2`, `peft==0.13.2`) and `requirements-ort.txt` (`onnxruntime-training==1.23.0+cpu`, `optimum==1.23.3`, `onnxscript==0.3.1`) are pinned to **pre-split Optimum** (the ONNX split into `optimum-onnx` came later); they define no package extras. Reconciling these with the Tier-0.3 toolchain decision is part of `00_code_plans/03`.
- The repo already contains evaluation code (`evaluation/`: `eval_adapter_models.py`, `mobile_evaluator.py`, `benchmark/`, `mobile/`) for benchmarks, mobile traces, and task plots, but those are standalone and not wired into CI.
- `LICENSE.md` is CC-BY-NC-4.0.
- `CITATION.cff` already declares `version: 1.0.0`, `date-released: 2025-10-18` — both must be reconciled with the actual v1.0 release (SemVer 1.0.0 should match the tagged release, not predate it).
- **`agent_docs/` is currently listed in `.gitignore`** — these planning/code-plan docs are untracked. Decide whether to un-ignore them so the plan ships with the release (non-blocking; see `05_code_plans/05`).

## External Research Summary

- Modern Python packaging should use `pyproject.toml` for project metadata, optional dependencies, and entry points. Source: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Hugging Face Hub upload docs support model/package publication, including model cards. Source: https://huggingface.co/docs/hub/models-uploading
- ONNX Runtime on-device training docs describe the offline artifact generation plus runtime training split that must be documented for users. Source: https://onnxruntime.ai/docs/get-started/training-on-device.html
- Maven Central already distributes ONNX Runtime Training Android AARs, so MobileTransformers should provide an AAR/local Maven consumption path too. Source: https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-training-android
- Android's library publishing docs describe release preparation, publication variants, and variant-aware Gradle Module Metadata for Android libraries. Source: https://developer.android.com/build/publish-library
- Gradle's Maven Publish Plugin provides `MavenPublication`, `publishToMavenLocal`, generated POM metadata, and Maven-repository publication tasks. Source: https://docs.gradle.org/current/userguide/publishing_maven.html
- GitHub Actions workflow syntax supports matrix jobs, timeouts, fail-fast behavior, and reusable workflows, which fit the staged Python/Android/device CI design. Source: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- Semantic Versioning requires a declared public API before `1.0.0`; this is especially important because the Android facade, model package format, and CLI commands become compatibility contracts at release. Source: https://semver.org/spec/v2.0.0.html
- SPDX license identifiers provide machine-readable license metadata and reduce ambiguity across source files, package metadata, and published artifacts. Source: https://spdx.dev/learn/handling-license-info/
- CC-BY-NC-4.0 is non-commercial; that is suitable for research sharing but a real adoption constraint for a reusable Android/HuggingFace framework. Source: https://creativecommons.org/licenses/by-nc/4.0/
- MobileFineTuner sets an expectation for clean Android developer adoption and API ergonomics. Source: https://arxiv.org/abs/2512.08211

## Recommended Decision

Create a release train around one repeatable command path, but only after the foundational contracts are in place:

```bash
make setup
make export-model MODEL=Qwen/Qwen2-0.5B PEFT=mars-opt1 QUANT=qint8
make test-smoke
make build-aar
make publish-local
```

The commands should be thin wrappers around Python CLI and Gradle. The goal is not to hide complexity forever; it is to make the happy path obvious and CI-compatible.

The global order should be foundation-first:

1. Stabilize package/config/dependency/build scaffolding.
2. Prove the source-built ORT training and export toolchain.
3. Define the MobileTransformers manifest/cache/public API contracts.
4. Rename Android modules and add the facade around existing repositories.
5. Decide the inference engine handoff and GenAI/manual default.
6. Build HF pull/export/push workflows against the stable model format.
7. Promote inference/RAG into first-class subsystems.
8. Add cross-cutting release automation and docs continuously.
9. Attempt extensions only after the core contracts are tested.

Anything that would change `mobiletransformers_manifest.json`, Android cache layout, public Kotlin config names, trainable tensor ordering, or CLI command names should be treated as foundational and resolved before it is consumed by Tier 1 or Tier 2 implementation.

## Global Implementation Order

This order overrides the shorter per-tier lists when there is a conflict. The per-tier docs describe local work; this section describes the safe release spine.

### Phase 0 - Baseline And Freeze Points

- Record current working commands for Python export, ORT artifact generation, Android library build, sample app build, train one step, merge, generate, and RAG if available.
- Add fixture or tiny-model smokes before moving packages or Android modules.
- Record current source-built ORT wheel/AAR provenance enough to reproduce the current state.
- Decide which existing root modules and Android classes are compatibility surfaces for one release.

Exit gate: the team can prove whether a future migration broke something that worked before.

### Phase 1 - Python Package, Config, And Dependency Foundation

- Add `pyproject.toml`, `uv.lock`, package metadata, dependency groups, public extras, and exported requirements.
- Add `config/config.yml`, `config/settings.py`, `config/constants.py`, and config precedence tests.
- Add package skeleton under `src/mobiletransformers/` but keep root compatibility wrappers.
- Keep implementation moves small until the export/toolchain gates pass.

Exit gate: `uv sync` can install at least core, export, and local ORT-training profiles, and current root imports still work.

### Phase 2 - ORT/Optimum/Export Toolchain Gate

- Reproduce source-built ORT Training wheel/AAR creation and manifest metadata.
- Prove `onnxruntime.training.artifacts.generate_artifacts` on a tiny fixture.
- Add Optimum ONNX support discovery with `TasksManager`.
- Decide the durable inference-export path: Optimum ONNX, direct `torch.onnx`, or temporary legacy Optimum.
- Generate the first `model_support_matrix.json`.

Exit gate: one tiny model can produce inference artifacts, training artifacts, and a recorded support-matrix entry.

### Phase 3 - MobileTransformers Model Package Contract

- Define and version `mobiletransformers_manifest.json`.
- Define variant IDs, feature groups, required files, checksums, Optimum metadata, ORT/GenAI metadata, Android runtime requirements, and cache mapping.
- Add package validators and tiny fixtures in Python and Android.
- Do not implement direct Hub download until the manifest and cache installer are stable.

Exit gate: a fixture package can validate and materialize into the current Android cache shape.

### Phase 4 - Android Rename And Public Facade

- Rename Gradle root/module/app to `MobileTransformers`, `:MobileTransformers`, and `:MobileTransformersApp`.
- Add public facade classes under `com.martinkorelic.mobiletransformers`.
- Wrap existing `LLMRepository`, `TrainingRepository`, `InferenceRepository`, and `RagRepository` through a repository-backed runtime.
- Add compatibility wrappers/typealiases where practical for old `com.martinkorelic.ortmobile` imports.
- Keep native library names and C++ internals stable unless JNI smokes cover the rename.

Exit gate: the renamed SDK and sample app build, the facade can load a local fixture package, and existing repository behavior still works.

### Phase 5 - Tier 0 GenAI/Manual Inference Decision

- Emit a GenAI package from the same manifest/artifact format.
- Prove or reject GenAI model-input injection, initializer injection, and adapter loading.
- Run manual low-copy or mmap experiments against the same tiny package.
- Record the engine decision in the support matrix and Tier 2 docs.

Exit gate: there is one supported train -> merge -> infer handoff path with memory measurements.

### Phase 6 - Tier 1 HF-Integrated Core

- Add Python export/pull/install/push commands around the stable manifest.
- Add Android manifest-first Hub downloader only after local install works.
- Publish one starter package candidate after license and model-card checks.
- Move the sample app to the public facade.

Exit gate: one command can build a package and one facade call can load it locally or from a prepared Hub-style package.

### Phase 7 - Tier 2 Inference And RAG

Code plans: `03_code_plans/01`–`05`.

- Wire the native path into the existing `ModelRuntime` boundary (`01_code_plans/03`) — do **not** add a new `InferenceEngine` interface — and retire the `inference/merged/` probe (`03_code_plans/01`).
- Harden the chosen inference engine without changing public generation config names; align sampling/streaming public names (`03_code_plans/02`).
- Add `VectorStore` boundary, `InMemoryVectorStore`, ObjectBox wrapper (`03_code_plans/03`), ingestion (`03_code_plans/04`), and grounded generation helper (`03_code_plans/05`).

Exit gate: train/merge/generate and ingest/retrieve/generate smokes pass through public APIs.

### Phase 8 - Release Modernization

Code plans: `05_code_plans/01`–`05`.

- Add Makefile targets + `scripts/` (`05_code_plans/01`), staged CI (`05_code_plans/02`), local Maven publication + AAR (`05_code_plans/03`), docs set + compatibility matrix (`05_code_plans/04`), and versioning/license/release (`05_code_plans/05`).
- Keep fast CI small and push model-zoo/device checks to manual or scheduled workflows.
- Resolve license before declaring `1.0.0`.

Exit gate: release artifacts can be built, tested, documented, and consumed by a tiny external Android app.

### Phase 9 - Tier 3 Extensions

Code plans: `04_code_plans/01`–`05`.

- Start only after Phases 0-8 have stable contracts.
- Encoder support may begin first because it mostly extends task/model metadata (`04_code_plans/01`).
- Scheduled training requires `TrainingJob`, checkpoint metadata, progress events (`00_code_plans/08`), and Android foreground-service decisions (`04_code_plans/02`).
- Flower/federated work builds the `FederatedAdapterRecord` on top of the existing `TrainableTensorCodec` (`00_code_plans/07`); Python simulation first, Android gateway second (`04_code_plans/03`, `04`).
- FunctionGemma/mobile-actions requires the inference engine decision, the Gemma-3 **inference-graph** export branch, and the structured-output validation gate (`04_code_plans/05`).

Exit gate: every extension is isolated behind capability checks and can fail without changing the core API.

## Reimplementation Avoidance Gates

Do not cement these contracts until their validators and smokes exist:

- Public Android facade names and config object fields.
- `mobiletransformers_manifest.json` schema and artifact format version.
- Android cache layout and active/staged package atomicity.
- Training checkpoint/resume metadata exposed to WorkManager.
- Adapter package format and trainable tensor ordering.
- CLI command names and arguments.
- Maven coordinates and SemVer public API declaration.

If a feature needs one of these contracts and the contract is not validated yet, implement the validator first.

## Makefile Targets

Add:

- `make setup`: install Python package with dev/export extras.
- `make lint`: run formatter/linter checks after the toolchain is chosen.
- `make test`: run Python unit tests.
- `make test-smoke`: run tiny export/artifact/inference smoke tests.
- `make export-model MODEL=... PEFT=... QUANT=...`: build a MobileTransformers-ready package.
- `make package-model`: validate and package an existing build directory.
- `make android-build`: run Gradle assemble for app and library.
- `make build-aar`: assemble `MobileTransformers` release AAR after the rename; during transition, it may call the legacy `ORTransformersMobile` task through a compatibility target.
- `make publish-local`: publish library to local Maven.
- `make docs`: build docs if MkDocs or another docs system is used.
- `make clean-generated`: remove generated build/model artifacts only.

Avoid destructive cleanup of user caches by default.

## CI Plan

Use a staged CI design:

### Fast Checks

- Python import checks.
- Unit tests for config/settings/manifest.
- Kotlin/JVM unit tests where possible.
- Markdown link check for docs if available.

### Export Smoke

- Run one tiny export or fixture-based artifact validation.
- Validate package manifest.
- Run a one-token or one-step desktop smoke when feasible.

### Android Build

- Before the rename, build `:ORTransformersMobile:assembleDebug` and `:app:assembleDebug`.
- After the rename, build `:MobileTransformers:assembleDebug` and `:MobileTransformersApp:assembleDebug`.
- Optionally build release AAR on tags.

### Device/Manual CI

Run on a physical device or scheduled manual workflow:

- train one step,
- merge,
- generate one token,
- ingest/query one RAG document,
- record basic time and memory metrics.

Do not make large model-zoo builds mandatory on every PR. Run those nightly or pre-release.

## Documentation Set

Create or update:

- `docs/ARCHITECTURE.md`: training/export/runtime architecture and data flow.
- `docs/PUBLIC_API.md`: Kotlin and Python public API.
- `docs/MODEL_FORMAT.md`: MobileTransformers-ready package manifest and required files.
- `docs/ANDROID_SDK.md`: Gradle/AAR setup, permissions, ABI notes, local Maven.
- `docs/RAG.md`: embedding model, ingestion, vector store, retrieval, grounded generation.
- `docs/EXPORT.md`: one-command export and dependency/toolchain notes.
- `docs/COMPATIBILITY_MATRIX.md`: model family x PEFT x quantization x task x device.
- `docs/RELEASE_CHECKLIST.md`: v1.0 release gate checklist.
- `CHANGELOG.md`: release notes.

Keep `agent_docs/` as research/planning docs. Public user docs should live under `docs/`.

## Compatibility Matrix

Track:

- Model family: Qwen2, SmolLM2, TinyLlama, Phi, Gemma, BERT/MiniLM.
- Task type: text generation, feature extraction, classification, RAG embeddings.
- PEFT method: LoRA, LoRA-XS, MARS OPT0, MARS OPT1, quantized MARS.
- Quantization: none, QInt8, QUInt8, int4 where supported.
- Export path: legacy Optimum, Optimum ONNX, direct `torch.onnx`.
- Inference engine: native, GenAI.
- Device: Pixel 6, Samsung Galaxy S21 FE, emulator where useful.
- Status: supported, experimental, blocked, not tested.
- Evidence: link to test run, issue, or doc note.

## AAR And Maven Publication

Add Gradle publishing support to the renamed `MobileTransformers` SDK module:

- group ID: `com.martinkorelic.mobiletransformers` or final organization namespace.
- artifact ID: `mobiletransformers-android`.
- version: from Gradle property or release tag.
- publication includes AAR, sources if feasible, and POM metadata.

Scripts:

- `scripts/android_build_aar.sh`
- `scripts/publish_local_maven.sh`

Consumer Gradle example:

```kotlin
repositories {
    mavenLocal()
}

dependencies {
    implementation("com.martinkorelic.mobiletransformers:mobiletransformers-android:1.0.0")
}
```

The AAR plan must include native library and ABI expectations for `arm64-v8a` and `x86_64`.

## Release And Versioning

Adopt semantic versioning:

- `0.x`: research/pre-v1 work.
- `1.0.0`: first stable package/API/model-format release.
- Patch releases: bug fixes and doc fixes.
- Minor releases: new model families, PEFT modes, or optional inference engines.

Before `v1.0.0`:

- Decide license.
- Update `CITATION.cff`.
- Add `CHANGELOG.md`.
- Add release notes with limitations.
- Tag release.
- Publish AAR or local-Maven instructions.
- Publish at least one ready model package or document how to build it.

## License Decision

Current license: CC-BY-NC-4.0.

Release decision options:

- Keep CC-BY-NC-4.0: best for research control, weaker for broad developer adoption.
- Relicense code to Apache-2.0 and keep docs/data separately licensed: stronger for framework adoption, requires author agreement.
- Dual-license: possible but needs clear legal text and contributor rights.

This is a release blocker if the stated goal is broad Android/HuggingFace ecosystem adoption.

## Implementation Sequence

Follow the global order above. For the cross-cutting work itself:

1. Add `docs/RELEASE_CHECKLIST.md`, `CHANGELOG.md`, and baseline smoke-command documentation before large moves.
2. Add `pyproject.toml`, `uv.lock`, config scaffolding, and initial `Makefile` targets.
3. Add fast CI for package import, settings, manifest/schema validation, and root-import compatibility.
4. Add source-built ORT training provenance checks and fixture export smoke.
5. Add Android assemble CI for the current module names; update it immediately after the MobileTransformers rename.
6. Add Gradle local Maven publishing for the renamed `MobileTransformers` module.
7. Add consumer-app/local-Maven smoke after the AAR has stable coordinates.
8. Add docs for API, model format, Android SDK, export, inference, RAG, and compatibility matrix as each contract stabilizes.
9. Add scheduled/manual device workflows only after tiny local smokes pass.
10. Resolve license, update `CITATION.cff`, finalize release notes, and tag `v1.0.0`.

## Risks

- CI can become too slow if it downloads large models on every PR.
- Public docs can drift from research docs unless ownership is clear.
- AAR publication may fail if native libraries or local third-party AARs are not packaged correctly.
- License change requires all relevant rights holders to agree.
- Versioning a model package format too early can lock in poor names.
- Adding extension code before manifest, config, checkpoint, and tensor-codec contracts are validated can force a second API migration.

## Tests And Smokes

- `make setup`
- `make test`
- `make test-smoke`
- `make android-build`
- `make build-aar`
- `make publish-local`
- Consumer app compiles against local Maven artifact.
- Package manifest validates for one starter model.
- Device smoke: train -> merge -> infer -> RAG query.
- Docs links and code snippets are checked before release.

## Acceptance Criteria

- One-command build/export/test targets exist.
- CI covers fast Python checks, package manifest checks, and Android assemble.
- AAR/local Maven consumption is documented and tested.
- Public docs exist for API, architecture, model format, Android SDK, export, and RAG.
- Compatibility matrix records tested and untested combinations.
- Changelog and version tags exist.
- License decision is documented before `v1.0.0`.

## Source Links

- PyPA `pyproject.toml` guide: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Hugging Face model uploads: https://huggingface.co/docs/hub/models-uploading
- ONNX Runtime on-device training: https://onnxruntime.ai/docs/get-started/training-on-device.html
- ONNX Runtime Training Android AAR: https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-training-android
- Android library publishing: https://developer.android.com/build/publish-library
- Android publication variants: https://developer.android.com/build/publish-library/configure-pub-variants
- Gradle Maven Publish Plugin: https://docs.gradle.org/current/userguide/publishing_maven.html
- GitHub Actions workflow syntax: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- Semantic Versioning 2.0.0: https://semver.org/spec/v2.0.0.html
- SPDX license identifiers: https://spdx.dev/learn/handling-license-info/
- Creative Commons BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
- MobileFineTuner: https://arxiv.org/abs/2512.08211

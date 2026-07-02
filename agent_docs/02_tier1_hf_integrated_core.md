# Tier 1 - HuggingFace-Integrated Core

## Purpose

Tier 1 turns MobileTransformers into a self-serve framework. A developer should bring a dataset, call a HuggingFace-style API, and receive a device-ready training, merge, inference, and optional adapter-publish workflow without manually assembling ONNX artifacts.

> Detailed code-implementation plans for these features live in `agent_docs/02_code_plans/` (and cross-referenced 00/01 plans); global order in `agent_docs/IMPLEMENTATION_ORDER.md`.

## Current Repo Evidence

- The Android public surface is repository-oriented: `LLMRepository`, `TrainingRepository`, `InferenceRepository`, and `RagRepository`.
- The Android Gradle workspace is currently named `ORTTransformer`, with modules `:ORTransformersMobile` and `:app`.
- The Android library module namespace is currently `com.martinkorelic.ortmobile`, while the app code lives under `com.martinkorelic.orttransformer`.
- The sample app depends on `implementation(project(":ORTransformersMobile"))`, so the app and SDK names are still ORT-branded even though the framework is now MobileTransformers.
- `MainActivity` still displays `ORTransformersMobile` and loads the native library through `System.loadLibrary("ortmobile")`, so branding, native names, and public API naming need a staged migration.
- `LLMRepository` discovers local model directories under a cache directory and expects `train/training_config.json`, `inference/generation_config.json`, and `embedding/rag_config.json`.
- `ORTTrainingConfig`, `ORTGenerationConfig`, and `ORTRagConfig` already exist, but they use MobileTransformers-specific naming instead of HuggingFace conventions.
- Python export is already capable of pulling HuggingFace models with `AutoModelForCausalLM`, `AutoConfig`, `AutoTokenizer`, and `HF_TOKEN`.
- `config.yml` already lists candidate model families such as TinyLlama, Phi, Qwen2, SmolLM2, DeepSeek-R1-Distill-Qwen, and embedding models.
- Artifact packaging already emits `training_config.json`, `generation_config.json`, tokenizer files, RAG config, and inference/merged weight paths.
- MARS is implemented as a custom PEFT type and is the primary differentiator versus generic LoRA systems.

## External Research Summary

- Hugging Face Hub download tooling supports snapshot-style repo downloads and file filtering, which maps well to ready-made model packages. Source: https://huggingface.co/docs/huggingface_hub/guides/download
- Hugging Face Hub download APIs expose file URLs, ETags, filtered snapshot downloads, `local_dir`, dry-run metadata, and revision pinning. Desktop tooling should use `snapshot_download(..., allow_patterns=...)`; Android should implement a small resolver/downloader around Hub resolve URLs rather than trying to embed the Python SDK. Source: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- Hugging Face Hub supports model uploads and model-card-centric sharing, which should be used for starter packages and trained adapters. Source: https://huggingface.co/docs/hub/models-uploading
- Hub repos can host custom model formats, not only Transformers-compatible checkpoints, as long as the repo documents the files with a model card. Source: https://huggingface.co/docs/hub/models-uploading
- `huggingface_hub.upload_folder` supports robust folder uploads through the Hub/Xet backend, making it appropriate for publishing large exported artifact directories from the Python CLI. Source: https://huggingface.co/docs/huggingface_hub/guides/upload
- Optimum ONNX is now a separate package that exports Transformers, Diffusers, Timm, and Sentence Transformers models to ONNX and includes an expanding supported-architecture list. Source: https://pypi.org/project/optimum-onnx/
- optimum-onnx 0.1.0 pins `transformers>=4.36,<4.58`, so v1 must pin transformers 4.x. The export path's `optimum.exporters.onnx` imports largely survive the optimum/optimum-onnx split (it is mostly which package you install), but the lower-level `OnnxConfigWithLoss` / `export` symbols must be verified in the Tier 0 spike. Source: https://pypi.org/project/optimum-onnx/
- Optimum ONNX exposes `TasksManager` for discovering supported ONNX tasks by model type and custom ONNX config hooks for advanced exports. Source: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
- Existing ONNX Hub packages show useful patterns: `onnx-community/Llama-3.2-1B` stores ONNX weights under an `onnx/` subfolder, while Microsoft's Phi-3 ONNX repo separates `cpu_and_mobile`, `cuda`, and `directml` variants and recommends a CPU/mobile int4 acc-level-4 package for mobile devices. Sources: https://huggingface.co/onnx-community/Llama-3.2-1B, https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
- ONNX Runtime GenAI conventionally consumes prepared ONNX model directories and config files; matching that package style reduces ecosystem friction if GenAI is adopted. Source: https://onnxruntime.ai/docs/genai/
- MobileFineTuner raises the bar on trainer ergonomics and Android packaging, so MobileTransformers should compete through a fuller system rather than trainer minimalism. Source: https://arxiv.org/abs/2512.08211

## Recommended Decision

Add a HuggingFace-style facade on top of the existing repository/engine classes. Do not replace `LLMRepository` or the JNI/C++ engine in Tier 1. Instead, create a new public API that wraps the current lower-level API and gradually moves app examples to the facade.

Tier 1 should also complete the public Android rebrand: rename `ORTransformersMobile` to `MobileTransformers` and rename the sample `app` module to `MobileTransformersApp`. This is not just cosmetics; the HF-style API should live in the new `com.martinkorelic.mobiletransformers` package, and the sample app should demonstrate the new public facade instead of wiring directly to repository classes.

For model-family support, align with Optimum ONNX instead of maintaining a fully separate support list. A model becomes a MobileTransformers candidate when Optimum ONNX can export its requested task, but it becomes MobileTransformers-ready only after the package manifest, training artifacts, tokenizer/config metadata, Android inference smoke, and optional RAG smoke pass.

Tier 1 should consume the mobile foundation contracts from `00_repository_restructure_plan.md` rather than creating a parallel Android framework. In practice, this means:

- `MobileTransformers.fromPretrained(...)` uses the manifest-first package validator and cache installer defined by the restructure plan.
- `MobileTransformerModel` delegates to a repository-backed runtime adapter around `LLMRepository`, `TrainingRepository`, `InferenceRepository`, and `RagRepository`.
- Public `TrainConfig`, `GenerationConfig`, `RagConfig`, `PeftConfig`, `DeviceConfig`, and `HubConfig` map internally to the current `ORT*Config` types.
- Direct Hub download, Python pull/install, starter-zoo packages, and sample-app loading all share the same `mobiletransformers_manifest.json` schema and cache path mapping.
- Scheduled training, adapter export, and federated-learning work only depend on stable foundation contracts such as `TrainingJob`, `AdapterPackage`, `TrainableTensorCodec`, progress events, and `RuntimeCapabilities`.

Target Android/Kotlin shape:

```kotlin
val model = MobileTransformers.fromPretrained(
    context = context,
    repoId = "mobiletransformers/Qwen2-0.5B-mobile",
    cacheDir = filesDir.absolutePath
)

model.applyPeft(PeftConfig.MarsOpt1(rank = 8))
model.train(localDataset, TrainConfig(epochs = 4, batchSize = 4))
model.merge()
val response = model.generate("Write a short reply")
```

Target Python CLI shape:

```bash
mobiletransformers export \
  --model Qwen/Qwen2-0.5B \
  --peft mars-opt1 \
  --rank 8 \
  --quant int8 \
  --output build/packages/qwen2-0.5b-mobile
```

## Public API Plan

### Android SDK And App Restructure

Target Android workspace:

```text
android/MobileTransformers/
  settings.gradle.kts
  MobileTransformers/
    build.gradle.kts
    src/main/java/com/martinkorelic/mobiletransformers/
    src/main/cpp/
  MobileTransformersApp/
    build.gradle.kts
    src/main/java/com/martinkorelic/mobiletransformers/app/
```

Rename plan:

| Current | Target | Tier 1 action |
| --- | --- | --- |
| Gradle root `ORTTransformer` | `MobileTransformers` | Update `rootProject.name` and wrapper/script references. |
| Module `:ORTransformersMobile` | `:MobileTransformers` | The reusable Android SDK/AAR module. |
| Module `:app` | `:MobileTransformersApp` | The sample/dev app module. |
| Library namespace `com.martinkorelic.ortmobile` | `com.martinkorelic.mobiletransformers` | Put the new facade here. |
| App namespace `com.martinkorelic.orttransformer` | `com.martinkorelic.mobiletransformers.app` | Align sample app source with the product name. |
| App `applicationId = "com.martinkorelic.ortmobile"` | `com.martinkorelic.mobiletransformers.app` | Accept reinstall/data reset for the sample app. |
| `implementation(project(":ORTransformersMobile"))` | `implementation(project(":MobileTransformers"))` | Keep sample app dependency obvious. |

Compatibility approach:

- Add the new `com.martinkorelic.mobiletransformers` facade first.
- Keep old repository classes usable during Tier 1 so existing demos/tests do not break.
- If Kotlin package migration happens before `v1.0`, provide deprecated typealiases or wrapper classes under `com.martinkorelic.ortmobile` for one release.
- Do not rename native shared library identifiers in the same change as the Gradle/module rename unless JNI tests are already in place. `System.loadLibrary("ortmobile")` can remain an internal detail briefly while the public SDK becomes `MobileTransformers`.
- Move ObjectBox/generated-entity package names carefully because generated classes can break when namespace/source packages move.

### Kotlin Facade

Add a high-level facade in the Android library:

- `MobileTransformers.fromPretrained(context, repoId, cacheDir, revision = null, token = null)`
- `MobileTransformerModel.applyPeft(peftConfig)`
- `MobileTransformerModel.train(dataset, config, callback = null)`
- `MobileTransformerModel.merge()`
- `MobileTransformerModel.generate(prompt, config = null, callback = null)`
- `MobileTransformerModel.retrieve(query, config = null, callback = null)`
- `MobileTransformerModel.pushAdapter(...)` only if Hub upload is feasible from device; otherwise expose as Python-side package operation first.

The facade should own:

- Model package validation.
- Delegation to `LLMRepository`.
- Conversion from HuggingFace-style arguments to existing `ORTTrainingConfig`, `ORTGenerationConfig`, and `ORTRagConfig`.
- Friendly error messages for missing artifacts.
- Isolation of legacy `ORT*` class names from the public MobileTransformers API.

### Config Objects

Introduce user-facing config names:

- `TrainConfig`
- `GenerationConfig`
- `RagConfig`
- `PeftConfig`
- `DeviceConfig`
- `DatasetConfig`

Keep existing `ORT*Config` classes as internal or compatibility types for now.

### PEFT Surface

Expose:

- `PeftConfig.Lora(rank, alpha, targetModules)`
- `PeftConfig.MarsOpt0(rank, alpha, targetModules)`
- `PeftConfig.MarsOpt1(rank, alpha, targetModules)`
- `PeftConfig.MarsQuantized(...)` if the optimization-level names need more precision.

Map the public names to the current `train_method`, `lora_rank`, `lora_alpha`, and `mars.optimization_level` options.

## Optimum ONNX Support Alignment Plan

Use Optimum ONNX as the first model-support filter and inference-export engine.

Proposed flow for `mobiletransformers export --model <repo>`:

1. Load `AutoConfig` and read `model_type`, architecture names, tokenizer class, and any `trust_remote_code` requirement.
2. Query `TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")`.
3. Pick a task automatically when safe, prioritizing `text-generation-with-past`, `text-generation`, `feature-extraction`, then `sentence-similarity`; otherwise require `--task`.
4. Export inference ONNX with `optimum-cli export onnx` or `optimum.exporters.onnx.main_export`.
5. Normalize the exported ONNX files into MobileTransformers package paths, including external data files, tokenizer files, generation config, and optional `genai_config.json`.
6. Run a graph-name normalization pass for KV-cache names and trainable-weight inputs/initializers expected by the Android engine.
7. Generate ORT training artifacts with the source-built `onnxruntime-training` wheel only after the PEFT/MARS trainable modules are known.
8. Emit a support matrix entry and package manifest status.

Support inheritance rule:

- `optimum_exportable`: Optimum ONNX can export the model/task.
- `mobile_package_exportable`: MobileTransformers can normalize files, tokenizer metadata, generation config, and manifest.
- `train_artifacts_exportable`: ORT training artifacts and PEFT/MARS metadata generate.
- `android_inference_ready`: Android can load the package and generate at least one token.
- `android_training_ready`: Android can run one train step and merge/apply updated weights.
- `rag_ready`: embedding artifacts and ObjectBox/vector-store config validate.

Only the last ready statuses should be used in user-facing starter-zoo docs. Earlier statuses are useful for contributors because they show where a new Optimum-supported model is blocked.

Candidate support matrix shape:

```json
{
  "modelId": "Qwen/Qwen2-0.5B",
  "modelType": "qwen2",
  "optimumOnnxVersion": "0.1.0",
  "transformersVersion": "x.y.z",
  "supportedTasks": ["text-generation", "text-generation-with-past"],
  "selectedTask": "text-generation-with-past",
  "trustRemoteCode": false,
  "statuses": {
    "optimum_exportable": true,
    "mobile_package_exportable": true,
    "train_artifacts_exportable": false,
    "android_inference_ready": false,
    "android_training_ready": false,
    "rag_ready": false
  },
  "blockers": ["MARS target module mapping not verified"]
}
```

This gives the project a sustainable way to benefit whenever Optimum ONNX adds another supported architecture while keeping the Android SDK honest.

## Hub Model Package Plan

Define a "MobileTransformers-ready" repo as a manifest-first HF model repository. The repo should be readable by standard Hub tools, but MobileTransformers should only require a small manifest fetch before downloading large files.

Recommended repository shape (illustrative sketch — the **canonical, current layout and manifest field list live in `02_code_plans/03_hub_model_package_format.md`**, which supersedes details below; notably the variant `inference/` dir is the FLAT canonical layout of `01_code_plans/01` — `model.onnx` + `frozen_base.onnx.data` + per-tensor `.bin`s, no `base/`/`merged/` subdirs — and merger graphs use descriptive registry filenames, not `merger 1/2` names):

```text
README.md
mobiletransformers_manifest.json
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
licenses/
optimum/
  export_report.json
  supported_tasks.json
  optimum_config.json
shared/
  tokenizer/
  chat_template.jinja
  default/
    train/
      training_model.onnx
      eval_model.onnx
      optimizer_model.onnx
      checkpoint/
      training_config.json
      trainable_parameters.json
      weight_handoff_map.json
    inference/
      model.onnx
      model.onnx.data
      generation_config.json
      genai_config.json
      session_options.json
    merger/
  embedding/
    embedding_model.onnx
    tokenizer/
    rag_config.json
  checksums.json
variants/
  cpu-int4/
    train/
    inference/
    embedding/
    checksums.json
  cpu-fp16/
    train/
    inference/
    embedding/
    checksums.json
```

`default/` exists so the simplest Android and CLI flow can download one known-good package. `variants/` exists for model families that need multiple mobile choices, such as int4 performance/accuracy variants, fp16 desktop smoke variants, or GenAI/manual inference variants.

On Android, the selected variant should be materialized into the existing cache shape expected by `LLMRepository`:

```text
<cacheDir>/<sanitizedRepoId>/
  train/
  inference/
  embedding/
  mobiletransformers_manifest.json
  checksums.json
```

The Android downloader can either copy files from `default/`/`variants/<id>/` into that shape or download directly into that shape using manifest path mappings. This avoids forcing the existing repository classes to understand every HF repository layout detail.

The manifest should include:

- `schemaVersion`
- `baseModelId`
- `exportedAt`
- `mobiletransformersVersion`
- `artifactFormatVersion`
- `architectures`
- `taskTypes`
- `peftMethods`
- `quantization`
- `requiredFiles`
- `variants`
- `defaultVariant`
- `downloadPlan`
- `fileSizes`
- `sha256`
- `etag`
- `weightHandoff`
- `androidRuntime`
- `onnxRuntimeTrainingVersion`
- `onnxRuntimeGenAIVersion` if applicable
- `optimumOnnxVersion`
- `transformersVersion`
- `supportedTasks`
- `selectedTask`
- `trustRemoteCode`
- `minimumAndroidApi`
- `recommendedDeviceMemoryMb`
- `license`

The `downloadPlan` should group files by feature so the app can avoid unnecessary downloads:

```json
{
  "groups": {
    "core": ["mobiletransformers_manifest.json", "config.json", "shared/tokenizer/**"],
    "inference": ["variants/cpu-int4/inference/**"],
    "train": ["variants/cpu-int4/train/**"],
    "rag": ["variants/cpu-int4/embedding/**"],
    "genai": ["variants/cpu-int4/inference/genai_config.json"],
    "checksums": ["variants/cpu-int4/checksums.json"]
  }
}
```

The `weightHandoff` section should point at a package-local handoff map, usually `default/train/weight_handoff_map.json` or `variants/<id>/train/weight_handoff_map.json`. This map is the contract that connects training checkpoints, merger outputs, inference initializers, and optional GenAI graph inputs. Android should use it instead of hard-coded name replacement when loading merged weights or feeding GenAI model inputs.

Minimum handoff-map shape:

```json
{
  "schemaVersion": "1.0",
  "minReaderVersion": "1.0",
  "handoffMode": "external_initializer",
  "entries": [
    {
      "trainingBaseLayerName": "backbone.model.layers.0.self_attn.q_proj.base_layer",
      "checkpointNames": {
        "weight": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight"
      },
      "mergedTensorNames": {
        "weight": "model.layers.0.attn.q_proj.MatMul.weight"
      },
      "inferenceInitializerNames": {
        "weight": "model.layers.0.attn.q_proj.MatMul.weight"
      },
      "genaiInputNames": {},
      "dtype": "float16",
      "shape": [4096, 4096],
      "transposePolicy": "already_transposed_for_inference"
    }
  ]
}
```

For `handoffMode = "model_input"`, `genaiInputNames` must be populated and the corresponding tensors must exist as ONNX graph inputs. For `handoffMode = "external_initializer"`, `inferenceInitializerNames` must exist as external-data initializers in the inference ONNX graph.

Use HuggingFace Hub repo IDs for ready packages, for example:

- `mobiletransformers/Qwen2-0.5B-mobile`
- `mobiletransformers/SmolLM2-360M-mobile`
- `mobiletransformers/TinyLlama-1.1B-mobile`
- `mobiletransformers/all-MiniLM-L6-v2-embedding-mobile`

## Hub Pull And Cache Flow

Desktop/Python pull should use `huggingface_hub.snapshot_download` with `allow_patterns` derived from the manifest. This is the most robust path for CLI users because the Python library already handles revisions, cache metadata, ETags, and filtered downloads.

Android pull should be a small native/Kotlin Hub client:

1. Resolve `repoId`, optional `revision`, and auth token.
2. Download only `mobiletransformers_manifest.json` first through the Hub resolve URL pattern.
3. Parse variants and choose a default from device/runtime constraints: ABI, preferred quantization, available storage, estimated memory, GenAI/manual inference choice, and whether training/RAG are requested.
4. Build a file list from `downloadPlan`.
5. Use HEAD metadata when available to show size and validate ETag/commit information before downloading large files.
6. Download with OkHttp or a similar Android HTTP client through WorkManager, using a `.partial` staging directory.
7. Support pause/resume with HTTP range requests where the Hub/CDN response permits it.
8. Validate SHA256 checksums and manifest schema before exposing the package to `LLMRepository`.
9. Atomically rename the staged package into `<cacheDir>/<sanitizedRepoId>/`.
10. Register the package through the same local cache layout currently used by `LLMRepository`.
11. Load configs through existing parsers.
12. Return a `MobileTransformerModel` facade instance.

Android direct Hub download is useful and should be part of the product API, but it should be implemented as a careful downloader rather than a full clone of the Python SDK. If Android implementation slips, ship the first release with Python-side `mobiletransformers pull` or `mobiletransformers install-package` that prepares the Android cache directory, then add direct device download as the next milestone.

Minimum Android API shape:

```kotlin
MobileTransformers.fromPretrained(
    context = context,
    repoId = "mobiletransformers/Qwen2-0.5B-mobile",
    revision = "main",
    variant = "cpu-int4",
    features = setOf(ModelFeature.Inference, ModelFeature.Training),
    cacheDir = context.filesDir
)
```

## MobileTransformersApp Improvements

The renamed `MobileTransformersApp` should become a proper SDK demonstration app rather than an internal test harness.

Recommended app changes:

- Rename app title and navigation labels from ORT/ORTransformers wording to MobileTransformers.
- Replace direct repository construction in `MainActivity` and view models with the `MobileTransformers.fromPretrained` facade.
- Add a model-package selection/validation flow that explains missing `train/`, `inference/`, or `embedding/` assets clearly.
- Keep three main workflows visible: train, generate, and RAG. Do not turn the app into a marketing landing page.
- Add a package cache screen that shows installed MobileTransformers-ready packages, manifest metadata, and storage usage.
- Add a simple adapter export/share action only after the adapter packaging path is defined.
- Keep advanced engine/debug controls behind a developer settings screen so the sample app demonstrates the public API first.

## Adapter Push-Back Plan

Do adapter upload in Python first:

1. After on-device training, export adapter tensors and metadata from the Android cache.
2. Copy or sync the adapter package to a desktop/dev environment.
3. Convert to PEFT-compatible adapter layout where possible.
4. Generate a model card that declares base model, PEFT type, rank, MARS optimization level, dataset notes, and privacy warning.
5. Upload with Hugging Face Hub APIs.

Android direct upload should be optional and gated behind auth/security review.

## One-Command Export Plan

Create a CLI that wraps current builder stages:

```bash
mobiletransformers export \
  --model Qwen/Qwen2-0.5B \
  --task text-generation \
  --peft mars-opt1 \
  --rank 8 \
  --quant qint8 \
  --output build/packages/qwen2-0.5b-mobile \
  --include-rag false
```

The command should:

1. Load `config/config.yml` defaults.
2. Resolve `Settings` from env.
3. Query Optimum ONNX `TasksManager` and select or validate the requested ONNX task.
4. Export inference ONNX with Optimum ONNX or the fallback direct exporter.
5. Normalize ONNX filenames, external-data layout, KV names, and trainable-weight hooks.
6. Export training ONNX or derive the training graph required by ORT training artifacts.
7. Generate ORT training artifacts with the source-built ORT training wheel.
8. Export tokenizer config, chat template, generation config, and optional GenAI config.
9. Optionally export embedding model/RAG config.
10. Emit `mobiletransformers_manifest.json`, `checksums.json`, `optimum/export_report.json`, and `optimum/supported_tasks.json`.
11. Run a desktop smoke if `--validate` is set.

Add a publish command after the export format stabilizes:

```bash
mobiletransformers push \
  --package build/packages/qwen2-0.5b-mobile \
  --repo-id mobiletransformers/Qwen2-0.5B-mobile \
  --private false
```

The push command should use `huggingface_hub.upload_folder`, generate or update the model card, and upload packages under stable variant paths. The model card must list base model, license, training support, inference support, RAG support, Android runtime requirements, ORT/GenAI versions, and known device limits.

## Implementation Sequence

Prerequisites before Tier 1 starts:

- Tier 0 has selected or narrowed the export/toolchain path.
- `mobiletransformers_manifest.json` has a validator and tiny fixture.
- Android cache mapping is stable enough to materialize `train/`, `inference/`, and `embedding/`.
- The Android rename/facade foundation from `00_repository_restructure_plan.md` is either complete or being implemented as the first Tier 1 step.

Sequence:

1. Finalize manifest schema, variant schema, checksum schema, feature groups, and validator in Python.
2. Add package path mapping from HF repo variant paths into the existing Android `train/`, `inference/`, and `embedding/` cache layout.
3. Rename Android Gradle root/module/app to `MobileTransformers`, `:MobileTransformers`, and `:MobileTransformersApp`.
4. Update Android namespaces, source directories, sample app `applicationId`, Gradle dependencies, build scripts, and visible app labels to MobileTransformers names.
5. Add `MobileTransformers` Kotlin facade and repository-backed runtime without removing repositories.
6. Add deprecated compatibility aliases/wrappers for old `com.martinkorelic.ortmobile` imports if the package move would otherwise break users.
7. Add Android model package validator that checks manifest, variant, checksums, feature groups, and expected cache layout before load.
8. Add Optimum ONNX support-discovery wrapper and `model_support_matrix.json` generation, reusing Tier 0's toolchain decision.
9. Add CLI wrapper around current export/artifact builder and emit package manifests from one command.
10. Add Python `pull`/`install-package` that downloads a selected variant and prepares an Android cache directory.
11. Move sample app usage to the facade with a local fixture package.
12. Add Android manifest-first Hub downloader behind `fromPretrained`, using WorkManager and an atomic staging directory.
13. Add starter package generation for one smallest Optimum-supported text-generation model first.
14. Add `PUBLIC_API.md`, `MODEL_FORMAT.md`, `HUB_PACKAGE_FORMAT.md`, and `ANDROID_CACHE_FORMAT.md`.
15. Add Hub upload for starter packages.
16. Add adapter upload only after adapter package semantics, trainable tensor ordering, and privacy warnings are settled.

## Risks

- Direct Android Hub download may be too heavy for v1 due to large files, auth, and retry behavior.
- Android rename churn can break Gradle module paths, app IDs, JNI load names, ObjectBox generated code, and sample app imports.
- The model package format may need separate paths for GenAI and manual inference if Tier 0 chooses manual fallback.
- Optimum ONNX may export a model that still fails MobileTransformers training, merge, or Android memory gates.
- Variant selection can become confusing if the manifest does not clearly distinguish `default`, CPU/mobile int4, desktop smoke, GenAI, manual inference, and RAG-capable variants.
- Android direct download must handle partial files, checksum failures, storage pressure, auth tokens, and user cancellation without corrupting the active cache.
- Hub repo layouts for existing ONNX packages are not standardized enough to consume directly; MobileTransformers needs its own manifest instead of assuming `onnx/` or `cpu_and_mobile/` means ready.
- MARS adapter push-back may not map cleanly to standard PEFT without a custom metadata extension.
- A facade can hide engine errors unless validation messages are careful.
- Starter model zoo hosting requires license checks for each base model.

## Tests And Smokes

- Manifest unit tests for missing required files.
- Manifest unit tests for variant selection, file groups, file sizes, SHA256, optional GenAI files, and optional RAG files.
- Handoff-map unit tests for training checkpoint names, merged tensor names, inference initializer names, GenAI graph input names, dtype, shape, quantization metadata, and transpose policy.
- CLI dry-run test from `config/config.yml`.
- Optimum ONNX support-discovery test for supported and unsupported model types.
- Export smoke for one tiny model or fixture.
- Package smoke that emits `mobiletransformers_manifest.json`, `checksums.json`, `optimum/export_report.json`, and `optimum/supported_tasks.json`.
- Python Hub pull smoke using `snapshot_download(..., allow_patterns=...)` into a local cache directory.
- Android downloader unit/instrumentation smoke with a tiny local HTTP fixture or mock server: manifest first, selected variant only, checksum validation, partial download cleanup, and atomic rename.
- Android package-load smoke using a local prepared package.
- Gradle rename smoke: `./gradlew :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug`.
- Android namespace smoke: compile a tiny consumer that imports `com.martinkorelic.mobiletransformers.MobileTransformers`.
- Compatibility smoke for old `com.martinkorelic.ortmobile` imports if wrappers/typealiases are kept.
- Facade smoke: `fromPretrained` -> `train(maxSteps = 1)` -> `merge()` -> `generate(maxSequenceLength = 1)`.
- Sample app regression: old repository usage still works until public migration is complete.
- Hub upload dry-run with generated model card.

## Acceptance Criteria

- A documented HuggingFace-style Android API exists.
- Android SDK module is renamed to `MobileTransformers`.
- Android sample app module is renamed to `MobileTransformersApp`.
- Public Android APIs live under `com.martinkorelic.mobiletransformers`, with legacy `ortmobile` compatibility either provided or explicitly scheduled for removal before `v1.0`.
- A MobileTransformers-ready model package schema exists and is validated.
- The package schema includes variants, download groups, checksums, Optimum export metadata, and Android cache path mapping.
- The package schema includes a validated weight handoff map for train-to-infer tensor transfer.
- A support matrix distinguishes Optimum-exportable, package-exportable, training-ready, Android-inference-ready, Android-training-ready, and RAG-ready states.
- One CLI command can build a ready package from a supported HF model.
- A Python pull/install command can download a selected package variant into the existing Android cache layout.
- Android direct Hub pull is either implemented with manifest-first staged downloads or explicitly deferred behind the Python pull/install path.
- At least one starter model package can be loaded by the Android library.
- Existing repository classes remain usable for compatibility.
- Docs explain how to bring a dataset and avoid manual export work.

## Source Links

- Hugging Face Hub download guide: https://huggingface.co/docs/huggingface_hub/guides/download
- Hugging Face Hub file download API: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- Hugging Face Hub upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload
- Hugging Face model upload guide: https://huggingface.co/docs/hub/models-uploading
- Android WorkManager: https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started/define-work
- Optimum ONNX PyPI: https://pypi.org/project/optimum-onnx/
- Optimum ONNX overview and supported architectures: https://huggingface.co/docs/optimum-onnx/onnx/overview
- Optimum ONNX export guide and TasksManager: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
- ONNX community Llama ONNX package example: https://huggingface.co/onnx-community/Llama-3.2-1B
- Microsoft Phi-3 ONNX package example: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
- ONNX Runtime GenAI: https://onnxruntime.ai/docs/genai/
- MobileFineTuner: https://arxiv.org/abs/2512.08211

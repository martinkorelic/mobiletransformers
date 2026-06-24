# Tier 0 - Foundation And Decisions

## Purpose

Tier 0 decides the architecture before implementation work fans out. It covers the GenAI inference decision, the memory-mapped or low-copy train-to-infer handoff, the Optimum/ONNX Runtime Training survivability problem, dependency/install hygiene, licensing, and the repository restructure gate.

This revision is more concrete than the first draft: ONNX Runtime Training is treated as a source-built local artifact, not as a normal public PyPI dependency, and ONNX Runtime GenAI is treated as feasible enough for a serious spike because the repo already contains several useful integration hooks.

> Detailed code-implementation plans live in `agent_docs/01_code_plans/` and the global order in `agent_docs/IMPLEMENTATION_ORDER.md`.

## Current Repo Evidence

- The active Android generation path is `ORTGeneratorNative`, backed by custom C++ inference and sampling code.
- `ORTGenAINative` is marked deprecated and its methods throw `NotImplementedError`, while `android/ORTransformer/ORTransformersMobile/src/main/cpp/onnx-genai.cpp` is fully commented out. There is therefore NO working GenAI path on device today; GenAI is a spike target, not an existing capability.
- The Android library still links `onnxruntime-genai` in `CMakeLists.txt` and includes a local `onnxruntime-genai.aar`, so the binary/build foothold exists even though the wrapper is abandoned.
- The vendored GenAI C header at `android/ORTransformer/ORTransformersMobile/src/main/cpp/onnxruntime-genai/ort_genai_c.h` exposes `OgaGeneratorParamsSetModelInput`, documented locally as the hook for additional model inputs such as LoRA fine-tuning values.
- The vendored local C++ wrapper exposes `OgaGeneratorParams::SetInputs(OgaNamedTensors&)`, but `OgaNamedTensors` itself has no local `Create`/`Set`/`Get` wrapper methods. In this checked-in header set, `OgaNamedTensors` is mainly returned by multimodal processors, so it is not yet a convenient user-built container for trained tensors.
- The same local header also exposes `OgaCreateModelWithInitializers(const char* config_path, OgaModel** out, const std::unordered_map<std::string, OrtValue>& initializers)`. This is verified NON-UPSTREAM: it does not exist in upstream `onnxruntime-genai`, it appears only in this repo's vendored header, and it uses C++ types in a C-style header. Treat it as fork-only and do NOT depend on it for v1.
- `inference/generator_genai.py` already prototypes a GenAI path that loads trainable initializers from an ONNX model and calls `params.set_model_input(initializer.name, W)`.
- `session_cache.h` already has a `WeightSessionCache` that loads serialized `.tensor` trainable weights into named `Ort::Value` objects for transfer into inference.
- `session_cache.h` adds those merged weights to ORT with `SessionOptions.AddExternalInitializers(...)` when `load_external_weights` is enabled.
- `weight_serializer.cpp` allocates memory through `Ort::AllocatorWithDefaultOptions`, copies raw tensor bytes into that allocation, and wraps the result as an `Ort::Value`. This is useful low-copy session injection, but it is not file-backed `mmap`. There is NO `mmap` anywhere in the repo today: every weight path copies bytes into allocator buffers, so all `mmap` language in this doc is aspirational and gated behind the experiments below.
- The current inference session options explicitly set `session.use_ort_model_bytes_for_initializers` to `0`, even though the vendored ORT header documents `session.use_ort_model_bytes_directly` and `session.use_ort_model_bytes_for_initializers` for ORT-format model byte reuse.
- `ORTGenerationConfig` includes `loadMergedWeights`, and `ORTGeneratorNative` checks for an `inference/merged` directory before loading a session.
- Root `config.yml` already has relevant switches, and the current defaults are `weight_input: false`, `export_genai_config: false`, `force_external_initializers: true`, and `loadMergedWeights: false`. Any plan that depends on weight-input or GenAI-config export must flip these explicitly rather than assume they are on.
- `inference/builder.py` can emit `genai_config.json`; it already fills decoder `session_options`, input/output names, KV-cache names, search parameters, and `past_present_share_buffer`, but it does not currently emit ORT memory-related `config_entries`.
- Training artifacts are generated with `onnxruntime.training.artifacts.generate_artifacts` in `artifact/onnx_builder.py`.
- The training export path imports `optimum.exporters.onnx` and model-specific ONNX configs in `trainer/builder.py`.
- `requirements-ort.txt` lists `onnxruntime-training==1.23.0+cpu` and `optimum==1.23.3`. The `1.23.0+cpu` build should be documented as a local/source-built wheel rather than a public package expectation.
- Public PyPI still shows `onnxruntime-training` latest as `1.19.2`, and Maven Central still shows `onnxruntime-training-android` `1.19.2`; this does not invalidate the local `1.23.0+cpu` workflow, but it means reproducibility depends on source-build documentation.
- The repo has root `config.yml` and `config.py`, but no `pyproject.toml`, no CI folder, no changelog, and no tags.
- `LICENSE.md` is currently CC-BY-NC-4.0, which blocks many commercial and ecosystem adoption paths. The decision is to relicense to Apache-2.0 before `v1.0`; base-model weights keep their upstream licenses and are out of scope for this relicense.

## External Research Summary

- ONNX Runtime GenAI provides the generative loop around ONNX Runtime, including tokenization, search/sampling, generation parameters, tensor helpers, and KV-cache-oriented generation APIs. Source: https://onnxruntime.ai/docs/genai/
- The public GenAI C and Java docs mark the APIs as preview and subject to change. Source: https://onnxruntime.ai/docs/genai/api/c.html
- Public GenAI C docs show `OgaCreateModel` loading a model from a configuration directory, while the checked-in local header adds `OgaCreateModelWithInitializers`. Source: https://onnxruntime.ai/docs/genai/api/c.html
- GenAI config is driven by a `genai_config.json` model directory that names the decoder model, inputs, outputs, provider/session options, KV-cache naming, and search defaults. Source: https://onnxruntime.ai/docs/genai/reference/config.html
- GenAI search config notes that `max_length` determines KV-cache memory allocation, so mobile packages must tune this deliberately rather than inheriting large desktop defaults. Source: https://onnxruntime.ai/docs/genai/reference/config.html
- GenAI config supports `session_options.config_entries`, so ORT session config keys can be tested through `genai_config.json`; however, the docs do not claim that GenAI can memory-map runtime-supplied trained weights by config alone. Source: https://onnxruntime.ai/docs/genai/reference/config.html
- GenAI's public config and runtime APIs expose model creation from a config directory, `OgaConfig` creation/overlays, provider selection, runtime settings handles, and Python `Config` provider helpers, but they do not expose a public raw `Ort::SessionOptions` object or public `AddExternalInitializers` equivalent. Sources: https://onnxruntime.ai/docs/genai/api/c.html, https://onnxruntime.ai/docs/genai/api/python.html
- GenAI's Python `GeneratorParams.set_model_input` and local C `OgaGeneratorParamsSetModelInput` are the public/available hooks for "extra model inputs" that GenAI does not manage itself. This matches the current `inference/generator_genai.py` prototype. Sources: https://onnxruntime.ai/docs/genai/api/python.html, https://onnxruntime.ai/docs/genai/api/c.html
- GenAI's current C++ docs expose `OgaNamedTensors::Create`, `Get`, `Set`, `Delete`, `Count`, and `GetNames`, plus `SetInputs`; however, current upstream source treats these as named graph inputs passed into the generator, not as initializer mutation. Source: https://onnxruntime.ai/docs/genai/api/cpp.html#oganamedtensors
- Upstream source currently wires `OgaNamedTensorsSet` into a `NamedTensors` map, `OgaGenerator_SetInputs` into `Generator::SetInputs`, and `OgaGenerator_SetModelInput` into the generator's `extra_inputs_` list. Source: https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp
- Upstream `Generator::SetInputs` currently rejects LLM and pipeline model types and directs callers toward token append flow; for decoder-only LLM trained-tensor handoff, `SetModelInput` is the safer hook than `SetInputs(OgaNamedTensors)`. Source: https://github.com/microsoft/onnxruntime-genai/blob/main/src/generators.cpp
- Upstream `ExtraInputs::Add` only appends user tensors whose names match required session input names, then `State::Run` forwards `input_names_` and `inputs_` to the underlying ORT session run. This confirms that `NamedTensors`/`SetModelInput` can feed additional ONNX graph inputs, but does not replace already-loaded model initializers. Sources: https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/extra_inputs.cpp, https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.cpp
- GenAI's past-present share buffer is an explicit KV-cache memory optimization; it shares past and present KV buffers when enabled and avoids per-token present-buffer reallocation/copy. Source: https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html
- GenAI has a `.onnx_adapter` flatbuffer adapter format containing named parameters with dimensions, type, and data bytes. Source: https://onnxruntime.ai/docs/genai/reference/adapter.html
- GenAI's LoRA tutorial shows runtime adapter loading and switching through `og.Adapters`, `load`, and `set_active_adapter`. Source: https://onnxruntime.ai/docs/genai/tutorials/finetune.html
- ONNX Runtime documents on-device training as an offline Python artifact-generation phase followed by device training. Source: https://onnxruntime.ai/docs/get-started/training-on-device.html
- ONNX Runtime's core SessionOptions initializer injection API can replace matching model initializers before session creation. The injected tensor and its backing buffer must outlive the session. Source: https://raw.githubusercontent.com/microsoft/onnxruntime/main/include/onnxruntime/core/session/onnxruntime_c_api.h
- ONNX Runtime's training build docs explicitly support source builds with `--enable_training_apis`; Python artifact generation requires a built Python wheel via `--build_wheel`. Source: https://onnxruntime.ai/docs/build/training.html
- ONNX Runtime Android build docs generate AAR files when building with `--build_java`, which matches the repo's local AAR workflow. Source: https://onnxruntime.ai/docs/build/android.html
- Current ONNX Runtime headers on `main` include `session.use_memory_mapped_ort_model` for mapping ORT-format models and immutable weights from disk, while this repo's vendored headers only show older ORT-format byte-reuse keys and external-prepacked-initializer options. Source: https://raw.githubusercontent.com/microsoft/onnxruntime/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
- Hugging Face Optimum v2 moved ONNX export/runtime integrations into Optimum ONNX and officially deprecated ONNX Runtime Training. Source: https://github.com/huggingface/optimum/releases
- Optimum ONNX release notes state that ORT training was removed from the split-out package. Source: https://github.com/huggingface/optimum-onnx/releases
- `optimum-onnx` is now its own PyPI package (`0.1.0` as of the latest check) with extras for `onnxruntime` and `onnxruntime-gpu`; it exports Transformers, Diffusers, Timm, and Sentence Transformers models and provides ORTModel inference helpers. Source: https://pypi.org/project/optimum-onnx/
- Optimum ONNX's supported Transformers architecture list includes several families already in this repo's roadmap, including Llama, Mistral, Phi/Phi3, Qwen2/Qwen3, Gemma/Gemma 2, DeepSeek-V3, Granite, ModernBert, and SmolLM3. Source: https://huggingface.co/docs/optimum-onnx/onnx/overview
- Optimum ONNX exposes `TasksManager.get_supported_tasks_for_model_type(..., "onnx")` and custom `OnnxConfig` hooks, which can become MobileTransformers' model-support discovery layer instead of hard-coding every family by hand. Source: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
- `uv` supports dependency groups, optional dependencies, local path sources, lock/sync flows, requirements export, and SBOM export, which makes it the better fit for this repo's multi-profile Python dependency problem. Sources: https://docs.astral.sh/uv/concepts/projects/dependencies/, https://docs.astral.sh/uv/concepts/projects/sync/, https://docs.astral.sh/uv/concepts/projects/export/
- MobileFineTuner is a direct competitor focused on mobile on-device LLM fine-tuning. Its gap relative to this repo is the absence of a complete train-to-merge-to-infer-to-RAG system. Source: https://arxiv.org/abs/2512.08211

## Updated Feasibility Assessment

GenAI is feasible enough to move from "uncertain idea" to "ranked implementation spike".

The most important change is recognizing that the repo already has two possible train-to-GenAI handoff mechanisms:

1. `OgaGeneratorParamsSetModelInput` / Python `params.set_model_input`.
2. Local `OgaCreateModelWithInitializers`, if the linked GenAI library exports it and the source branch can be pinned.

The first mechanism is the safest near-term path because `inference/generator_genai.py` already demonstrates the concept: load trainable weights, match them by name, and pass them as model inputs while GenAI handles normal token inputs and generation. This requires the exported inference model to expose trainable weights as model inputs instead of relying only on immutable initializers. `OgaNamedTensors` should be considered a useful container only after the GenAI AAR/header version is upgraded and pinned; the current local headers do not expose manual construction/setters, and upstream `SetInputs` is not the right decoder-only LLM path.

The second mechanism is potentially cleaner because it maps directly onto the existing `WeightSessionCache` idea: build an `unordered_map<string, OrtValue>` for updated trainable weights and create the GenAI model with those initializer overrides. It is riskier because the public docs do not show this API, the C++ wrapper does not currently expose it, and the Android AAR/shared library must be checked for the exported symbol.

Adapter files are promising for LoRA-style personalization and Hub publication, but they should not be assumed to solve MARS immediately. The `.onnx_adapter` format stores named tensors; the open question is whether MARS updates can be represented as adapter parameters without losing the repo's merge semantics or requiring a new graph export after every on-device train cycle.

## Optimum ONNX Support Alignment

Optimum ONNX should become the default support-discovery and inference-export front door, but not the only definition of MobileTransformers support.

The important distinction:

- Optimum ONNX support means a model family/task can be exported to ONNX through Hugging Face tooling.
- MobileTransformers export support means that exported graph can be normalized into this repo's package format, tokenizer metadata, generation config, optional GenAI config, and Android cache layout.
- MobileTransformers training support means ORT training artifacts, PEFT/MARS target modules, merge artifacts, and Android runtime smokes all pass.

Recommended policy: when Optimum ONNX adds support for another model family, MobileTransformers should automatically detect it as a candidate and add it to a generated support matrix, but it should not advertise it as ready until the mobile package and train-to-infer gates pass.

Concrete implementation:

1. Add an export registry wrapper around `optimum.exporters.tasks.TasksManager`.
2. For a requested HF model, read `config.model_type`, query `TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")`, and choose the task in this order: `text-generation-with-past`, `text-generation`, `feature-extraction`, `sentence-similarity`, then explicit user override.
3. Run `optimum-cli export onnx` or `optimum.exporters.onnx.main_export` for inference export.
4. Normalize graph names, KV-cache names, tokenizer files, `generation_config.json`, and external-data layout into MobileTransformers package conventions.
5. Generate training artifacts only after the model's PEFT target modules and trainable parameter set are known.
6. Emit `model_support_matrix.json` with statuses such as `optimum_exportable`, `mobile_package_exportable`, `train_artifacts_exportable`, `android_inference_smoked`, `android_training_smoked`, and `rag_ready`.

This keeps the repo aligned with Optimum's expanding model coverage while protecting users from a false promise that every exported ONNX graph can be trained, merged, and run inside the Android SDK.

## Recommended Decisions

1. Use `uv` as the Tier 0 Python dependency manager, with `uv.lock` as the reproducibility source and exported requirements files as generated compatibility artifacts.
2. Treat `onnxruntime-training==1.23.0+cpu` as a first-class local/source-built artifact. Do not model it as a normal public PyPI dependency.
3. Make Tier 0.3 the first gate: prove the training export/toolchain path before large API work.
4. Reopen ONNX Runtime GenAI adoption with a ranked spike instead of keeping the old deprecated wrapper as evidence that GenAI is impossible.
5. Preserve the current train-to-merge-to-infer handoff as a core differentiator. GenAI is adopted only if it can consume the on-device trained result without forcing on-device graph rewriting.
6. Prefer GenAI model-input injection for the first proof, initializer injection for the second proof, and `.onnx_adapter` only after LoRA/MARS adapter semantics are explicitly validated.
7. Use Optimum ONNX as the primary inference-export and support-discovery layer. Treat new Optimum support as a candidate signal, then graduate models through MobileTransformers package, training, and Android gates.
8. Use a two-track export strategy: durable path is Optimum ONNX or direct `torch.onnx` for inference export, plus direct `onnxruntime.training` artifact APIs where still available. Legacy Optimum pinning is allowed only as a documented stopgap.
9. Introduce packaging and config scaffolding early, but do not move implementation modules until toolchain and GenAI decisions are known.
10. Escalate the license decision to authors before `v1.0`. Keep the docs neutral: CC-BY-NC-4.0 is compatible with research sharing but not with becoming a broadly adopted Android/HF framework.

## GenAI Handoff Options

| Rank | Option | Feasibility | Recommended use |
| --- | --- | --- | --- |
| 1 | Model-input injection with `OgaGeneratorParamsSetModelInput` / `params.set_model_input` / version-appropriate `SetModelInput` | High for a spike, because local Python already prototypes it and the local C header documents it | First GenAI proof. Export inference model with trainable weights as additional graph inputs, then feed updated weights at generation time. |
| 2 | Initializer injection with local `OgaCreateModelWithInitializers` | Medium, pending symbol/export verification and wrapper work | Second proof. Reuse `WeightSessionCache` to pass named `Ort::Value` overrides when constructing the GenAI model. |
| 3 | `OgaNamedTensors` / `SetInputs` | Medium for non-LLM and multimodal inputs, low as the primary decoder-only LLM handoff | Use only after upgrading/pinning GenAI and proving `SetInputs` accepts the target model type. Upstream source currently rejects LLM/pipeline model types on this path. |
| 4 | `.onnx_adapter` files | Medium for LoRA, unknown for MARS | Long-term adapter/publication path if adapter semantics match the PEFT method. |
| 5 | On-device graph rewrite or full model export | Low | Avoid for v1. This undermines the current train-to-infer differentiator and likely increases latency/storage. |

## Memory-Mapping And Low-Copy Position

The release goal should be phrased carefully: "no on-device graph export and bounded-copy weight handoff" is immediately more defensible than "true zero-copy memory mapping".

The current `WeightSessionCache` loads serialized `.tensor` files into allocated buffers and wraps them as `Ort::Value`. `weight_serializer.cpp` then copies tensor payloads from protobuf/raw tensor storage into allocator-owned buffers. That is good enough for a first GenAI handoff proof, but it is not yet proof of file-backed memory mapping.

Verified position after reviewing local code and current docs:

- Manual ORT inference already supports updated weights through `AddExternalInitializers`, but those values are allocated/copied tensors, not `mmap` views.
- GenAI config exposes decoder `session_options` and `config_entries`, so ORT config entries can be passed through the package and tested.
- GenAI search config and `past_present_share_buffer` control KV-cache allocation behavior. They do not solve model-weight or merged-weight memory mapping.
- Current upstream ORT headers mention `session.use_memory_mapped_ort_model` for ORT-format models and immutable weights, but this repo's vendored headers do not show that key. Local headers instead expose `session.use_ort_model_bytes_directly`, `session.use_ort_model_bytes_for_initializers`, `session.model_external_initializers_file_folder_path`, and prepacked constant initializer externalization.
- GenAI's public/local APIs expose `OgaCreateModel`, version-appropriate `SetModelInput`, `OgaCreateTensorFromBuffer`, and adapter APIs; they do not expose the local `OgaCreateModelWithInitializers` function as a stable documented path.
- `OgaCreateTensorFromBuffer` and ORT `CreateTensor` can wrap user-owned buffers, but that only becomes a true mmap path if the user-owned buffer is file-backed, its lifetime is tied to the model/generator, and measurement proves GenAI/ORT do not immediately copy it internally.

A true memory-mapped path would require:

- File-backed buffers for merged weights, likely through `mmap` on Android app-internal storage.
- `Ort::Value` or `OgaTensor` creation from buffers whose lifetime is explicitly tied to the GenAI model/generator lifecycle.
- Validation that GenAI does not copy the full tensor set internally in the chosen API path.
- RSS/heap measurements before model load, after weight load, after first token, and after cache release.

Recommended experiments:

1. Base-model mmap experiment: export a tiny ORT-format inference model, add `session.use_ort_model_bytes_directly=1`, `session.use_ort_model_bytes_for_initializers=1`, and, if the ORT/GenAI build supports it, `session.use_memory_mapped_ort_model=1` through GenAI `session_options.config_entries`. Verify whether GenAI forwards the entries and whether RSS improves.
2. Manual-ORT merged-weight mmap experiment: replace the current copied `.tensor` load with file-backed buffers and `Ort::Value::CreateTensor`, then compare against `AddExternalInitializers` from copied buffers.
3. GenAI model-input mmap experiment: create `OgaTensor` values from file-backed merged-weight buffers and feed them through the pinned GenAI `SetModelInput` API.
4. GenAI initializer-injection experiment: only if the Android GenAI library exports the local `OgaCreateModelWithInitializers`, pass file-backed `OrtValue` overrides at model construction and compare memory/latency.
5. Adapter experiment: serialize LoRA-style weights into `.onnx_adapter` and test whether adapter loading copies the full adapter payload or leaves it bounded enough for target devices.

Recommended wording for Tier 0 implementation: first prove "no graph rewrite, no full model duplicate on disk"; then benchmark whether actual mmap reduces heap pressure enough to become a v1 requirement. Do not advertise GenAI as the memory-mapping solution until one of the experiments above passes on Android.

## GenAI Session Options And Initializer Handoff Revision

The current finding is:

- GenAI does expose ONNX Runtime session configuration through `genai_config.json` under `model.decoder.session_options`.
- The supported `session_options` fields include threading, memory arena, memory pattern, profiling, `use_env_allocators`, provider lists/options, graph optimization, and arbitrary `config_entries`.
- The public GenAI C API also exposes `OgaCreateConfig`, `OgaConfigOverlay`, `OgaCreateModelFromConfig`, and provider mutation helpers. This lets MobileTransformers build or overlay the same session options it writes to `genai_config.json`.
- GenAI's public Python API exposes `Config(config_path)`, provider helpers, and `Model(Config)`.
- GenAI does not publicly expose the raw `Ort::SessionOptions` handle used by the current native path, and the public docs do not show an `AddExternalInitializers` hook.
- The local vendored `ort_genai_c.h` declares `OgaCreateModelWithInitializers(const char* config_path, OgaModel** out, const std::unordered_map<std::string, OrtValue>& initializers)`, but this is not in the public docs and uses C++ types in a C-style header. Treat it as a local/private or preview hook until the actual Android GenAI library exports the symbol and the exact source revision is pinned.
- Current upstream `OgaNamedTensors` is a named tensor map that feeds graph inputs through `SetInputs`; it is not an initializer override mechanism. It also is not the preferred upstream decoder-only LLM path because `Generator::SetInputs` rejects LLM/pipeline model types.

Implication: there are two different concepts that should not be mixed:

1. ORT session config through GenAI: good for provider selection, memory/threading/profile options, and ORT config entries.
2. Runtime trained-weight replacement: not solved by GenAI session config alone.

Recommended handoff order:

| Rank | Handoff | Export requirement | Why |
| --- | --- | --- | --- |
| 1 | GenAI model-input injection | The inference ONNX graph exposes each trained/merged tensor as a normal input. | Public API path through `GeneratorParams.set_model_input` / local `OgaGeneratorParamsSetModelInput` / upstream `OgaGenerator::SetModelInput`; already prototyped locally. |
| 2 | GenAI local initializer injection | The inference ONNX graph keeps trainable tensors as initializers, and the local GenAI library exports `OgaCreateModelWithInitializers`. | Closest to current `SessionOptions.AddExternalInitializers`, but not public/stable yet. |
| 3 | Manual ORT `AddExternalInitializers` | The inference ONNX graph contains external-data initializers with exactly matching names. | Current Android native path; keep as fallback and reference implementation. |
| 4 | GenAI `OgaNamedTensors` / `SetInputs` | The GenAI version exposes manual named-tensor creation and the target model type accepts `SetInputs`. | Helpful for multimodal/non-LLM inputs, but upstream decoder-only LLM code currently rejects this route. |
| 5 | GenAI `.onnx_adapter` | Adapter tensors can be serialized with names/shapes/dtypes matching GenAI adapter semantics. | Promising for LoRA publication, uncertain for MARS merge semantics. |

`OgaNamedTensors` is therefore not a substitute for initializer replacement. It can carry named tensors into a run only if those names are actual model/session inputs. The export still has to remove trainable initializers from the inference graph and re-add them as inputs, or use an initializer-injection path before session creation.

For GenAI, `session_options.config_entries` should still be emitted and tested. Recommended entries to test are the same class of entries used by the manual native path:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "log_id": "onnxruntime-genai",
        "use_env_allocators": true,
        "config_entries": [
          ["session.dynamic_block_base", "2"],
          ["session.use_device_allocator_for_initializers", "1"],
          ["session.use_env_allocators", "1"],
          ["session.intra_op.allow_spinning", "0"],
          ["session.qdq_matmulnbits_accuracy_level", "4"],
          ["session.use_ort_model_bytes_for_initializers", "0"],
          ["session.qdqisint8allowed", "1"]
        ],
        "provider_options": []
      }
    }
  }
}
```

This config should be treated as a performance/memory/session-behavior experiment, not a trained-weight injection mechanism.
In particular, ORT-format memory-map keys can reduce base-model load copies for immutable model bytes, but they do not replace the separate trained-weight handoff contract for tensors produced by on-device training.

## Inference Export Script Revision

The current code has two partially overlapping inference export paths:

- `inference/builder.py` builds GenAI-style ONNX graphs, emits `genai_config.json`, saves all initializers into one external `.data` file, and can already carry `session_options`.
- `artifact/onnx_builder.py::gen_genai(...)` has older logic that, when `weight_input=true`, removes trainable initializers from the graph and re-adds them as graph inputs so GenAI can receive them via `set_model_input`.
- `artifact/onnx_builder.py::gen_genai(...)` also forces external initializers when training config exists, which is required by the manual ORT initializer-injection path.
- Android manual inference currently loads merged tensors from `inference/merged` and injects them with `SessionOptions.AddExternalInitializers(...)`.

The export plan should unify these paths instead of keeping two separate inference model builders.

Required export changes:

1. Add a first-class `trainable_tensor_map.json` or `weight_handoff_map.json` emitted by the package builder.
2. For every trainable base layer in `training_config.json` / `peft_mapping`, record:
   - training/checkpoint parameter names,
   - merger input names,
   - merged output names,
   - inference initializer name,
   - optional GenAI model-input name,
   - dtype,
   - shape,
   - quantization fields if present,
   - transpose policy.
3. Make `inference/builder.py` own GenAI config and model export, but port the older `weight_input=true` graph-input conversion from `artifact/onnx_builder.py`.
4. Keep `force_external_initializers=true` for the manual ORT path and for validation. ORT initializer injection only works when the model still has matching external-data initializers.
5. Add a selectable export mode:
   - `handoff_mode = external_initializer`: keep trained tensors as external initializers for manual ORT and possible local GenAI initializer injection.
   - `handoff_mode = model_input`: remove selected trainable initializers and expose them as graph inputs for GenAI `set_model_input` / `SetModelInput`. For decoder-only LLMs, do not depend on upstream `SetInputs(OgaNamedTensors)` unless the pinned GenAI version proves it is supported for that model type.
   - `handoff_mode = adapter`: serialize LoRA-style updates into `.onnx_adapter` after adapter semantics are verified.
6. Do not infer tensor names on Android with ad-hoc string replacements. Android should read the exported handoff map and fail closed if any merged tensor does not match an inference initializer or GenAI model input.
7. Validate the current Android naming logic. `WeightMerger::inference_name(...)` removes `backbone.`, changes `self_attn` to `attn`, and changes `base_layer` to `MatMul`; Python validation also replaces `backbone.model` with `model`. These rules must become data in the exported map.
8. Investigate the current quantized save-name inconsistency: quantized weight and zero point use `safe_name`, while scale is saved as `base_layer_name + ".weight_scale"`. Keep it only if the exported inference model actually uses that scale name; otherwise fix the save name and add a regression test.

The most important invariant is: the name in `inference/merged/*.tensor` must exactly equal a name that the inference model can consume, either as an initializer override or as a graph input. Shape, dtype, transposition, and quantization parameters must match too.
Base-model mmap and external-data layout should be validated independently from this invariant; they optimize how immutable model files are loaded, while the handoff map proves that updated tensors land in the correct inference slots.

## Source-Built ORT Training Strategy

The local ORT training package should become a documented artifact pipeline:

- Add `scripts/build_ort_training_wheel.sh` for Python with `--enable_training_apis` and `--build_wheel`.
- Add `scripts/build_ort_training_android.sh` for Android with the same ORT commit/tag, Android NDK version, ABI list, Android API level, build type, and `--build_java`.
- Record ORT source revision, build commands, compiler/NDK versions, produced wheel/AAR filenames, and SHA256 checksums in `third_party/onnxruntime/manifest.json`.
- Document manual rebuild steps and known-good versions in `third_party/onnxruntime/BUILD.md`.
- Keep built wheels, AARs, extracted headers, and native libraries out of tracked source unless a separate release artifact policy is chosen.
- Declare the local Python wheel in `pyproject.toml` through `tool.uv.sources` and a local-only `ort-training-local` dependency group.
- Add a CI/manual job that verifies the local wheel can run `onnxruntime.training.artifacts.generate_artifacts` on a tiny model fixture.

This makes the repo sustainable even if public PyPI/Maven training artifacts lag behind the ORT source build used by the project.

## Decision Gates

### Gate 0.1 - GenAI Re-Evaluation

Adopt GenAI only if all are true:

- A GenAI model can load from a MobileTransformers-ready package and run on Android.
- The package builder can emit a valid `genai_config.json` from existing inference export metadata.
- The generation path can consume weights trained on device through model inputs, initializer overrides, or adapter files.
- It does not force on-device model export or graph rewriting after training.
- It supports the target sampling behavior and streaming callback shape.
- Build size and ABI story are acceptable for AAR distribution.
- Memory use is no worse than the manual loop by more than an explicitly accepted threshold.

Keep the manual loop if any of the following are true:

- GenAI cannot reflect newly trained weights without on-device graph rewrite.
- GenAI copies the full model or trainable tensor set in a way that breaks target-device memory budgets.
- Android packaging or Java/C++ APIs remain preview-only in a way that would destabilize `v1.0`.
- The model-input/initializer-injection path cannot preserve current MARS/merge semantics.

### Gate 0.2 - Memory-Mapping And Weight Handoff

The release must document one supported handoff:

- Existing path: training checkpoint -> on-device merger -> serialized merged weights under `inference/merged` -> native inference session with `loadMergedWeights`.
- GenAI path A: training checkpoint -> merger -> trainable-weight tensors -> GenAI `SetModelInput`.
- GenAI path B: training checkpoint -> merger -> `WeightSessionCache` -> GenAI `CreateModelWithInitializers`.
- GenAI path C: training checkpoint -> adapter serialization -> GenAI `.onnx_adapter` load/switch.

No Tier 2 inference refactor starts until one path is proven with one real tiny model package.

### Gate 0.3 - Training Export Toolchain

Choose one:

- Path A, preferred: `optimum-onnx` or direct `torch.onnx` for inference export, direct `onnxruntime.training.artifacts` for training artifacts, source-built ORT training wheel/AAR, and explicit pinned versions.
- Path B, temporary: legacy `optimum==1.23.x` environment for current graph export, with a migration issue and sunset deadline.
- Path C, high-control: own export layer over `torch.onnx` and ORT training artifacts, minimizing reliance on Optimum internals.

Path A is the default recommendation if direct ORT training artifact generation still works for the selected architectures.

Path A must include an Optimum alignment check:

- Query `TasksManager` for the HF model architecture and requested ONNX task.
- Record the exact Optimum ONNX version, Transformers version, ONNX opset, exporter mode, and whether `trust_remote_code` was required.
- Store the selected task and supported-task list in the package manifest.
- Fail closed if Optimum can export inference but MobileTransformers cannot generate matching training artifacts, tokenizer metadata, or Android runtime configs.
- Add a release-matrix entry instead of silently accepting or silently rejecting the model.

### Gate 0.4 - Packaging, Install, And License

Proceed to Tier 1 only when:

- The Python export environment is installable through `uv sync` from `pyproject.toml` and `uv.lock`.
- The local ORT training wheel build process is documented and reproducible.
- Android can build the library module with a documented local Maven/AAR path.
- The repository restructure plan is accepted.
- License direction is documented, even if the legal relicense happens later.

## Implementation Sequence

Follow the global order in `05_cross_cutting_release_modernization.md`. Tier 0 should answer architecture/toolchain questions before Tier 1 hardens the public API.

1. Create spike branches or issues for `0.3-toolchain`, `0.1-genai`, and `0.2-weight-handoff`.
2. Record baseline commands for the current export, artifact generation, Android build, one-step train, merge, and generation paths.
3. Add the minimal `uv` and config scaffolding from `00_repository_restructure_plan.md`, including `ort-training-local` as a local-only dependency group, but do not move large implementation modules yet.
4. Build a clean environment from the current pinned requirements and record the exact versions that actually resolve.
5. Document the current source-built ORT Training wheel/AAR provenance in `third_party/onnxruntime/BUILD.md` and `manifest.json`.
6. Verify that the source-built Python wheel can still import `onnxruntime.training.artifacts` and run `generate_artifacts` on a tiny fixture.
7. Try the current `trainer/builder.py` + `artifact/onnx_builder.py` flow on the smallest supported model and record failures before changing the exporter.
8. Add an Optimum ONNX support-discovery spike using `TasksManager`, then run `optimum-cli export onnx` or `main_export` on the same tiny model.
9. Decide the durable inference-export path: Optimum ONNX, direct `torch.onnx`, or temporary legacy Optimum with a sunset date.
10. Emit `model_support_matrix.json` with Optimum task support, package-export status, training-artifact status, Android smoke status, and blockers.
11. Define the minimal MobileTransformers package manifest fields needed by Tier 0 smokes: inference files, training files, tokenizer files, engine candidate, ORT/GenAI versions, checksums, cache mapping, and handoff mode.
12. Add `trainable_tensor_map.json` / `weight_handoff_map.json` generation that maps training checkpoint names to merger names, merged tensor names, inference initializer names, and optional GenAI model-input names.
13. Extend the Python inference builder to emit a minimal GenAI package directory with `genai_config.json`, decoder ONNX, tokenizer files, `session_options.config_entries`, and MobileTransformers manifest metadata.
14. Port the older `weight_input=true` graph-input conversion from `artifact/onnx_builder.py` into the chosen inference builder so GenAI model-input injection is generated by the same export path as `genai_config.json`.
15. Validate that each merged tensor in the handoff map exactly matches an inference initializer or GenAI graph input by name, dtype, shape, quantization metadata, and transpose policy.
16. Run the current `inference/generator_genai.py` desktop smoke with model-input injection from known trainable weights.
17. Add an Android JNI GenAI proof that loads the generated package and produces one token without custom sampling.
18. Implement model-input injection on Android using `OgaGeneratorParamsSetModelInput` and the exported handoff map instead of hard-coded name replacement.
19. Verify whether the linked GenAI library exports `OgaCreateModelWithInitializers` using `nm`, `readelf`, or the Android build equivalent. If present, add a narrow wrapper and compare memory/latency against model-input injection.
20. Test GenAI `session_options.config_entries` for ORT-format byte reuse and, if supported by the chosen ORT/GenAI build, `session.use_memory_mapped_ort_model`.
21. Replace one tiny merged-weight load with an `mmap`-backed buffer in the manual ORT path and compare against the current copied-buffer `AddExternalInitializers` path.
22. If upgrading GenAI, verify whether the Android AAR/header exposes `OgaCreateNamedTensors`, `OgaNamedTensorsSet`, and `OgaNamedTensorsGetNames`; if it does, keep `OgaNamedTensors` as a convenience wrapper, not as the primary LLM trained-weight handoff.
23. Spike `.onnx_adapter` conversion only after LoRA/MARS parameter naming and merge semantics are documented.
24. Measure memory for manual inference, manual mmap external initializer injection, GenAI model-input injection, GenAI initializer injection, and adapter load on the same tiny package.
25. Decide GenAI path or manual path and update Tier 2 docs before any broader inference refactor.
26. Export generated requirements/SBOM from `uv.lock` after the winning export strategy is known.

## Risks

- The local `OgaCreateModelWithInitializers` declaration may not be exported by the actual AAR/shared library, or it may be from a temporary branch.
- `OgaCreateModelWithInitializers` uses C++ types in a C-style header, which may create ABI and wrapper stability issues.
- Model-input injection may require exporting different ONNX graph shapes than the current manual loop.
- GenAI adapter support may map cleanly to LoRA but not to MARS merged-weight semantics.
- GenAI may internally copy tensors even when buffers are supplied by the app, weakening the memory story.
- GenAI may accept `session_options.config_entries` but ignore or reject ORT-format memory-map keys depending on the bundled ORT/GenAI build.
- ORT model memory mapping applies to immutable ORT-format model bytes and weights; it should not be confused with mapping mutable trained deltas after on-device training.
- Optimum ONNX may support exporting a model family before MobileTransformers can train, merge, quantize, or run it within Android memory budgets.
- `trust_remote_code=True` can be necessary for some Optimum exports, but it increases security/reproducibility review requirements for starter packages.
- `onnxruntime-training==1.23.0+cpu` is not a normal public dependency; onboarding fails unless source-build docs and local wheel paths are maintained.
- Optimum v2 removes the training wrapper path and may break imports used by `trainer/builder.py`.
- The manual inference loop has known maintenance burden but is currently aligned with the weight handoff.
- Licensing cannot be solved by engineering alone.

## Tests And Smokes

- `uv sync --frozen --group ort-training-local --extra train`
- `uv sync --frozen --extra export`
- Source-built ORT Python wheel smoke: import `onnxruntime.training.artifacts` and run `generate_artifacts` on a tiny fixture.
- Source-built ORT Android AAR smoke: compile the current library module before rename and `:MobileTransformers` after rename; verify headers/libs match the ORT manifest.
- Optimum ONNX support-discovery smoke: query `TasksManager.get_supported_tasks_for_model_type` for at least `llama`, `phi3`, `qwen2`, and one unsupported/unknown model type.
- Optimum ONNX export smoke: export one tiny text-generation model with `main_export` and record selected task, opset, exporter mode, and generated ONNX files.
- Export one tiny text-generation training ONNX graph.
- Generate ORT training artifacts and run a one-step desktop train smoke.
- Package artifacts into the current Android cache shape.
- Desktop GenAI smoke: load generated `genai_config.json` package and generate one token.
- Desktop GenAI handoff smoke: inject one changed trainable tensor with `params.set_model_input` and verify output/logits change.
- Android symbol smoke: verify whether `OgaCreateModelWithInitializers` is exported by the linked GenAI library.
- GenAI session-options smoke: add harmless `session_options.config_entries` to `genai_config.json` and verify they reach ORT logging; then try ORT-format memory-related keys on a tiny ORT-format package.
- GenAI config overlay smoke: create an `OgaConfig`, overlay or emit `session_options.config_entries`, construct a model from config, and verify behavior matches the file-based config.
- GenAI C++ API smoke: compile against the pinned Android GenAI headers and record whether the API surface is local params-level `SetModelInput`, upstream generator-level `SetModelInput`, or full `OgaNamedTensors` creation/set/get.
- Android GenAI smoke: load package, tokenize prompt, generate one token, stream token callback.
- Android handoff smoke: train -> merge -> GenAI model-input or initializer path sees changed output.
- Handoff-map validation smoke: every `requires_grad` / PEFT base layer maps to exactly one inference initializer or GenAI model input with matching dtype/shape.
- Export-mode smoke: generate the same tiny package in `external_initializer` and `model_input` modes and verify the manual ORT and GenAI paths consume the expected tensors.
- Quantized tensor naming smoke: verify merged `weight_quantized`, `weight_zero_point`, and `weight_scale` names match the exported inference graph exactly.
- Memory smoke: compare peak heap/RSS for current manual copied-buffer inference, manual `mmap` external initializer injection, GenAI model-input injection, GenAI initializer injection, and adapter load on the same fixture.

## Acceptance Criteria

- A written A/B/C decision exists for GenAI model-input injection, initializer injection, adapter files, or manual inference.
- A written A/B/C decision exists for the export/training-artifact toolchain.
- A pinned dependency set exists for at least one reproducible train -> merge -> infer path.
- The source-built ORT training wheel/AAR process is documented with build flags, ORT revision, checksums, and local install path.
- Memory-mapped or low-copy merged-weight handoff is documented and covered by a smoke test.
- The inference export path emits a validated trainable/merged tensor handoff map.
- GenAI `session_options.config_entries` are supported as session configuration, but not documented as a trained-weight injection mechanism.
- The chosen handoff mode explicitly states whether trained tensors are consumed as ORT external initializers, GenAI model inputs, or adapters.
- Packaging and config restructure scope is approved.
- License decision is recorded as a release blocker or accepted deferral.

## Source Links

- ONNX Runtime GenAI: https://onnxruntime.ai/docs/genai/
- ONNX Runtime GenAI C API: https://onnxruntime.ai/docs/genai/api/c.html
- ONNX Runtime GenAI C++ API: https://onnxruntime.ai/docs/genai/api/cpp.html
- ONNX Runtime GenAI Java API: https://onnxruntime.ai/docs/genai/api/java.html
- ONNX Runtime GenAI Python API: https://onnxruntime.ai/docs/genai/api/python.html
- ONNX Runtime GenAI config reference: https://onnxruntime.ai/docs/genai/reference/config.html
- ONNX Runtime GenAI past-present share buffer: https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html
- ONNX Runtime GenAI adapter spec: https://onnxruntime.ai/docs/genai/reference/adapter.html
- ONNX Runtime GenAI LoRA fine-tuning tutorial: https://onnxruntime.ai/docs/genai/tutorials/finetune.html
- ONNX Runtime session option config keys: https://raw.githubusercontent.com/microsoft/onnxruntime/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
- ONNX Runtime C API SessionOptions initializer APIs: https://raw.githubusercontent.com/microsoft/onnxruntime/main/include/onnxruntime/core/session/onnxruntime_c_api.h
- ONNX Runtime GenAI `OgaNamedTensors` source: https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp
- ONNX Runtime GenAI generator source: https://github.com/microsoft/onnxruntime-genai/blob/main/src/generators.cpp
- ONNX Runtime GenAI extra input source: https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/extra_inputs.cpp
- ONNX Runtime GenAI model/state source: https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.cpp
- ONNX Runtime on-device training: https://onnxruntime.ai/docs/get-started/training-on-device.html
- ONNX Runtime training build: https://onnxruntime.ai/docs/build/training.html
- ONNX Runtime Android build: https://onnxruntime.ai/docs/build/android.html
- ONNX Runtime Training PyPI: https://pypi.org/project/onnxruntime-training/
- ONNX Runtime Training Android AAR: https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-training-android
- Optimum releases: https://github.com/huggingface/optimum/releases
- Optimum ONNX releases: https://github.com/huggingface/optimum-onnx/releases
- Optimum ONNX PyPI: https://pypi.org/project/optimum-onnx/
- Optimum ONNX overview and supported architectures: https://huggingface.co/docs/optimum-onnx/onnx/overview
- Optimum ONNX export guide and TasksManager: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
- uv dependency management: https://docs.astral.sh/uv/concepts/projects/dependencies/
- uv locking and syncing: https://docs.astral.sh/uv/concepts/projects/sync/
- uv lockfile export: https://docs.astral.sh/uv/concepts/projects/export/
- MobileFineTuner: https://arxiv.org/abs/2512.08211

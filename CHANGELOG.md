# Changelog

All notable changes to MobileTransformers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows Semantic Versioning from
v1.0.0 onward.

## [Unreleased]

### Added
- **Repository restructure complete.** All Python now lives in `src/mobiletransformers/`; the seven
  legacy roots (`trainer/`, `artifact/`, `inference/`, `tools/`, `peft_models/`, `database/`,
  `evaluation/`) are gone, along with their deprecation shims. The built wheel is self-contained,
  verified by installing it into a clean venv and importing from outside the checkout.
- Architecture registry covers every supported model (16 rows): adding one is a data row, not an
  `elif`. Branch side effects (forced execution provider/precision, `exclude_embeds`, `hidden_act`)
  are fields on the row.
- `export/quantizer_compat.py`: resolves ONNX Runtime's weight-only MatMul quantizer across the
  `MatMul4BitsQuantizer` → `MatMulNBitsQuantizer` rename, so the inference builder works on either.
- Export-time merge-contract check: every `trainingBaseLayerName` in `weight_handoff_map.json` must
  name a real checkpoint parameter, or the export fails.
- **Export-time numeric gates on the training stage.** `artifacts/parameter_budget.py` counts the
  training graph's parameters **per dtype** against the source model's own count (recorded at export
  time) and fails closed on a shortfall or on a graph with no trainable float parameters;
  `artifacts/train_inference_parity.py` runs identical tokens through the package's inference and
  training graphs and bounds the cross-entropy gap. Until now every check on the training stage was
  byte-level or structural, so a package carrying a fraction of its model would have passed them all.
- `inferenceGraphPrecision` in `inference/optimum_config.json`, **measured from the shipped graph**.
  A variant id names the training-side quantization (`--quant`), which the inference export does not
  apply — so `cpu-int4` can legitimately hold an fp32 inference graph. Precision is now declared
  rather than inferred from a directory name.
- Task registry (`config/registry/task.py`): the auto-model class, KV-cache kwargs, PEFT `task_type`,
  label shape, quantization exclusions and training-wrapper class are data per `TaskType`, removing
  the last task-shaped `if/elif` chain from the export path. Adding a training objective is a row.
- **Encoder fine-tuning (#33), host legs complete.** `TaskType.SEQUENCE_CLASSIFICATION` plus registry
  rows for BERT/RoBERTa/DistilBERT classification: export → training artifacts → train step → metric
  works on a real encoder (loss −21.8%, accuracy 0.25 → 1.00). The Android smoke remains device-gated.
- PEFT target modules now come from the architecture registry per model, with `--peft-target` (CLI)
  and `peft_target:` (export YAML) to override. **Decoder exports now adapt `q_proj`/`v_proj`** (the
  LoRA convention) rather than `q_proj`/`k_proj`; pass `--peft-target q_proj,k_proj` for the old
  behaviour.
- Device acceptance suite (13 instrumented tests) covering load→generate, conversation reset,
  dual-engine parity (first token **and** ordered callback sequence), RAG, ObjectBox ranking, a
  four-point RSS table per engine, and train→merge→generate.
- One-command export CLI + Hub package format (manifest-first cache bridge), with `--config` (YAML
  supplying any unset flag, CLI > YAML > default) and `--validate`.
- Hub pull/install + adapter push-back (Python), background download via WorkManager (Android).
- Kotlin facade: `MobileTransformers.fromPretrained` → `MobileTransformerModel`, plus `TrainingJob`
  (status/event flows, cooperative cancel, checkpoint/resume).
- Dual inference engines (`NATIVE`, `GENAI`) behind one `ModelRuntime` over a single package.
- On-device RAG: `VectorStore` boundary, ingestion/chunking, grounded generation.
- Federated adapter record codec + FedAvg aggregation (`mobiletransformers federated simulate`).
- Per-role `tensorDtypes`/`tensorShapes` in `weight_handoff_map.json`, so the device can load packed
  quantized tensors from headerless external data.
- Test harnesses for all three languages: Robolectric (Kotlin), googletest (C++), and `make guard`
  ratchets for credential reads and registry-dispatch literals.
- `maven-publish` for the Android SDK
  (`com.martinkorelic.mobiletransformers:mobiletransformers-android`).
- Docs: `ARCHITECTURE.md`, `HUB_PACKAGE_FORMAT.md`, `ANDROID_CACHE_FORMAT.md`, `FEDERATED.md`,
  `MODEL_FORMAT.md`, `CONFIGURATION.md`, `PUBLIC_API.md`, `EXPORT.md`, `RAG.md`, and a generated
  `COMPATIBILITY_MATRIX.md`.

### Fixed
- GenAI never loaded on any package the training stage produced: `genai_config.json` carried a
  `session_options.config_entries` key that onnxruntime-genai 0.14 rejects outright, taking the whole
  config with it. The runtime then fell back to Native **silently**, so the dual-engine parity test
  compared Native with Native and passed. Unsupported keys are now stripped at export, and an
  explicitly requested engine that cannot load raises instead of substituting another.
- A generation session that failed to construct was logged and forgotten, leaving `generate` to return
  nothing with no error. The cause is now retained and re-raised.
- One shared C++ layer-name normalizer (`cpp/layer_name.h`) replaces nine open-coded prefix rewrites
  that had produced five device-only merge defects.
- `make_mlp_unpacked_lora` referenced unbound `q_proj`/`k_proj` instead of the `gate_proj`/`up_proj` it
  builds — a latent `NameError` on any unpacked-MLP LoRA export.
- PEFT's `task_type` was hardcoded to `"CAUSAL_LM"` at both LoRA call sites, which mis-configures any
  non-decoder model: PEFT uses it to decide which modules to wrap and which head stays trainable. It
  now comes from the task registry.
- The training-graph wrapper's forward signature was decoder-only (`position_ids`), so a BERT-family
  export failed inside optimum at `check_dummy_inputs_are_allowed`. Encoders get their own wrapper
  (`token_type_ids`); because the signature's parameter names become the exported ONNX input names,
  the decoder's are deliberately unchanged.
- `onnx_checktrain` pre-shifted its labels even though the exported graph already applies the HF causal
  shift, inflating every loss it printed and making it incomparable to the on-device number. The
  Android path was always correct.
- Dynamic quantization could place a **quantized activation on the gradient path**, which has no
  gradient at all (quantized *weights* are frozen and dequantize to float, so they were never the
  problem). ONNX Runtime rewrites `Gemm` → `MatMul` before quantizing and matches `nodes_to_exclude`
  against the rewritten name, so excluding such a node by its own name silently did nothing.
- `LayerNormalization` was exported with only its `Y` output, but ORT's gradient reads the optional
  saved mean / inverse-std outputs — so no gradient graph could be built through it. Decoders were
  unaffected (RMSNorm exports as `SimplifiedLayerNormalization`, which already carries them).
- Gemma2/Gemma3 were bound to the generic `GemmaOnnxConfig`; they now bind their own.
- `emit_merger_models` was missing from its module's `__all__`, so the declared public surface
  disagreed with the real one.
- Re-exporting into an existing directory silently corrupted the package: `onnx` *appends* external
  data, so a second export doubled every trainable tensor and the device rejected the sizes.
- GenAI was unreachable end to end: the config mapper never set the engine field the runtime factory
  reads, and the repository dispatched on a string that dropped every GenAI config — leaving `generate`
  suspended forever.
- Merged weights are written as raw external data but were parsed as `TensorProto`, so merged-weight
  load failed on the shipping (non-mmap) path.
- Merged-weight load failure, and a partial on-device merge, both reported success. Every gate on the
  train → merge → generate path now fails closed rather than silently serving base weights.
- Post-merge checksum contract: the `.bin.sha256` sidecar (refreshed by the merger) now takes
  precedence over the manifest-time digest, which a correct merge necessarily invalidates.
- `LinearLRScheduler.stateDict()`/`loadFromState()` were `TODO()`, crashing any run on the **default**
  schedule at its first checkpoint.
- A Mode-1 (PEFT) adapter push published `adapter_config.json` with no weights.
- RAG configuration was applied only once per session; later `topK`/`minScore`/`searchType` changes
  were silently ignored.
- Engine parity: identical sampling resolution, token count and throughput reporting across engines.
- Package installation is crash-safe (rename-aside → rename-in → delete-old); the previous
  delete-then-rename could destroy an installed model, including local training state.
- Credentials are read through `config.settings` instead of ad-hoc `os.environ[...]` reads.

### Known issues
- **Train and inference halves of one package are quantized differently, and the gap is unmeasured
  end to end.** A variant may ship a uint8 weight-quantized training graph beside an fp32 inference
  graph. The export now gates on parameter count and on a train-vs-inference loss delta, but the
  device-side numeric equivalence of the two halves is asserted only at export, not after a merge.
  *(This replaces an earlier "training starts from weights that are not the pretrained ones" entry,
  which was wrong: it counted uint8 checkpoint tensors as fp32 and concluded two thirds of the model
  was missing. The training graph carries all 135,436,915 parameters — see `agent_docs/HANDOFF.md`.)*
- **Memory-mapped weight loading covers only the trainable split** (~8% of weight bytes); the frozen
  base still loads through ORT's own external-data path, so whole-process peak RSS improves by ~6%.
  Within its own scope the zero-copy path realises 92.9% of the attainable saving. It is default-off
  (`debug.mtf.mmap_weights`) and does not block v1; extending it to `frozen_base.onnx.data` is tracked
  as a non-blocking follow-up.
- **arm64-v8a only.** No x86_64 build of ONNX Runtime/tokenizers exists here, so the library does not
  run on an x86_64 emulator.
- The project is CC-BY-NC-4.0, which is incompatible with distributing the AAR for commercial use.

### Non-goals
- **GPU/NPU training.** Inference may use an accelerated execution provider; training is CPU-only.
- **Multimodal training.** Text-generation and encoder tasks only.
- **Competing with server-side trainers on throughput.** The target is feasibility and privacy on a
  phone, not tokens/second parity with a datacentre.
- **On-device engine/facade device parity**, which remains gated on device acceptance runs.

## [0.1.0] — unreleased

Pre-release development line. The public API is not yet frozen; `mobiletransformers.__all__`, the CLI
surface and the Kotlin facade are version-locked at v1.0.0.

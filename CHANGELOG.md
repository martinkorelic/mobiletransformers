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
- **On-device training starts from weights that are not the pretrained ones.** The optimizer works —
  loss falls monotonically — but the initial loss is ~14.3 against a uniform-prediction floor of
  `ln(49152) = 10.80`, while the *same* package generates coherent text through the inference path.
  The train stage ships a 2.6 MB graph plus a 176 MB checkpoint (~44M parameters) for a ~135M-parameter
  model, so most weights are in neither artifact. `TrainConvergenceTest
  .trainingStartsFromPretrainedWeightsNotRandomOnes` fails on purpose until this is fixed; fine-tuning
  cannot improve on the base model until it is.
- **Gate 0.2 (memory-mapped weights) is not met**: mmap covers only the trainable split (~8% of weight
  bytes) and measured a 6.3% peak-RSS reduction against a 15% target. It is default-off and does not
  block v1.
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

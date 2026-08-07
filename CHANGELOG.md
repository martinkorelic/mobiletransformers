# Changelog

All notable changes to MobileTransformers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows Semantic Versioning from
v1.0.0 onward.

## [Unreleased]

### Added
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

### Non-goals
- **GPU/NPU training.** Inference may use an accelerated execution provider; training is CPU-only.
- **Multimodal training.** Text-generation and encoder tasks only.
- **Competing with server-side trainers on throughput.** The target is feasibility and privacy on a
  phone, not tokens/second parity with a datacentre.
- **On-device engine/facade device parity**, which remains gated on device acceptance runs.

## [0.1.0] — unreleased

Pre-release development line. The public API is not yet frozen; `mobiletransformers.__all__`, the CLI
surface and the Kotlin facade are version-locked at v1.0.0.

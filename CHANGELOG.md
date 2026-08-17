# Changelog

All notable changes to MobileTransformers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows Semantic Versioning from
v1.0.0 onward.

## [0.2.0] — 2026-08-17

First public release.

### Added

**Host — exporting models**

- One-command export: a Hugging Face checkpoint becomes an ONNX inference graph, a PEFT-enabled
  training graph with an optimiser, a tokenizer, an optional embedding stage, and a manifest tying
  them together.
- **PEFT methods:** LoRA, LoRA-XS, and **MARS** (Multi-Adapter Rank Sharing) — this project's own
  method, which shares adapter components across layers so trainable parameters grow with rank
  rather than with depth.
- Decoder and encoder tasks: text generation, sequence classification, and feature extraction for
  retrieval.
- A `mobiletransformers` CLI covering export, packaging, validation, Hub push/pull, adapter
  conversion and federated aggregation.

**Android SDK**

- `mobiletransformers-android` — an AAR consumable from any app, with a single `fromPretrained`
  entry point that resolves, downloads, verifies and atomically installs a package.
- **Generation** — streaming, with a chat template and KV cache, over either of two selectable ONNX
  Runtime engines.
- **Fine-tuning** — a real training loop with an optimiser and a live loss curve, surviving the app
  going to the background.
- **Merging** — folding a trained adapter back into the inference weights, so the fine-tuned model is
  the one that then generates.
- **Retrieval** — chunking, embedding and vector search over documents you supply, with grounded
  generation that reports its sources before the answer.
- **Classification** for encoder packages, scored per label.
- **Tool calling** — a model's answer parsed into a structured call, validated against an allowlist
  the app owns, and bound to an Android intent. Nothing executes without user consent.
- **Federated adapter exchange** — export local factors, aggregate on a host, import the average
  back. Default-off and consent-gated.

All of it runs on the device, with no server and no data leaving the phone.

**Published artifacts**

- **Six model packages** on the Hub under
  [`mobiletransformers`](https://huggingface.co/mobiletransformers), each shipping both an inference
  and a training stage — including one exported with MARS.
- The gitignored native build artifacts are hosted, so a fresh clone can provision itself with
  `make fetch-native-deps` and no credentials.

**Sample app and documentation**

- A reference app exercising every capability through the public facade only, with navigation that
  derives from what the loaded package can actually do.
- A documentation site built from this repository's own `docs/`, and `make doctor` — a preflight
  report naming every missing prerequisite and the command that fixes it.

### Known issues

- The host gates are green, but nothing in this release is exercised end to end on a device by CI.
- **arm64-v8a only.** The SDK does not run on a standard x86_64 Android emulator.
- The training export path is Linux x86_64 and Python 3.12 only, because of the source-built ONNX
  Runtime Training wheel.
- The licence is CC BY-NC 4.0, which does not suit a consumable Android library. Relicensing is a
  rights-holders decision and is the outstanding blocker for v1.0.

### Non-goals

- **GPU/NPU training.** Inference may use an accelerated execution provider; training is CPU-only.
- **Multimodal training.** Text-generation and encoder tasks only.
- **Competing with server-side trainers on throughput.** The target is feasibility and privacy on a
  phone, not tokens/second parity with a datacentre.
- **On-device engine/facade device parity**, which remains gated on device acceptance runs.

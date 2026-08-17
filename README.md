![MobileTransformers](docs/assets/mobiletransformers_banner.png)

# MobileTransformers: An On-Device LLM PEFT Framework for Fine-Tuning and Inference

[![checks](https://github.com/martinkorelic/mobiletransformers/actions/workflows/checks.yml/badge.svg)](https://github.com/martinkorelic/mobiletransformers/actions/workflows/checks.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Android 7.0+](https://img.shields.io/badge/Android-API%2024%2B-3DDC84?logo=android&logoColor=white)](docs/ANDROID_SDK.md)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-mobiletransformers-FFD21E)](https://huggingface.co/mobiletransformers)

**Export a Hugging Face model, pull it onto a phone, then chat with it, retrieve over your own
documents, classify text, fine-tune it, merge the adapter into the weights, and let it call tools —
entirely on the device.** No server, no inference API, no data leaving the phone.

Built on **ONNX Runtime**, for both inference *and* training on Android.

---

## See it work

|🧩 Base model |⚙️ Fine-tuned on the phone|
|----|----|
|![Base on-device model](docs/assets/base-model.gif)|![On-device trained LLM model](docs/assets/on-device-trained.gif)|

**The same prompt, before and after a training run and an on-device merge.** Adapted to a
smartphone-automation dataset, where a user states an intent and the model proposes the action.
Trained, merged and run entirely on the phone — no server at any point.

<!--
  RECORDING PLACEHOLDERS. Mechanics and the ffmpeg recipe: docs/assets/README.md

  1. REPLACE the two-GIF table above with a single clip, on-device-finetune.gif — one unbroken take of
     prompt -> train -> merge -> the same prompt. Two files the reader has to compare by eye is a
     weaker version of the same claim. Delete base-model.gif and on-device-trained.gif in that commit.

  2. ADD offline-generation.gif below this comment, with the caption:
     "Airplane mode, still generating. Nothing is uploaded and nothing is fetched — the weights and
      the tokenizer are already on the device."
     This is the strongest thing the app can do and no current asset shows it.

  3. ortransformer-feature.gif predates the app rewrite and is no longer referenced. Delete it.

  Only these two clips belong in the README. Everything else — the drawer changing shape per package,
  tool calls firing a real alarm, install-from-catalog, retrieval, classify — goes in
  docs/SHOWCASE.md beside the section it illustrates. A README with seven GIFs is a README nobody
  finishes loading.
-->

More, capability by capability, in **[docs/SHOWCASE.md](docs/SHOWCASE.md)**.

---

## What is actually here

| | |
| --- | --- |
| **A host export pipeline** | Hugging Face → PEFT-enabled training graph + ONNX inference graph + a manifest, in one command |
| **An Android SDK** (Kotlin + C++) | `mobiletransformers-android` — an AAR you can consume from your own app |
| **A sample app** | The reference consumer of that SDK, and the fastest way to see the whole loop |
| **A published model shelf** | Six packages on the Hub, each shipping both an inference and a training stage — including one exported with MARS |
| **Custom PEFT methods** | LoRA, LoRA-XS, and **MARS** (Multi-Adapter Rank Sharing) — the project's own method |
| **Federated adapter exchange** | Export local factors, aggregate on a host, import the average back |

The two things that distinguish this from "run a small model on a phone": **training happens on the
device**, and the trained adapter is **merged into the inference weights on the device**, so the
personalised model is the one that then generates.

## Quick start

```bash
make doctor                   # what is missing, and the command that fixes each
make setup                    # core + dev environment (uv, Python 3.10)
make check                    # lint + typecheck + enum parity + guards + unit tests
```

Export a package and put it on a phone:

```bash
make setup-export
mobiletransformers export --model HuggingFaceTB/SmolLM2-135M-Instruct \
                          --output build/pkg --genai --validate

make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct TRAIN=1 RAG=1
make device-test
```

Or build the app and install a package from the Hub inside it:

```bash
make fetch-native-deps        # the gitignored Android natives — see below
make android-build
```

Dependency profiles are deliberately isolated: the `export` extra and the `ort-training-local` group
**cannot** co-install. Always pass an explicit `--group`/`--extra` to `uv run`, and reset with
`uv sync --frozen --group dev --python 3.10` before `make check` — a leftover profile is the single
most common way to "break" the repo. See [docs/EXPORT.md](docs/EXPORT.md).

> **A fresh clone cannot build the Android SDK on its own.** ~180 MB of prebuilt native binaries and
> vendored headers are gitignored. `make doctor` tells you what is missing; `make fetch-native-deps`
> gets it. See [docs/ARCHITECTURE.md ▸ Native dependencies](docs/ARCHITECTURE.md).

## The model shelf

Six packages under [`mobiletransformers`](https://huggingface.co/mobiletransformers) on the Hub.
Every one ships **both an inference and a training stage** — a shelf entry that cannot be fine-tuned
demonstrates half the framework, so `scripts/publish_catalog.sh` asserts it.

| model | task | inference | total | features |
| --- | --- | --- | --- | --- |
| [SmolLM2-135M-Instruct](https://huggingface.co/mobiletransformers/SmolLM2-135M-Instruct) | text-generation | 663 MB | 935 MB | inference, train, rag |
| [functiongemma-270m-it](https://huggingface.co/mobiletransformers/functiongemma-270m-it) | text-generation | 3557 MB | 3875 MB | inference, train |
| [gemma-3-270m-it](https://huggingface.co/mobiletransformers/gemma-3-270m-it) | text-generation | 1814 MB | 2131 MB | inference, train (**MARS**) |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/mobiletransformers/Qwen2.5-0.5B-Instruct) | text-generation | 2554 MB | 3212 MB | inference, train, rag |
| [all-MiniLM-L6-v2](https://huggingface.co/mobiletransformers/all-MiniLM-L6-v2) | text-classification | 94 MB | 214 MB | inference, train, rag |
| [distilbert-sst2-english](https://huggingface.co/mobiletransformers/distilbert-sst2-english) | text-classification | 270 MB | 361 MB | inference, train |

Sizes are measured off each pushed package's manifest, not estimated. Start with **SmolLM2**. Full
detail, and why the encoders are exported as `text-classification`, in
[docs/CATALOG.md](docs/CATALOG.md).

## The sample app

Eight destinations, and which you see depends on what the loaded package can actually do — a chat box
on an embedding model is a promise the package cannot keep, so it is hidden rather than greyed out.

**Models** → **Chat** (streaming, grounded answers, tool calls) → **Retrieval** → **Classify** →
**Train** (live loss curve, then merge) → **Federated** → **Configuration** → **About**.

[docs/SHOWCASE.md](docs/SHOWCASE.md) is the tour: one section per capability, the package each needs,
and what you should see.

## Documentation

| Page | Covers |
| --- | --- |
| [docs/SHOWCASE.md](docs/SHOWCASE.md) | a tour of the sample app, capability by capability |
| [docs/CATALOG.md](docs/CATALOG.md) | the published packages: sizes, features, which to start with |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the host exporter and the Android SDK fit together; native dependencies |
| [docs/EXPORT.md](docs/EXPORT.md) | the one-command export CLI, profiles, per-task flag rules |
| [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md) | the manifest + `weight_handoff_map.json` on-disk contracts |
| [docs/HUB_PACKAGE_FORMAT.md](docs/HUB_PACKAGE_FORMAT.md) | package layout on the Hub; pull/verify/install |
| [docs/ANDROID_SDK.md](docs/ANDROID_SDK.md) | consuming the AAR: install, load, generate, classify, retrieve, train, merge |
| [docs/COOKBOOK.md](docs/COOKBOOK.md) | copy-pasteable Kotlin per task, mirroring the app's screens |
| [docs/ANDROID_CACHE_FORMAT.md](docs/ANDROID_CACHE_FORMAT.md) | where an installed model lives on device |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | the enum vocabulary, typed configs, extension points |
| [docs/PUBLIC_API.md](docs/PUBLIC_API.md) | the Python, CLI and Kotlin public surfaces |
| [docs/RAG.md](docs/RAG.md) | on-device retrieval, ingestion, grounded generation |
| [docs/FEDERATED.md](docs/FEDERATED.md) | federated adapter exchange + the Flower simulation |
| [docs/COMPATIBILITY_MATRIX.md](docs/COMPATIBILITY_MATRIX.md) | per-model support, generated from the matrix |
| [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) | what a release requires |
| [docs/mobile_evaluation.md](docs/mobile_evaluation.md) | host-side evaluation of on-device runs |

### External

- [MobileTransformers Documentation](https://martinkorelic.github.io/mobiletransformers-docs/)
- [Master's Thesis — Parameter-Efficient Tuning of Large Language Models on Mobile Devices](https://repozitorij.uni-lj.si/IzpisGradiva.php?lang=eng&id=175561)
  — the research behind MARS, the on-device training methodology, and the experimental results.

## Built on

- [**ONNX Runtime**](https://onnxruntime.ai/) — training and inference, with XNNPACK / NNAPI /
  Qualcomm QNN execution providers
- [**Hugging Face Transformers**](https://huggingface.co/) + [**Optimum**](https://huggingface.co/docs/optimum/)
  — model export
- [**ObjectBox**](https://objectbox.io/) — the on-device vector database behind RAG

## Where this is going

- Beyond generation and classification: NER, question answering, summarization
- On-device reinforcement learning
- More PEFT methods and quantization techniques
- Additional hardware acceleration backends, and platforms beyond Android

## Citation

If you are using this framework for your own work, please cite:

```
@misc{mobiletransformers2025,
  author       = {Koreli\v{c}, Martin and Pejovi{\'c}, Veljko},
  title        = {MobileTransformers: An On-Device LLM PEFT Framework for Fine-Tuning and Inference},
  year         = {2025},
  howpublished = {\url{https://gitlab.fri.uni-lj.si/lrk/mobiletransformers}}
}
```

## Acknowledgements

This work was supported by the Slovenian Research Agency grant no. N2-0393 approXimation for adaptable diStributed artificial intelligence and grant no. J2-3047 Context-Aware On-Device Approximate Computing.

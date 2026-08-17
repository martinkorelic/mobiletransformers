# MobileTransformers

Export a Hugging Face model, pull it onto a phone, then chat with it, retrieve over your own
documents, classify text, fine-tune it, merge the adapter into the weights, and let it call tools —
**entirely on the device**. No server, no inference API, no data leaving the phone.

Built on **ONNX Runtime**, for both inference *and* training on Android.

[Get started](getting-started.md){ .md-button .md-button--primary }
[See the app](SHOWCASE.md){ .md-button }
[Source on GitHub](https://github.com/martinkorelic/mobiletransformers){ .md-button }

---

## Two halves that meet at a file format

The project is a **Python exporter** and an **Android SDK**, and they never call each other. The
exporter turns a Hugging Face checkpoint into a package of ONNX graphs, a tokenizer and a manifest;
the SDK reads that package on a phone. Everything they have to agree about is written down in
[the package format](MODEL_FORMAT.md) rather than implied by matching code.

```bash
# Host: produce a package
mobiletransformers export --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --output build/package --train --rag
```

```kotlin
// Device: consume one
val model = MobileTransformers.fromPretrained(
    context, repoId = "mobiletransformers/SmolLM2-135M-Instruct",
    features = setOf(ModelFeature.Inference, ModelFeature.Training),
)
val answer = model.generate("Summarise this in one line: …")
```

## Where to go

| If you want to… | Read |
| --- | --- |
| install the app and see it work | [Getting started](getting-started.md), then [the tour](SHOWCASE.md) |
| pick a model to try first | [The model shelf](CATALOG.md) — six published packages, measured sizes |
| build an app on the SDK | [Using the SDK](ANDROID_SDK.md), then the [cookbook](COOKBOOK.md) |
| know exactly what is API and what is not | [Public API](PUBLIC_API.md) |
| export your own model | [Export a model](EXPORT.md) |
| understand the fine-tuning methods | [PEFT methods](on-device-peft.md) — LoRA, LoRA-XS, MARS |
| ground answers in your own documents | [Retrieval](RAG.md) |
| know how fast it actually is | [Measured performance](mobile_evaluation.md) |
| understand how the pieces fit | [Architecture](ARCHITECTURE.md) |

## What is genuinely on the device

Every one of these runs with the network off, once the package is installed:

- **Generation** — streaming, with a chat template, KV cache and a choice of two ONNX Runtime engines.
- **Fine-tuning** — a real training loop with an optimiser and a loss curve, not a fixed-function call.
  LoRA, LoRA-XS and **MARS**, this project's own method.
- **Merging** — folding a trained adapter back into the inference weights, so the fine-tune survives
  into ordinary generation.
- **Retrieval** — chunking, embedding and vector search over documents you supply.
- **Classification** — encoder packages with a real head, scored per label.
- **Tool calling** — a model's answer parsed into a structured call, validated against an allowlist
  the app owns, and bound to an Android intent.

## Status

Version **0.2.0**. The exporter, the Android SDK, the sample app and the published model shelf all
work end to end. See the [release checklist](RELEASE_CHECKLIST.md) for what stands between this and a
1.0, and the repository's `CHANGELOG.md` for what has changed.

!!! warning "Licensing"

    The repository currently ships **CC BY-NC 4.0**, which does not suit a consumable Android
    library. Relicensing is pending agreement between both authors — see
    [the release checklist](RELEASE_CHECKLIST.md). Check the licence before depending on this.

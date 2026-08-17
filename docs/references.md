# Further reading

## This project

- [Source code on GitHub](https://github.com/martinkorelic/mobiletransformers) — where to get it, and
  what these docs are built from
- [The original codebase](https://gitlab.fri.uni-lj.si/lrk/mobiletransformers) — the address the work
  was published under, and the one the [citation](citation.md) names
- [Model packages on Hugging Face](https://huggingface.co/mobiletransformers) — six exported
  packages, each shipping both an inference and a training stage
- [*Parameter-Efficient Tuning of Large Language Models on Mobile Devices*](https://repozitorij.uni-lj.si/IzpisGradiva.php?lang=eng&id=175561)
  — the master's thesis this framework accompanies

## Work built on this framework

- Korelič, M. and Pejović, V. — [**AI health agents on mobile**](https://link.springer.com/article/10.1186/s12919-026-00367-3#Sec27),
  *BMC Proceedings* 2026, 20(12):A7. Presented at EHRCON25, the openEHR International Conference.

    The first on-device Retrieval-Augmented Generation prototype for openEHR-based personal health
    data, running entirely on a smartphone: a small language model, an embedding model and a vector
    database of vital signs, medications, allergies and laboratory results, with no network
    dependency and no records leaving the device. The Android application is built on this
    framework.

    Two INT4-quantized models were measured on a Pixel 6 CPU — TinyLlama (1.1B) at 0.94 GB and
    9.04 tokens/second, and Phi-3-mini-4k (3.5B) at 2.7 GB and 3.6 tokens/second — with answer
    quality judged against cloud LLM responses using G-Eval. It is a useful independent read on what
    this framework's [measured performance](mobile_evaluation.md) looks like in an applied setting.

## What it is built on

- [ONNX Runtime](https://onnxruntime.ai/) — the inference and **training** engine, on device
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) — the alternative generation
  engine, selectable per package
- [Optimum](https://huggingface.co/docs/optimum/) — the Hugging Face → ONNX export path
- [PEFT](https://huggingface.co/docs/peft/) — LoRA and the tuner base MARS is built on
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) — the on-device tokenizer
- [ObjectBox](https://objectbox.io/) — the on-device vector store behind [retrieval](RAG.md)

Every third-party component and its licence is listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/martinkorelic/mobiletransformers/blob/main/THIRD_PARTY_NOTICES.md).

## Background

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters](https://arxiv.org/abs/2405.17604)

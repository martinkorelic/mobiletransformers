# Further reading

## This project

- [**Source code on GitHub**](https://github.com/martinkorelic/mobiletransformers) — where to get it, and
  what these docs are built from
- [**The original codebase**](https://gitlab.fri.uni-lj.si/lrk/mobiletransformers) — the address the work
  was published under, and the one the [citation](citation.md) names
- [**Model packages on Hugging Face**](https://huggingface.co/mobiletransformers) — six exported
  packages, each shipping both an inference and a training stage
- [*Parameter-Efficient Tuning of Large Language Models on Mobile Devices*](https://repozitorij.uni-lj.si/IzpisGradiva.php?lang=eng&id=175561) — the master's thesis this framework accompanies
- [**SELLMA: Semantic Location through On-Device LLMs and WiFi Sensing**](https://dl.acm.org/doi/10.1145/3721888.3722091) — On-device semantic location recognition through privacy-preserving local LLMs with WiFi sensing.
- [**AI health agents on mobile**](https://link.springer.com/article/10.1186/s12919-026-00367-3#Sec27),
  *BMC Proceedings* 2026, 20(12):A7 (EHRCON25 — openEHR International Conference).
  On-device RAG prototype over openEHR personal health records: a small language model, an
  embedding model and a vector database of vital signs, medications, allergies and lab results, all
  running on the phone. Built on this framework.

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

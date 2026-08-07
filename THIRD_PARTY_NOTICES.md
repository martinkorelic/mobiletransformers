# Third-party notices

MobileTransformers redistributes or depends on the components below. Each remains under its own
licence; nothing here alters the terms of [`LICENSE.md`](LICENSE.md).

Vendored native binaries live under `android/MobileTransformers/MobileTransformers/src/main/jniLibs/`
and `.../src/main/cpp/` and are **not** covered by this project's copyright.

## Redistributed in the Android artifact

| Component | Licence | Notes |
| --- | --- | --- |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | training build (`libonnxruntime.so`) + a stock build shipped under a distinct soname (`libort_gen.so`) so the GenAI engine can coexist |
| [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) | MIT | `libonnxruntime-genai.so`; its `dlopen` target is repointed at `libort_gen.so` |
| [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) | Apache-2.0 | native tokenizer (`libtokenizers_c`, `libtokenizers_cpp`) |
| [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | via tokenizers-cpp |
| [ObjectBox](https://objectbox.io/) | Apache-2.0 | on-device vector store (`libobjectbox-jni.so`) |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | header-only JSON (fetched at build time) |

## Build- and test-time only

| Component | Licence |
| --- | --- |
| [googletest](https://github.com/google/googletest) | BSD-3-Clause |
| [Robolectric](https://robolectric.org/) | MIT |
| [OkHttp / MockWebServer](https://square.github.io/okhttp/) | Apache-2.0 |
| [AndroidX](https://developer.android.com/jetpack/androidx) (core, appcompat, work, test) | Apache-2.0 |
| [Kotlin stdlib / coroutines](https://kotlinlang.org/) | Apache-2.0 |
| [Gson](https://github.com/google/gson) | Apache-2.0 |

## Python dependencies

| Component | Licence |
| --- | --- |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| [PEFT](https://github.com/huggingface/peft) | Apache-2.0 |
| [Optimum](https://github.com/huggingface/optimum) / optimum-onnx | Apache-2.0 |
| [ONNX](https://onnx.ai/) | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 |
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [Pydantic](https://docs.pydantic.dev/) | MIT |
| [PyYAML](https://pyyaml.org/) | MIT |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | BSD-3-Clause |
| [Flower](https://flower.ai/) *(optional, installed out-of-band)* | Apache-2.0 |

## Vendored source

| Path | Origin | Licence |
| --- | --- | --- |
| `src/mobiletransformers/export/onnx_config_with_loss.py` | Optimum 1.24 (`OnnxConfigWithLoss`, removed in Optimum 2.1) | Apache-2.0 |
| `.../cpp/onnxruntime/`, `.../cpp/onnxruntime-genai/` | upstream C API headers | MIT |
| `.../cpp/tokenizers/` | tokenizers-cpp headers | Apache-2.0 |

Run `uv tree` (Python) or `./gradlew :MobileTransformers:dependencies` (Android) for the exact resolved
dependency set at a given version.

# research/

**Not part of the shipped framework.** Nothing here is installed by the wheel, imported by
`src/mobiletransformers/`, or run by any gate. It is the experimental and measurement work the
framework came out of, kept because the results in the thesis and in
[`docs/mobile_evaluation.md`](../docs/mobile_evaluation.md) were produced by these scripts and would
otherwise be unreproducible.

Expect rougher edges than the rest of the repository: these are experiment scripts, not a library.
Several assume paths, datasets or credentials that are not in this repo, and they are excluded from
`ruff`/`mypy` for that reason. **If you are here to use MobileTransformers, you want
[`docs/`](../docs/) instead.**

| directory | what it is |
| --- | --- |
| `evaluation/` | On-device and host-side benchmarking: the harnesses behind the measured numbers, plus mobile profiling captures. Has its own [README](evaluation/README.md). |
| `plots/` | The figure generators for those results — ablations, size comparisons, per-task radar charts. |
| `genai/` | A desktop reference implementation of the GenAI generation loop. The Android engine is a port of it, so this is the version you can step through in a debugger. Has its own [README](genai/README.md). |
| `onnx_experiments/` | Exploratory ONNX graph work from before the export pipeline existed: quantization at several widths, external-data handling, graph rewriting, dtype casting. Superseded by `src/mobiletransformers/export/`, kept as the record of what was tried. |
| `pytorch_experiments/` | Host-side PEFT experiments that preceded MARS — low-rank matrix behaviour, dynamic training. `cca_core.py` is a third-party CCA/SVCCA implementation used for representation-similarity analysis. |
| `tflite/` | An abandoned TensorFlow Lite export path, evaluated and not taken. Kept so the decision is legible: this project is ONNX Runtime end to end, and this is why it is not TFLite. |
| `ablation_analysis.py`, `offline_train_eval.py`, `trainer_callbacks.py`, `utils.py` | Shared helpers for the above. |

## Running any of it

These are not covered by the project's dependency profiles and may need packages the framework does
not depend on (`tensorflow`, plotting libraries, evaluation harnesses). Install what a given script
imports, in a throwaway environment.

Credentials, where a script needs them, are read from the environment — never written into the file.
See [`.env.example`](../.env.example).

"""Reusable evaluators (Migration Map S8, formerly the ``evaluation/`` root).

What landed here is the code with an **importable API** — evaluator classes and functions a caller can
drive. The former `evaluation/benchmark/` and `evaluation/test/` trees did **not**: they carry no
classes or functions at all, run their work in top-level statements at import time, and hardcode
`experiment_results/...` paths. Packaging those would put side-effecting scripts inside an installable
wheel, so they moved to `research/evaluation/` instead, following the same call S5 made for
`artifact/tflite_builder.py`. (Despite its name, `evaluation/test/` contained no tests.)

Modules:

* :mod:`~mobiletransformers.evaluation.eval_adapter_models` — `CustomPeftModel`, the PEFT-adapter
  wrapper the deepeval benchmarks drive.
* :mod:`~mobiletransformers.evaluation.eval_adapter_onnx_model` — the ONNX counterpart.
* :mod:`~mobiletransformers.evaluation.mobile_evaluator` — `MobileEvaluator`.
* :mod:`~mobiletransformers.evaluation.mobile` — on-device recommendation / personal-QA evaluators.
* :mod:`~mobiletransformers.evaluation.openehr` — the openEHR case study and its plots.

Nothing is re-exported at package level: these need the ``eval`` extra (deepeval, matplotlib) and in
places torch/transformers, none of which the core profile installs. Import the submodule you need.

``evaluation/`` still holds deprecation shims re-exporting these names; they are removed in S9.
"""

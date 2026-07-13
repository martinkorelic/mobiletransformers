"""Generate the tiny trainable ONNX fixture + training_config.json for the ORT-training smoke.

The fixture mirrors the shape `artifact/onnx_builder.py::gen_artifacts` expects: an ONNX graph with
at least one trainable initializer and a scalar output that IS the loss (generate_artifacts is called
with ``loss=None``), plus a ``training_config.json`` carrying the exact fields the builder reads
(``requires_grad``, ``peft_mapping``, ``rank``, ``alpha``, ``peft_target``,
``trainable_parameter_count``).

Regenerate with: ``python tests/fixtures/make_tiny_trainable.py``. Uses only onnx (a core dep), so it
runs in any environment — no onnxruntime-training needed to BUILD the fixture (only to consume it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FIXTURE_DIR = Path(__file__).parent
MODEL_PATH = FIXTURE_DIR / "tiny_trainable.onnx"
CONFIG_PATH = FIXTURE_DIR / "training_config.json"

# Trainable-parameter name substrings (gen_artifacts matches initializer names against these).
REQUIRES_GRAD_SUBSTRINGS = ["weight"]


def build_model() -> onnx.ModelProto:
    # x: [batch, 4] float32 (dynamic batch); loss: scalar float32.
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    loss = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])

    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.standard_normal((4, 1)).astype(np.float32), name="linear.weight")
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="linear.bias")  # frozen

    matmul = helper.make_node("MatMul", ["input", "linear.weight"], ["mm"])
    add = helper.make_node("Add", ["mm", "linear.bias"], ["logits"])
    # ReduceMean over all axes, keepdims=0 -> scalar loss (opset-17 style: axes as attribute).
    reduce_mean = helper.make_node("ReduceMean", ["logits"], ["loss"], keepdims=0)

    graph = helper.make_graph(
        [matmul, add, reduce_mean],
        "tiny_trainable",
        [x],
        [loss],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # onnxruntime-training 1.23 rejects newer IR versions
    onnx.checker.check_model(model, full_check=True)
    return model


def build_config() -> dict:
    return {
        "requires_grad": REQUIRES_GRAD_SUBSTRINGS,
        "peft_mapping": {"linear.weight": "linear.weight"},
        "rank": 8,
        "alpha": 16,
        "peft_target": ["linear"],
        "trainable_parameter_count": 4,
    }


def main() -> None:
    onnx.save(build_model(), MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(build_config(), indent=2) + "\n", encoding="utf-8")
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"wrote {MODEL_PATH} ({size_kb:.1f} KiB) and {CONFIG_PATH}")


if __name__ == "__main__":
    main()

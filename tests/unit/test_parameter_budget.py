"""Unit tests for the export-time parameter-budget gate (`artifacts/parameter_budget.py`).

These build synthetic ORT-training-shaped graphs with `onnx` alone, so they run in the core profile —
the same reason the gate itself counts shapes instead of loading external data.

The gate exists because a byte-count-over-one-dtype slip became a recorded v1 blocker. The dtype-mix
test below is the direct regression for that: a graph whose parameters are mostly uint8 must pass, and
would not if anything in this path assumed 4 bytes per parameter.
"""

from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from mobiletransformers.artifacts.parameter_budget import (
    describe_graph_precision,
    summarize_training_parameters,
    verify_checkpoint_parameter_budget,
)
from mobiletransformers.exceptions import ExportError

#: bytes per element, for sizing the synthetic initializers below.
_ELEM_SIZE = {TensorProto.FLOAT: 4, TensorProto.UINT8: 1, TensorProto.INT64: 8}


def _graph_with_initializers(inits: list[tuple[str, int, tuple[int, ...]]], path) -> str:
    """A graph holding its weights as initializers — the inference-graph shape."""
    tensors = []
    for name, dt, shape in inits:
        count = 1
        for d in shape:
            count *= d
        tensors.append(
            helper.make_tensor(name, dt, list(shape), b"\x00" * (count * _ELEM_SIZE[dt]), raw=True)
        )
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["x"], ["y"])],
        name="inference_model",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b"])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["b"])],
        initializer=tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, str(path))
    return str(path)


def _training_graph(params: list[tuple[str, int, tuple[int, ...]]], path) -> str:
    """A graph shaped like an ORT training model: parameters as fully-shaped inputs, data as symbolic."""
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("labels", TensorProto.INT64, ["batch_size", "sequence_length"]),
    ]
    inputs += [helper.make_tensor_value_info(n, dt, list(shape)) for n, dt, shape in params]

    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["input_ids"], ["loss"])],
        name="training_model",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("loss", TensorProto.INT64, ["batch_size", "sequence_length"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, str(path))
    return str(path)


def test_parameters_are_separated_from_data_by_shape_not_name(tmp_path):
    """Data inputs carry a symbolic dim; parameters are fully concrete. No name matching."""
    path = _training_graph(
        [("embed.weight", TensorProto.FLOAT, (100, 8)), ("layer.0.weight", TensorProto.FLOAT, (8, 8))],
        tmp_path / "training_model.onnx",
    )
    summary = summarize_training_parameters(path)

    assert summary.tensor_count == 2
    assert summary.total == 100 * 8 + 8 * 8
    assert sorted(summary.data_input_names) == ["input_ids", "labels"]


def test_counts_are_per_dtype_so_a_quantized_graph_is_not_undercounted(tmp_path):
    """The regression for the arithmetic that produced a false 'two thirds missing' blocker.

    A graph whose parameters are overwhelmingly uint8 carries exactly as many *parameters* as its
    element count says. Anything that reasoned in bytes-over-fp32 would report a third of this.
    """
    path = _training_graph(
        [
            ("base.weight_quantized", TensorProto.UINT8, (1000, 100)),  # 100_000 uint8
            ("adapter.lora_A.weight", TensorProto.FLOAT, (100, 8)),  # 800 fp32
        ],
        tmp_path / "training_model.onnx",
    )
    summary = summarize_training_parameters(path)

    assert summary.total == 100_800
    assert summary.quantized == 100_000
    assert summary.float_elements == 800
    # The whole point: a naive bytes/4 reading of the same graph would claim ~25_200.
    verify_checkpoint_parameter_budget(path, expected_total=100_800)


def test_budget_passes_within_tolerance(tmp_path):
    """PEFT adds parameters and tied embeddings are shared; small deviation is expected, not fatal."""
    path = _training_graph(
        [("w", TensorProto.FLOAT, (1000, 100))], tmp_path / "training_model.onnx"
    )  # 100_000
    assert verify_checkpoint_parameter_budget(path, expected_total=102_000).total == 100_000


def test_budget_fails_closed_when_the_graph_is_short(tmp_path):
    """The check the pipeline was missing: a graph carrying a fraction of its model must not ship."""
    path = _training_graph([("w", TensorProto.FLOAT, (100, 100))], tmp_path / "training_model.onnx")

    with pytest.raises(ExportError) as excinfo:
        verify_checkpoint_parameter_budget(path, expected_total=135_000_000)

    message = str(excinfo.value)
    assert "10,000" in message and "135,000,000" in message, "both counts must be named"
    assert "%" in message


def test_all_quantized_graph_fails_because_nothing_is_trainable(tmp_path):
    """A quantizer that swept the adapters leaves no float parameter to take a gradient on."""
    path = _training_graph(
        [("base.weight_quantized", TensorProto.UINT8, (1000, 100))], tmp_path / "training_model.onnx"
    )

    with pytest.raises(ExportError, match="NONE in float storage"):
        verify_checkpoint_parameter_budget(path, expected_total=100_000)


def test_missing_reference_skips_loudly_rather_than_passing(tmp_path, caplog):
    """A package exported before the gate existed has no reference; that must be said, not assumed."""
    path = _training_graph([("w", TensorProto.FLOAT, (10, 10))], tmp_path / "training_model.onnx")

    summary = verify_checkpoint_parameter_budget(path, expected_total=None)

    assert summary.total == 100
    assert "UNVERIFIED" in caplog.text


def test_graph_with_no_parameter_inputs_fails(tmp_path):
    path = _training_graph([], tmp_path / "training_model.onnx")
    with pytest.raises(ExportError, match="no fully-shaped parameter inputs"):
        summarize_training_parameters(path)


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(ExportError, match="not found"):
        summarize_training_parameters(tmp_path / "absent.onnx")


def test_precision_is_measured_not_taken_from_the_variant_name(tmp_path):
    """The `cpu-int4` variant shipped an fp32 inference graph; only measurement catches that."""
    fp32 = _graph_with_initializers([("w", TensorProto.FLOAT, (4, 4))], tmp_path / "fp32.onnx")
    assert describe_graph_precision(fp32) == "float32"

    quantized = _graph_with_initializers(
        [("w_quantized", TensorProto.UINT8, (16, 16)), ("w_scale", TensorProto.FLOAT, (16,))],
        tmp_path / "mixed.onnx",
    )
    # uint8 dominates by element count, so it leads.
    assert describe_graph_precision(quantized) == "mixed(uint8/float32)"


def test_shape_constants_do_not_make_a_float_graph_look_mixed(tmp_path):
    """int64 `Reshape` targets are not weights; counting them reported fp32 graphs as mixed."""
    path = _graph_with_initializers(
        [("w", TensorProto.FLOAT, (8, 8)), ("reshape_target", TensorProto.INT64, (2,))],
        tmp_path / "with_shapes.onnx",
    )
    assert describe_graph_precision(path) == "float32"


def test_precision_of_a_graph_with_no_weights_is_unknown_not_guessed(tmp_path):
    path = _graph_with_initializers([], tmp_path / "empty.onnx")
    assert describe_graph_precision(path) == "unknown"

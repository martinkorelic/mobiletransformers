"""Regression for the LayerNorm gradient-output patch (#33, `artifacts/builder.py`).

ORT's `LayerNormalizationGrad` reads the forward node's optional **second and third outputs** (saved
mean and inverse standard deviation) instead of recomputing them. `torch.onnx` exports the node with
only `Y`, so building a gradient graph through it trips an assertion deep inside ORT that names
neither the op nor the node:

    GradientBuilderBase::O(size_t, bool) const i < node_->OutputDefs().size() was false

Decoders never hit it — Llama-family RMSNorm exports as `SimplifiedLayerNormalization`, which already
carries 2 outputs — so encoder support was the first thing to need this. These tests run on `onnx`
alone (core profile); the end-to-end proof is the env-gated encoder integration test.
"""

from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from mobiletransformers.artifacts.graph_prep import ensure_layernorm_grad_outputs


def _model(nodes, path) -> str:
    graph = helper.make_graph(
        nodes=nodes,
        name="g",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b", 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["b", 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [4], b"\x00" * 16, raw=True),
            helper.make_tensor("bias", TensorProto.FLOAT, [4], b"\x00" * 16, raw=True),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, str(path))
    return str(path)


def _layernorm(name="ln", outputs=("y",)):
    return helper.make_node("LayerNormalization", ["x", "scale", "bias"], list(outputs), name=name)


def test_single_output_layernorm_gains_mean_and_inv_std(tmp_path):
    src = _model([_layernorm()], tmp_path / "m.onnx")

    out = ensure_layernorm_grad_outputs(src)

    assert out != src, "a rewrite was needed, so a new path must be returned"
    node = onnx.load(out).graph.node[0]
    assert len(node.output) == 3
    # Order is fixed by the ONNX spec: Y, Mean, InvStdDev. Y must not move.
    assert node.output[0] == "y"
    assert node.output[1] != node.output[2]


def test_rewrite_is_written_beside_the_source_so_external_data_still_resolves(tmp_path):
    """External-data references are relative to the model file; a temp dir elsewhere would break them."""
    src = _model([_layernorm()], tmp_path / "quant_model.onnx")

    out = ensure_layernorm_grad_outputs(src)

    assert str(tmp_path) == str(__import__("pathlib").Path(out).parent)


def test_already_complete_layernorm_is_left_alone(tmp_path):
    """No rewrite, and the original path is returned unchanged — the common case must stay free."""
    src = _model([_layernorm(outputs=("y", "mean", "inv_std"))], tmp_path / "m.onnx")

    assert ensure_layernorm_grad_outputs(src) == src


def test_decoder_style_norm_is_untouched(tmp_path):
    """`SimplifiedLayerNormalization` (RMSNorm) already carries its 2 outputs and is a different op.

    This is what makes the patch a no-op on every decoder package that already works.
    """
    node = helper.make_node("SimplifiedLayerNormalization", ["x", "scale"], ["y", "inv_std"], name="rms")
    src = _model([node], tmp_path / "m.onnx")

    assert ensure_layernorm_grad_outputs(src) == src


def test_every_single_output_layernorm_is_patched(tmp_path):
    src = _model(
        [_layernorm(name=f"ln{i}", outputs=(f"t{i}",)) for i in range(4)] + [_layernorm(name="last")],
        tmp_path / "m.onnx",
    )

    out = ensure_layernorm_grad_outputs(src)

    model = onnx.load(out)
    assert all(len(n.output) == 3 for n in model.graph.node)
    names = [o for n in model.graph.node for o in n.output]
    assert len(names) == len(set(names)), "generated output names must be unique across nodes"


@pytest.mark.parametrize("outputs", [("y",), ("y", "mean")])
def test_patching_is_idempotent(tmp_path, outputs):
    src = _model([_layernorm(outputs=outputs)], tmp_path / "m.onnx")

    once = ensure_layernorm_grad_outputs(src)
    twice = ensure_layernorm_grad_outputs(once)

    assert twice == once, "a patched graph needs no second rewrite"
    assert len(onnx.load(once).graph.node[0].output) == 3

"""#9: re-exporting into an existing package directory must not corrupt its external data.

`onnx.write_external_data_tensors` APPENDS to an existing blob and records each tensor's
`(offset, length)` into the graph. A second export into the same directory therefore doubles every
per-tensor `.bin` and points the graph at the second copy. The result is self-consistent as an ONNX
model, so nothing on the host complains — but #23's on-disk contract is one raw tensor per file at
offset 0, and the device rejects it with a size mismatch at load. This test pins the invariant that
made that failure reachable at all.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from mobiletransformers.export.inference_package import _split_external_data


def _tiny_model() -> onnx.ModelProto:
    w = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), name="layer.MatMul.weight")
    frozen = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="frozen.weight")
    node = helper.make_node("Identity", ["layer.MatMul.weight"], ["out"])
    graph = helper.make_graph(
        [node],
        "g",
        [],
        [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [2, 3])],
        initializer=[w, frozen],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def test_repeated_split_leaves_one_tensor_per_bin(tmp_path):
    trainable = {"layer.MatMul.weight"}
    expected = 2 * 3 * 4  # 6 float32 elements

    for _ in range(3):
        _split_external_data(_tiny_model(), tmp_path, trainable)

        blob = tmp_path / "layer.MatMul.weight.bin"
        assert blob.stat().st_size == expected, (
            f"per-tensor blob grew to {blob.stat().st_size} bytes after a re-export "
            "(onnx appended instead of replacing)"
        )

    # The graph must reference the tensor at offset 0 — the device reads the whole file as one tensor.
    model = _tiny_model()
    _split_external_data(model, tmp_path, trainable)
    init = next(i for i in model.graph.initializer if i.name == "layer.MatMul.weight")
    entries = {e.key: e.value for e in init.external_data}
    assert entries.get("offset", "0") == "0"
    assert entries["length"] == str(expected)


def test_repeated_split_does_not_grow_the_frozen_base_blob(tmp_path):
    trainable = {"layer.MatMul.weight"}
    sizes = []
    for _ in range(3):
        _split_external_data(_tiny_model(), tmp_path, trainable)
        sizes.append((tmp_path / "frozen_base.onnx.data").stat().st_size)
    assert len(set(sizes)) == 1, f"frozen base blob grew across re-exports: {sizes}"

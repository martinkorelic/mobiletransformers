"""The upstream external-data blob must not survive into the shipped package.

Optimum writes every weight to one ``model.onnx_data`` beside ``model.onnx``. ``_split_external_data``
then re-points **every** initializer at ``frozen_base.onnx.data`` or a per-tensor ``<name>.bin`` — so
the upstream blob ends up referenced by nothing and was simply left in the package.

Nothing caught it, and that is the interesting part: a package with a spare file still passes every
gate there is. The graph is self-consistent, `validate` resolves every declared file, and the
checksums match. Only the **download** notices, because `downloadPlan` ships the inference stage as
the glob ``variants/<id>/inference/**``:

    FunctionGemma   1,743 MB dead of 3,875 MB   (45%)  <- already published to the Hub
    SmolLM2-135M      651 MB dead of 1,586 MB   (41%)

So every user paid nearly double the download for every package. These tests pin the removal, and pin
that a **referenced** blob is never touched — a package that legitimately keeps the upstream layout
must degrade to a no-op, not to a corrupt graph.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mobiletransformers.export.inference_package import _drop_orphaned_external_data


def _model_with_external_weight(location: str) -> onnx.ModelProto:
    """A one-initializer graph whose weight lives in ``location``."""
    weight = numpy_helper.from_array(np.zeros((4, 4), dtype=np.float32), name="w")
    weight.data_location = TensorProto.EXTERNAL
    del weight.external_data[:]
    for key, value in (("location", location), ("offset", "0"), ("length", "64")):
        entry = weight.external_data.add()
        entry.key, entry.value = key, value
    weight.ClearField("raw_data")

    node = helper.make_node("Identity", ["w"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])],
        initializer=[weight],
    )
    return helper.make_model(graph)


def test_an_unreferenced_upstream_blob_is_removed(tmp_path):
    """The defect: 45% of every published package was this file."""
    model = _model_with_external_weight("frozen_base.onnx.data")
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    (tmp_path / "frozen_base.onnx.data").write_bytes(b"\0" * 64)
    orphan = tmp_path / "model.onnx_data"
    orphan.write_bytes(b"\0" * 4096)

    assert _drop_orphaned_external_data(path, tmp_path) == 1
    assert not orphan.exists()
    # The blob the graph actually uses survives — the whole package depends on it.
    assert (tmp_path / "frozen_base.onnx.data").exists()


def test_a_referenced_blob_is_never_removed(tmp_path):
    """Fail-safe: if a layout genuinely keeps the upstream blob, this must be a no-op."""
    model = _model_with_external_weight("model.onnx_data")
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    referenced = tmp_path / "model.onnx_data"
    referenced.write_bytes(b"\0" * 64)

    assert _drop_orphaned_external_data(path, tmp_path) == 0
    assert referenced.exists(), "removing a REFERENCED blob would corrupt the package"


def test_our_own_blobs_are_out_of_scope(tmp_path):
    """Only `*.onnx_data` is considered. `frozen_base.onnx.data` and `*.bin` are ours, by name."""
    model = _model_with_external_weight("frozen_base.onnx.data")
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    (tmp_path / "frozen_base.onnx.data").write_bytes(b"\0" * 64)
    # Unreferenced, but not named `.onnx_data` — deliberately left alone.
    stray = tmp_path / "model.layers.0.self_attn.q_proj.MatMul.weight.bin"
    stray.write_bytes(b"\0" * 16)

    _drop_orphaned_external_data(path, tmp_path)
    assert stray.exists()


def test_nothing_to_do_is_not_an_error(tmp_path):
    model = _model_with_external_weight("frozen_base.onnx.data")
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    (tmp_path / "frozen_base.onnx.data").write_bytes(b"\0" * 64)
    assert _drop_orphaned_external_data(path, tmp_path) == 0


@pytest.mark.parametrize("count", [1, 3])
def test_several_orphans_are_all_removed(tmp_path, count):
    model = _model_with_external_weight("frozen_base.onnx.data")
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    (tmp_path / "frozen_base.onnx.data").write_bytes(b"\0" * 64)
    for i in range(count):
        (tmp_path / f"decoder{i}.onnx_data").write_bytes(b"\0" * 128)
    assert _drop_orphaned_external_data(path, tmp_path) == count

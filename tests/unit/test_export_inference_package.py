"""Unified inference-export package builder (#9): base/trainable external split + handoff-map emit.

onnx-only (core env). Builds a synthetic Llama-shaped graph with one adapted MatMul + one frozen base
tensor, exports the package, and asserts the flat layout, the handoff map (which self-validates on load),
per-tensor checksums, and the emitted merger model.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from inference.export_inference_package import FROZEN_BASE_BLOB, export_inference_package
from onnx import TensorProto, helper, numpy_helper

from mobiletransformers.artifacts.handoff_map import HandoffMap
from mobiletransformers.config.constants import HandoffMode, PEFTMethod
from mobiletransformers.exceptions import ExportError

TRAINABLE_WEIGHT = "model.layers.0.attn.q_proj.MatMul.weight"
FROZEN_WEIGHT = "model.embed_tokens.weight"
BASE_LAYER = "backbone.model.layers.0.self_attn.q_proj"


class _Config:
    architectures = ["LlamaForCausalLM"]


def _build_input_model(path) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 4])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((4, 3)).astype(np.float32), name=TRAINABLE_WEIGHT)
    bias = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name=FROZEN_WEIGHT)
    mm = helper.make_node("MatMul", ["X", TRAINABLE_WEIGHT], ["mm"])
    add = helper.make_node("Add", ["mm", FROZEN_WEIGHT], ["logits"])
    graph = helper.make_graph([mm, add], "tiny", [x], [logits], initializer=[w, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def _training_config() -> dict:
    return {
        "requires_grad": ["lora"],
        "peft_mapping": {BASE_LAYER: {"adapter_A": "lora_A", "adapter_B": "lora_B"}},
    }


def test_export_produces_flat_package(tmp_path):
    src = tmp_path / "src_model.onnx"
    _build_input_model(src)
    out = tmp_path / "inference"

    pkg = export_inference_package(
        model_path=src,
        output_dir=out,
        training_config=_training_config(),
        model_config=_Config(),
        peft_method=PEFTMethod.LORA,
        quant_in=False,
        quant_out=False,
    )

    # Flat layout: model.onnx, one trainable .bin (+.sha256), frozen base blob, handoff map, merger onnx.
    assert pkg.model_path.exists()
    assert len(pkg.trainable_bins) == 1
    bin_path = pkg.trainable_bins[0]
    assert bin_path.name == f"{TRAINABLE_WEIGHT}.bin"
    assert bin_path.exists()
    assert (bin_path.parent / (bin_path.name + ".sha256")).exists()
    assert pkg.frozen_base_blob is not None and pkg.frozen_base_blob.name == FROZEN_BASE_BLOB
    assert pkg.frozen_base_blob.exists()

    # Merger model emitted and recorded under its variant tag.
    assert pkg.merger_models  # {"lora": "merger_lora_fpin_fpout.onnx"}
    for filename in pkg.merger_models.values():
        assert (out / filename).exists()


def test_handoff_map_is_valid_and_keys_observed_names(tmp_path):
    src = tmp_path / "src_model.onnx"
    _build_input_model(src)
    out = tmp_path / "inference"
    export_inference_package(
        model_path=src,
        output_dir=out,
        training_config=_training_config(),
        model_config=_Config(),
        peft_method=PEFTMethod.LORA,
        quant_in=False,
        quant_out=False,
    )

    # load() runs check_compat + validate(): a bad map raises here.
    handoff = HandoffMap.load(out / "weight_handoff_map.json")
    assert len(handoff.entries) == 1
    entry = handoff.entries[0]
    assert entry.inference_initializer_names["weight"] == TRAINABLE_WEIGHT
    assert entry.external_data_location["weight"] == f"{TRAINABLE_WEIGHT}.bin"
    # sha256 recorded matches the sidecar written for the .bin.
    sidecar = (out / f"{TRAINABLE_WEIGHT}.bin.sha256").read_text().strip()
    assert entry.sha256["weight"] == sidecar


def test_frozen_base_is_not_a_trainable_bin(tmp_path):
    src = tmp_path / "src_model.onnx"
    _build_input_model(src)
    out = tmp_path / "inference"
    pkg = export_inference_package(
        model_path=src,
        output_dir=out,
        training_config=_training_config(),
        model_config=_Config(),
        peft_method=PEFTMethod.LORA,
        quant_in=False,
        quant_out=False,
    )
    # The frozen tensor must never be one of the per-tensor trainable files.
    assert all(FROZEN_WEIGHT not in b.name for b in pkg.trainable_bins)
    assert not (out / f"{FROZEN_WEIGHT}.bin").exists()


def test_model_input_mode_fails_closed(tmp_path):
    src = tmp_path / "src_model.onnx"
    _build_input_model(src)
    with pytest.raises(NotImplementedError, match="not supported in v1"):
        export_inference_package(
            model_path=src,
            output_dir=tmp_path / "inference",
            training_config=_training_config(),
            model_config=_Config(),
            peft_method=PEFTMethod.LORA,
            quant_in=False,
            quant_out=False,
            handoff_mode=HandoffMode.MODEL_INPUT,
        )


def test_naming_drift_fails_closed(tmp_path):
    """A peft_mapping base layer with no matching inference initializer aborts (no silent fallback)."""
    src = tmp_path / "src_model.onnx"
    _build_input_model(src)
    cfg = {"requires_grad": ["lora"], "peft_mapping": {"backbone.model.layers.9.self_attn.k_proj": {}}}
    with pytest.raises(ExportError, match="naming drifted"):
        export_inference_package(
            model_path=src,
            output_dir=tmp_path / "inference",
            training_config=cfg,
            model_config=_Config(),
            peft_method=PEFTMethod.LORA,
            quant_in=False,
            quant_out=False,
        )

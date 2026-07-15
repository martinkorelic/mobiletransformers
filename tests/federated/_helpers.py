"""Build a tiny two-entry HandoffMap + matching arrays for #35 federated tests (pure numpy)."""

from __future__ import annotations

import numpy as np

from mobiletransformers.artifacts.handoff_map import HandoffEntry, HandoffMap

_NAME_A = "model.layers.0.attn.q_proj.MatMul.weight"
_NAME_B = "model.layers.1.attn.v_proj.MatMul.weight"


def make_handoff() -> HandoffMap:
    """A deterministic 2-trainable-tensor handoff map (both float32, distinct shapes)."""
    entry_a = HandoffEntry(
        training_base_layer_name="backbone.model.layers.0.self_attn.q_proj.base_layer",
        dtype="float32",
        shape=(2, 3),
        checkpoint_names={"weight": "l0.weight"},
        merger_output_names={"weight": "merged_weight"},
        merged_tensor_names={"weight": _NAME_A},
        inference_initializer_names={"weight": _NAME_A},
        external_data_location={"weight": f"{_NAME_A}.bin"},
    )
    entry_b = HandoffEntry(
        training_base_layer_name="backbone.model.layers.1.self_attn.v_proj.base_layer",
        dtype="float32",
        shape=(3,),
        checkpoint_names={"weight": "l1.weight"},
        merger_output_names={"weight": "merged_weight"},
        merged_tensor_names={"weight": _NAME_B},
        inference_initializer_names={"weight": _NAME_B},
        external_data_location={"weight": f"{_NAME_B}.bin"},
    )
    return HandoffMap(entries=[entry_a, entry_b])


def make_arrays(scale: float = 1.0) -> list[np.ndarray]:
    """Arrays matching the codec order of :func:`make_handoff` (sorted by canonical weight name)."""
    # Sorted order: _NAME_A ("model.layers.0...") < _NAME_B ("model.layers.1...").
    a = np.arange(6, dtype=np.float32).reshape(2, 3) * scale
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32) * scale
    return [a, b]

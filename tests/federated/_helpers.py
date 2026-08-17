"""Build a tiny two-layer HandoffMap + matching arrays for #35 federated tests (pure numpy).

**Rank-r shaped as of #35's vocabulary decision (2026-08-09).** Federation exchanges the adapter
FACTORS (`lora_A`/`lora_B`), not the merged inference initializers, so the fixture describes two
adapted layers with their factor dtypes/shapes rather than two full-size merged weights.

Each layer is deliberately given a different `(rank, in, out)` so a codec that silently transposed or
reordered anything would produce a shape error rather than a plausible-looking record.
"""

from __future__ import annotations

import numpy as np

from mobiletransformers.artifacts.handoff_map import HandoffEntry, HandoffMap

_NAME_A = "model.layers.0.attn.q_proj.MatMul.weight"
_NAME_B = "model.layers.1.attn.v_proj.MatMul.weight"

#: Layer 0: rank 2, in 3, out 4.  Layer 1: rank 2, in 5, out 6.
_A0_SHAPE, _B0_SHAPE = (2, 3), (4, 2)
_A1_SHAPE, _B1_SHAPE = (2, 5), (6, 2)


def _entry(
    name: str,
    layer: int,
    merged_shape: tuple[int, ...],
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
) -> HandoffEntry:
    return HandoffEntry(
        training_base_layer_name=(
            f"backbone.model.layers.{layer}.self_attn.{'q_proj' if layer == 0 else 'v_proj'}.base_layer"
        ),
        dtype="float32",
        shape=merged_shape,
        checkpoint_names={
            "adapter_A": f"l{layer}.lora_A.lora",
            "adapter_B": f"l{layer}.lora_B.lora",
            "weight": f"l{layer}.weight",
        },
        # Schema 1.1: the map DESCRIBES the factors, it does not merely name them.
        adapter_dtypes={"adapter_A": "float32", "adapter_B": "float32"},
        adapter_shapes={"adapter_A": a_shape, "adapter_B": b_shape},
        merger_output_names={"weight": "merged_weight"},
        merged_tensor_names={"weight": name},
        inference_initializer_names={"weight": name},
        external_data_location={"weight": f"{name}.bin"},
    )


def make_handoff() -> HandoffMap:
    """A deterministic 2-layer / 4-factor handoff map (all float32, all shapes distinct)."""
    return HandoffMap(
        entries=[
            _entry(_NAME_A, 0, (4, 3), _A0_SHAPE, _B0_SHAPE),
            _entry(_NAME_B, 1, (6, 5), _A1_SHAPE, _B1_SHAPE),
        ]
    )


def make_arrays(scale: float = 1.0) -> list[np.ndarray]:
    """Arrays in codec order for :func:`make_handoff`.

    Order: entries sorted by canonical weight name (`layers.0` < `layers.1`), and within an entry the
    canonical `HandoffEntry.ADAPTER_ROLE_ORDER` (`adapter_A` before `adapter_B`).
    """
    arrays = []
    for shape in (_A0_SHAPE, _B0_SHAPE, _A1_SHAPE, _B1_SHAPE):
        size = int(np.prod(shape))
        arrays.append(np.arange(size, dtype=np.float32).reshape(shape) * scale)
    return arrays

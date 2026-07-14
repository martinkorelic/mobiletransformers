"""Build synthetic materialized-cache dirs for #22 adapter tests (the tiny_package train/ is too thin)."""

from __future__ import annotations

import json
from pathlib import Path

from mobiletransformers.artifacts.handoff_map import HandoffEntry, HandoffMap

TRAINABLE = "model.layers.0.attn.q_proj.MatMul.weight"
BASE_LAYER = "backbone.model.layers.0.self_attn.q_proj.base_layer"


def make_cache(
    root: Path,
    *,
    peft_method: str,
    component_roles: dict[str, str],
    rank: int | None = 8,
    alpha: float | None = 16.0,
    model_id: str = "org/base-model",
) -> Path:
    """Create ``<root>/{train,inference}`` with a training_config, handoff map, and one merged .bin.

    ``component_roles`` maps adapter-role -> checkpoint tensor name (e.g. {"adapter_A": "...lora_A"}).
    Include them to simulate a checkpoint that still carries the A/B factors (Mode-1 eligible).
    """
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "inference").mkdir(parents=True, exist_ok=True)

    checkpoint_names = {"weight": f"{BASE_LAYER}.weight", **component_roles}
    entry = HandoffEntry(
        training_base_layer_name=BASE_LAYER,
        dtype="float32",
        shape=(4, 3),
        checkpoint_names=checkpoint_names,
        merger_output_names={"weight": "merged_weight"},
        merged_tensor_names={"weight": TRAINABLE},
        inference_initializer_names={"weight": TRAINABLE},
        external_data_location={"weight": f"{TRAINABLE}.bin"},
    )
    (root / "train" / "weight_handoff_map.json").write_text(HandoffMap(entries=[entry]).to_json())

    cfg: dict = {
        "modelId": model_id,
        "peftMethod": peft_method,
        "peft_target": ["q_proj"],
        "trainable_parameter_count": 42,
    }
    if rank is not None:
        cfg["rank"] = rank
    if alpha is not None:
        cfg["alpha"] = alpha
    if peft_method == "mars":
        cfg["optimization_level"] = 2
    (root / "train" / "training_config.json").write_text(json.dumps(cfg, indent=2))

    (root / "inference" / f"{TRAINABLE}.bin").write_text("MERGED_TENSOR_BYTES\n")
    (root / "inference" / f"{TRAINABLE}.bin.sha256").write_text("0" * 64 + "\n")
    return root

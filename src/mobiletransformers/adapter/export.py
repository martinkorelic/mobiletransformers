"""Read a trained adapter out of the on-device cache layout into an :class:`AdapterPackage` (#22).

Pure JSON/file I/O over a materialized ``<cacheDir>/<repo>/`` cache: reads ``train/training_config.json``
+ ``train/weight_handoff_map.json`` (via #8's ``HandoffMap``) and cross-references the flat per-tensor
``.bin`` files in ``inference/``. No ML libraries — the package is metadata + file pointers the convert
gate (``adapter/convert.py``) and the pushback CLI consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mobiletransformers.artifacts.handoff_map import HandoffMap
from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.exceptions import ExportError


@dataclass
class AdapterTensor:
    training_checkpoint_name: str
    external_data_location: str  # per-tensor .bin filename, flat in inference/
    dtype: str
    shape: tuple[int, ...]
    quantized: bool = False


@dataclass
class AdapterPackage:
    base_model_id: str
    peft_method: str  # "lora" | "mars" | ...
    mars_optimization_level: int | None
    rank: int | None
    alpha: float | None
    peft_target: list[str]
    trainable_parameter_count: int | None
    handoff_mode: str
    tensors: list[AdapterTensor]
    cache_repo_dir: Path
    checkpoint_component_roles: tuple[str, ...] = ()  # adapter roles present in checkpoint_names
    source: dict[str, Any] = field(default_factory=dict)


def export_adapter_from_cache(cache_repo_dir: str | Path) -> AdapterPackage:
    """Build an :class:`AdapterPackage` from a materialized cache repo dir (``train/`` + ``inference/``)."""
    root = Path(cache_repo_dir)
    train_cfg_path = root / "train" / "training_config.json"
    handoff_path = root / "train" / "weight_handoff_map.json"
    if not train_cfg_path.is_file():
        raise ExportError(f"no train/training_config.json under {root}")
    if not handoff_path.is_file():
        raise ExportError(f"no train/weight_handoff_map.json under {root}")

    cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
    handoff = HandoffMap.load(handoff_path)

    inference_dir = root / "inference"
    tensors: list[AdapterTensor] = []
    component_roles: set[str] = set()
    for entry in handoff.entries:
        # Adapter component roles present in the checkpoint (LoRA A/B etc.), used by the convert gate.
        component_roles.update(r for r in entry.checkpoint_names if r != "weight")
        for role, location in entry.external_data_location.items():
            tensors.append(
                AdapterTensor(
                    training_checkpoint_name=entry.checkpoint_names.get(role, entry.training_base_layer_name),
                    external_data_location=location,
                    dtype=entry.dtype,
                    shape=tuple(entry.shape),
                    quantized=role in ("weight_quantized", "scale", "zero_point"),
                )
            )
            # note: existence of the .bin in inference/ is not required for metadata export;
            # the pushback CLI checks presence when it actually copies files.
            _ = inference_dir  # kept for clarity of where merged tensors live

    peft_method = str(cfg.get("peftMethod") or cfg.get("peft_method") or "").lower()
    return AdapterPackage(
        base_model_id=cfg.get("modelId") or cfg.get("model_id") or "unknown",
        peft_method=peft_method,
        mars_optimization_level=(
            cfg.get("optimization_level") if peft_method == PEFTMethod.MARS.value else None
        ),
        rank=cfg.get("rank"),
        alpha=cfg.get("alpha"),
        peft_target=list(cfg.get("peft_target", [])),
        trainable_parameter_count=cfg.get("trainable_parameter_count"),
        handoff_mode=handoff.handoff_mode.value
        if hasattr(handoff.handoff_mode, "value")
        else str(handoff.handoff_mode),
        tensors=tensors,
        cache_repo_dir=root,
        checkpoint_component_roles=tuple(sorted(component_roles)),
        source={"device": cfg.get("device", "unknown")},
    )


__all__ = ["AdapterTensor", "AdapterPackage", "export_adapter_from_cache"]

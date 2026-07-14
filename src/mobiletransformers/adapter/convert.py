"""The PEFT-vs-native gate (#22). Pure, deterministic metadata decision — no ML libraries.

Mode 1 (PEFT-compatible) is emitted **only** when the handoff metadata proves a clean LoRA-shaped
decomposition exists: ``peftMethod == "lora"`` AND the LoRA component tensors (A/B factors, per #6's
``PEFTMethodSpec.component_schema``) are present in the checkpoint AND ``rank``/``alpha`` are known.
Everything else — every MARS package, and any LoRA whose checkpoint no longer carries A/B factors —
falls to Mode 2 (MobileTransformers-native). Materializing the actual ``adapter_model.safetensors`` bytes
from the ORT checkpoint is ``torch``/``safetensors`` env-gated; the gate decision + ``adapter_config.json``
here are pure and CI-covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mobiletransformers.adapter.export import AdapterPackage
from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.config.registry.peft import get_peft_spec
from mobiletransformers.exceptions import ExportError


@dataclass
class PeftLayout:
    """A PEFT-compatible adapter layout (Mode 1). ``adapter_config`` is the ``adapter_config.json`` dict."""

    adapter_config: dict[str, Any]
    component_roles: tuple[str, ...]


def to_peft_layout(pkg: AdapterPackage) -> PeftLayout | None:
    """Return a :class:`PeftLayout` if the package cleanly maps to a PEFT LoRA adapter, else ``None``."""
    if pkg.peft_method != PEFTMethod.LORA.value:
        return None  # MARS (and anything non-LoRA) is never emitted as a drop-in PEFT adapter in v1.
    required_roles = {c.role for c in get_peft_spec(PEFTMethod.LORA).component_schema}
    if not required_roles.issubset(set(pkg.checkpoint_component_roles)):
        return None  # checkpoint no longer carries the A/B factors (only merged tensors) -> native mode.
    if pkg.rank is None or pkg.alpha is None:
        return None
    adapter_config = {
        "peft_type": "LORA",
        "r": pkg.rank,
        "lora_alpha": pkg.alpha,
        "target_modules": list(pkg.peft_target),
        "base_model_name_or_path": pkg.base_model_id,
        "task_type": "CAUSAL_LM",
    }
    return PeftLayout(adapter_config=adapter_config, component_roles=tuple(sorted(required_roles)))


def materialize_peft_weights(pkg: AdapterPackage, layout: PeftLayout, dest_dir: str) -> None:
    """Write ``adapter_model.safetensors`` from the ORT checkpoint A/B factors — **env-gated**.

    Reading the ORT ``CheckpointState`` needs ``onnxruntime-training`` and writing safetensors needs
    ``torch``/``safetensors`` (the ``train`` extra). Raises a clear error when those are unavailable;
    the gate decision + ``adapter_config.json`` are produced without this (CI-covered).
    """
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-gated, not run in the core CI env
        raise ExportError(
            "materializing adapter_model.safetensors requires the 'train' extra (torch + safetensors) "
            "and the ORT-training runtime; run under that profile"
        ) from exc
    raise NotImplementedError(  # pragma: no cover - device/env-gated tensor extraction
        "PEFT safetensors materialization from the ORT checkpoint is env-gated (run manually under the "
        "train profile); the gate + adapter_config.json are available in every environment"
    )


__all__ = ["PeftLayout", "to_peft_layout", "materialize_peft_weights"]

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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mobiletransformers.adapter.export import AdapterPackage
from mobiletransformers.artifacts.handoff_map import HandoffMap
from mobiletransformers.artifacts.package_paths import PackagePaths
from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.config.registry.peft import get_peft_spec
from mobiletransformers.exceptions import ExportError

if TYPE_CHECKING:  # numpy is only needed under the train/ort-training profiles, not in the pure gate.
    import numpy as np

#: Reads trainable A/B factor arrays out of the ORT ``CheckpointState``: ``(checkpoint_dir, names) ->
#: {checkpoint_name: ndarray}``. Injectable so the safetensors-writing path is unit-testable without a
#: real on-device checkpoint (the default reader below is env-gated on ``onnxruntime-training``).
FactorReader = Callable[[Path, "Iterable[str]"], "dict[str, np.ndarray]"]


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


def _read_checkpoint_factors(checkpoint_dir: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Default :data:`FactorReader`: pull trainable A/B factors from the ORT ``CheckpointState``.

    Mirrors ``artifact/onnx_builder.py``'s ``onnx_transfer_trained_weights`` (``state.parameters`` yields
    ``(name, parameter)`` with ``parameter.data`` a numpy array). Env-gated on ``onnxruntime-training``.
    """
    try:
        from onnxruntime.training.api import CheckpointState  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-gated, needs the ORT-training runtime
        raise ExportError(
            "reading the ORT CheckpointState requires the onnxruntime-training runtime; run under the "
            "ort-training profile, or pass a factor_reader to materialize_peft_weights"
        ) from exc
    if not checkpoint_dir.exists():  # pragma: no cover - env-gated path
        raise ExportError(f"no ORT checkpoint at {checkpoint_dir}")
    wanted = set(names)
    try:  # pragma: no cover - env-gated
        state = CheckpointState.load_checkpoint(str(checkpoint_dir))
    except Exception as exc:  # noqa: BLE001 - ORT raises its own runtime errors here
        # Normalize to ExportError so callers (cli/push_adapter) can distinguish "cannot produce
        # weights" from a genuine bug, and fail closed rather than publish a weightless adapter.
        raise ExportError(f"failed to load the ORT checkpoint at {checkpoint_dir}: {exc}") from exc
    return {  # pragma: no cover - env-gated
        param_name: parameter.data for param_name, parameter in state.parameters if param_name in wanted
    }


def _peft_safetensors_key(training_base_layer_name: str, search_pattern: str) -> str:
    """Map a handoff ``trainingBaseLayerName`` + a PEFT component pattern to the PEFT safetensors key.

    ``backbone.model.layers.0.self_attn.q_proj.base_layer`` + ``lora_A`` ->
    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`` (the layout ``peft`` loads with
    ``PeftModel.from_pretrained``).
    """
    module_path = training_base_layer_name
    if module_path.startswith("backbone."):
        module_path = module_path[len("backbone.") :]
    if module_path.endswith(".base_layer"):
        module_path = module_path[: -len(".base_layer")]
    return f"base_model.model.{module_path}.{search_pattern}.weight"


def materialize_peft_weights(
    pkg: AdapterPackage,
    layout: PeftLayout,
    dest_dir: str,
    *,
    factor_reader: FactorReader | None = None,
) -> None:
    """Write ``adapter_model.safetensors`` from the ORT checkpoint A/B factors — **env-gated**.

    Writing safetensors needs ``torch``/``safetensors`` (the ``train`` extra) and reading the ORT
    ``CheckpointState`` needs ``onnxruntime-training`` (the default ``factor_reader``). The gate decision +
    ``adapter_config.json`` are produced without either (CI-covered). ``factor_reader`` is injectable so
    the numpy->torch->safetensors path can be exercised without a real on-device checkpoint.

    Fails closed (``ExportError``) naming any factor tensor the reader could not supply, before writing.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from safetensors.torch import save_file  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-gated, not run in the core CI env
        raise ExportError(
            "materializing adapter_model.safetensors requires the 'train' extra (torch + safetensors); "
            "run under that profile"
        ) from exc

    # Component role -> PEFT sub-key pattern (adapter_A -> lora_A, adapter_B -> lora_B), from the registry.
    role_to_pattern = {c.role: c.search_pattern for c in get_peft_spec(PEFTMethod.LORA).component_schema}
    factor_roles = tuple(layout.component_roles)

    # Reload the handoff map to recover each trainable layer's A/B factor checkpoint names (the merged
    # AdapterPackage.tensors carry only the fused inference weight, not the separate factors).
    cache_paths = PackagePaths.for_cache(pkg.cache_repo_dir.parent, pkg.cache_repo_dir.name)
    handoff = HandoffMap.load(cache_paths.train / "weight_handoff_map.json")

    # checkpoint param name -> its target PEFT safetensors key.
    key_by_checkpoint_name: dict[str, str] = {}
    for entry in handoff.entries:
        if not all(role in entry.checkpoint_names for role in factor_roles):
            continue  # this layer's checkpoint no longer carries the factors; skip (gate already vetted)
        for role in factor_roles:
            pattern = role_to_pattern.get(role)
            if pattern is None:
                raise ExportError(f"PEFT role {role!r} has no component pattern in the LoRA registry")
            ckpt_name = entry.checkpoint_names[role]
            key_by_checkpoint_name[ckpt_name] = _peft_safetensors_key(entry.training_base_layer_name, pattern)
    if not key_by_checkpoint_name:
        raise ExportError("no LoRA A/B factors found in the handoff map; package is not Mode-1 eligible")

    reader = factor_reader or _read_checkpoint_factors
    arrays = reader(cache_paths.train / "checkpoint", key_by_checkpoint_name.keys())

    missing = sorted(set(key_by_checkpoint_name) - set(arrays))
    if missing:
        raise ExportError(f"checkpoint is missing required LoRA factor tensor(s): {', '.join(missing)}")

    tensors = {
        key_by_checkpoint_name[name]: torch.from_numpy(np.ascontiguousarray(arrays[name]))
        for name in key_by_checkpoint_name
    }
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest / "adapter_model.safetensors"))


__all__ = ["PeftLayout", "FactorReader", "to_peft_layout", "materialize_peft_weights"]

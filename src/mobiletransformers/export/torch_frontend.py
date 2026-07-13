"""``torch.onnx`` export frontend — the reserved last-resort training-graph path (Fallback A).

The plan flagged ``torch.onnx`` as Fallback A for when optimum's ``OnnxConfigWithLoss``/``export`` are
gone. The migration spike found ``OnnxConfigWithLoss`` removed but ``export()`` *surviving*, so the
active training path vendors ``OnnxConfigWithLoss`` (see ``onnx_config_with_loss.py``) and stays on the
durable optimum ``export()`` — no manual graph reconstruction required.

This module therefore stays a **declared, fail-closed** registry row: it keeps ``EXPORT_FRONTEND_REGISTRY``
extensible (F3) and documents the escape hatch, but selecting it raises with guidance rather than
shipping an unexercised torch.onnx reconstruction. Wire it up only if a future optimom removes
``export()`` too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from mobiletransformers.exceptions import ExportError


def torch_available() -> bool:
    """Availability probe for the torch.onnx frontend."""
    return importlib.util.find_spec("torch") is not None


def torch_onnx_training_export(
    model: Any,
    out_dir: Path,
    task: str,
    opset: int,
    trust_remote_code: bool,
    token: str | None,
) -> dict[str, str]:
    """Reserved fallback (not active). Fails closed with guidance toward the vendored path."""
    raise ExportError(
        "torch.onnx export frontend is a reserved fallback and is not implemented: the active "
        "training-graph path uses the vendored OnnxConfigWithLoss on optimum's surviving export() "
        "(mobiletransformers.export.onnx_config_with_loss). Only wire this up if optimum removes "
        "export() as well (see spikes/optimum_migration)."
    )


__all__ = ["torch_available", "torch_onnx_training_export"]

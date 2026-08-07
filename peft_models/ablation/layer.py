"""DEPRECATED shim — moved to ``mobiletransformers.peft.ablation.layer`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.ablation.layer import (  # noqa: F401
    AblationLayer,
    Linear,
    ManualQuantizedLinear,
)

warnings.warn(
    "peft_models.ablation.layer moved to mobiletransformers.peft.ablation.layer.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AblationLayer",
    "Linear",
    "ManualQuantizedLinear",
]

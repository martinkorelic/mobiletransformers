"""DEPRECATED shim — moved to ``mobiletransformers.peft.ablation.model`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.ablation.model import (  # noqa: F401
    AblationModel,
)

warnings.warn(
    "peft_models.ablation.model moved to mobiletransformers.peft.ablation.model.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AblationModel",
]

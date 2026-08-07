"""DEPRECATED shim — moved to ``mobiletransformers.peft.ablation.config`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.ablation.config import (  # noqa: F401
    AblationConfig,
    AblationVariant,
)

warnings.warn(
    "peft_models.ablation.config moved to mobiletransformers.peft.ablation.config.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AblationConfig",
    "AblationVariant",
]

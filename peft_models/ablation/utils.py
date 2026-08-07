"""DEPRECATED shim — moved to ``mobiletransformers.peft.ablation.utils`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.ablation.utils import (  # noqa: F401
    TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING,
)

warnings.warn(
    "peft_models.ablation.utils moved to mobiletransformers.peft.ablation.utils.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING",
]

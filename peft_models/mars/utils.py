"""DEPRECATED shim — moved to ``mobiletransformers.peft.mars.utils`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.mars.utils import (  # noqa: F401
    TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING,
)

warnings.warn(
    "peft_models.mars.utils moved to mobiletransformers.peft.mars.utils.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING",
]

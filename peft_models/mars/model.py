"""DEPRECATED shim — moved to ``mobiletransformers.peft.mars.model`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.mars.model import (  # noqa: F401
    MarsModel,
)

warnings.warn(
    "peft_models.mars.model moved to mobiletransformers.peft.mars.model.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MarsModel",
]

"""DEPRECATED shim — moved to ``mobiletransformers.peft.mars.config`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.mars.config import (  # noqa: F401
    MarsConfig,
)

warnings.warn(
    "peft_models.mars.config moved to mobiletransformers.peft.mars.config.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MarsConfig",
]

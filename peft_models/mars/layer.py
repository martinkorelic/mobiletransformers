"""DEPRECATED shim — moved to ``mobiletransformers.peft.mars.layer`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.mars.layer import (  # noqa: F401
    Linear,
    MarsLayer,
    QuantizedBaseLayer,
    SharedAttentionAdapter,
    SharedMLPAdapter,
)

warnings.warn(
    "peft_models.mars.layer moved to mobiletransformers.peft.mars.layer.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Linear",
    "MarsLayer",
    "QuantizedBaseLayer",
    "SharedAttentionAdapter",
    "SharedMLPAdapter",
]

"""DEPRECATED shim — moved to ``mobiletransformers.peft.lora_xs.latent_utils`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.lora_xs.latent_utils import (  # noqa: F401
    forward_latent,
    get_delta_weight,
    transpose,
)

warnings.warn(
    "peft_models.lora_xs.latent_utils moved to mobiletransformers.peft.lora_xs.latent_utils.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "forward_latent",
    "get_delta_weight",
    "transpose",
]

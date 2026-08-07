"""DEPRECATED shim — moved to ``mobiletransformers.peft.lora_xs.svd_utils`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.lora_xs.svd_utils import (  # noqa: F401
    get_linear_rec_svd,
    run_svd,
)

warnings.warn(
    "peft_models.lora_xs.svd_utils moved to mobiletransformers.peft.lora_xs.svd_utils.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "get_linear_rec_svd",
    "run_svd",
]

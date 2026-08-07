"""DEPRECATED shim — moved to ``mobiletransformers.peft.lora_xs.merger`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.lora_xs.merger import (  # noqa: F401
    load_and_merge_lora_model,
    main,
)

warnings.warn(
    "peft_models.lora_xs.merger moved to mobiletransformers.peft.lora_xs.merger.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "load_and_merge_lora_model",
    "main",
]

"""DEPRECATED shim — moved to ``mobiletransformers.peft.lora_xs.initialization_utils`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.lora_xs.initialization_utils import (  # noqa: F401
    find_and_initialize,
    get_replacement_module,
    init_module_weights,
    kaiming_uniform_init,
    kaiming_uniform_init_lower_half,
    replace_module_weights,
    update_decoder_weights,
)

warnings.warn(
    "peft_models.lora_xs.initialization_utils moved to mobiletransformers.peft.lora_xs.initialization_utils.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "find_and_initialize",
    "get_replacement_module",
    "init_module_weights",
    "kaiming_uniform_init",
    "kaiming_uniform_init_lower_half",
    "replace_module_weights",
    "update_decoder_weights",
]

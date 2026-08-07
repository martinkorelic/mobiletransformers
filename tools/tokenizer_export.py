"""DEPRECATED compatibility shim — moved to ``mobiletransformers.export.tokenizer_export`` (S1)."""

import warnings

from mobiletransformers.export.tokenizer_export import (  # noqa: F401
    export_tokenizer_config,
    export_tokenizer_config_advanced,
)

warnings.warn(
    "tools.tokenizer_export moved to mobiletransformers.export.tokenizer_export.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["export_tokenizer_config", "export_tokenizer_config_advanced"]

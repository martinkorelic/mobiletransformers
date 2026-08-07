"""DEPRECATED compatibility shim — moved to ``mobiletransformers.inference.generator`` (S1)."""

import warnings

from mobiletransformers.inference.generator import generate_tokens_onnx  # noqa: F401

warnings.warn(
    "inference.generator moved to mobiletransformers.inference.generator.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["generate_tokens_onnx"]

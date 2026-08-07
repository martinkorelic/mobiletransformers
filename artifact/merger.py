"""DEPRECATED shim — moved to ``mobiletransformers.config.registry.merger`` (Migration Map S5).

The ~20-line ``emit_merger_models`` driver was folded into the merger REGISTRY, which
already owned ``resolve_merger``/``build_merger_model`` — there is no separate merger module now.
"""

import warnings

from mobiletransformers.config.registry.merger import (  # noqa: F401
    emit_merger_models,
)

warnings.warn(
    "artifact.merger moved to mobiletransformers.config.registry.merger.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "emit_merger_models",
]

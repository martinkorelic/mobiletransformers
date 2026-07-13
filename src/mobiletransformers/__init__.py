"""MobileTransformers — export and Android runtime tooling for on-device transformers.

``__all__`` is the SemVer-governed public Python surface (peer to the Kotlin facade and the CLI).
It is intentionally small today and grows as later plans land their public entrypoints
(``export_model``, ``package_model``, ``pull_package``, ``push_adapter``, the typed config models,
the enums, and the registry ``register_*`` helpers). ``public_api.txt`` is a checked-in golden of
this list so accidental surface changes fail the public-API test.
"""

from __future__ import annotations

from mobiletransformers.config import resolve
from mobiletransformers.config.settings import Settings, get_settings
from mobiletransformers.exceptions import (
    ConfigValidationError,
    ExportError,
    HandoffError,
    HubError,
    ManifestError,
    MergeError,
    MobileTransformersError,
    UnsupportedModelError,
)
from mobiletransformers.utils.logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # settings & precedence
    "Settings",
    "get_settings",
    "resolve",
    # logging
    "get_logger",
    "configure_logging",
    # exception hierarchy
    "MobileTransformersError",
    "ConfigValidationError",
    "ExportError",
    "ManifestError",
    "HandoffError",
    "MergeError",
    "UnsupportedModelError",
    "HubError",
]

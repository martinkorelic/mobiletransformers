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


def _read_version() -> str:
    """Resolve the package version from installed metadata (#32).

    ``pyproject.toml`` is the SINGLE write-site for the version; hardcoding it here made a second one
    that could silently disagree. The fallback covers running straight from a source tree with no
    installed distribution, and is only ever a development value.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("mobiletransformers")
    except PackageNotFoundError:  # pragma: no cover - source tree without an install
        return "0.0.0+unknown"


__version__ = _read_version()

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

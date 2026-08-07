"""DEPRECATED compatibility shim — moved to ``mobiletransformers.export.inference_package`` (S2).

The Migration Map targets ``export/inference_export.py`` for this file, but that name is already taken
by the optimum front door (``export_inference``), so it landed as ``export/inference_package.py``.
"""

import warnings

from mobiletransformers.export.inference_package import (  # noqa: F401
    EXTERNAL_INITIALIZERS_FOLDER_KEY,
    FROZEN_BASE_BLOB,
    GENAI_CONFIG_FILENAME,
    HANDOFF_MAP_FILENAME,
    MODEL_FILENAME,
    ExportedPackage,
    export_inference_package,
    logger,
)

warnings.warn(
    "inference.export_inference_package moved to mobiletransformers.export.inference_package.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EXTERNAL_INITIALIZERS_FOLDER_KEY",
    "FROZEN_BASE_BLOB",
    "GENAI_CONFIG_FILENAME",
    "HANDOFF_MAP_FILENAME",
    "MODEL_FILENAME",
    "ExportedPackage",
    "export_inference_package",
    # Module-level logger: part of the original surface, so the shim re-exports it rather than
    # quietly narrowing what `from inference.export_inference_package import *` used to give you.
    "logger",
]

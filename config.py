"""DEPRECATED compatibility shim.

Import from ``mobiletransformers.config.settings`` (secrets) and
``mobiletransformers.config.constants`` (experiment constants) instead. This module re-exports the
legacy names so existing imports (e.g. ``from config import AZURE_API_VERSION``,
``from config import TASK_EPOCHS``) keep working during migration.

Non-circular: this module is named ``config``; it imports from the ``mobiletransformers.config``
package path, which is a different name, so there is no shadowing/collision.
"""

from __future__ import annotations

import warnings

from mobiletransformers.config.constants import (
    BATCH_SIZE,
    EXPERIMENT_RANKS,
    GRADIENT_ACCUMULATION,
    PER_DEVICE_BATCH_SIZE,
    TASK_EPOCHS,
)
from mobiletransformers.config.settings import get_settings as _get_settings

warnings.warn(
    "Import from mobiletransformers.config.settings / mobiletransformers.config.constants "
    "instead of the root config.py module.",
    DeprecationWarning,
    stacklevel=2,
)

_settings = _get_settings()

# Secrets (were read from env at import in the old config.py; same behavior via Settings).
HF_TOKEN = _settings.hf_token
AZURE_OPENAI_ENDPOINT = _settings.azure_openai_endpoint
AZURE_OPENAI_API_KEY = _settings.azure_openai_api_key
AZURE_DEPLOYMENT_NAME = _settings.azure_deployment_name
AZURE_MODEL_NAME = _settings.azure_model_name
AZURE_API_VERSION = _settings.azure_api_version

__all__ = [
    "HF_TOKEN",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_DEPLOYMENT_NAME",
    "AZURE_MODEL_NAME",
    "AZURE_API_VERSION",
    "TASK_EPOCHS",
    "BATCH_SIZE",
    "PER_DEVICE_BATCH_SIZE",
    "GRADIENT_ACCUMULATION",
    "EXPERIMENT_RANKS",
]

"""DEPRECATED compatibility shim.

The config section names and dataset map moved to ``mobiletransformers.config.constants``.
This module re-exports them so existing imports (e.g.
``from tools.parser_config import TRAIN_CONFIG``) keep working during migration.
"""

from __future__ import annotations

import warnings

from mobiletransformers.config.constants import (
    ARTIFACT_CONFIG,
    ARTIFACT_VALIDATOR_CONFIG,
    INFERENCE_ARTIFACT_CONFIG,
    INFERENCE_CONFIG,
    TASK_NAME_TO_DATASET,
    TEST_GENERATION_CONFIG,
    TRAIN_CONFIG,
)

warnings.warn(
    "Import from mobiletransformers.config.constants instead of tools.parser_config.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ARTIFACT_CONFIG",
    "ARTIFACT_VALIDATOR_CONFIG",
    "TRAIN_CONFIG",
    "INFERENCE_CONFIG",
    "INFERENCE_ARTIFACT_CONFIG",
    "TEST_GENERATION_CONFIG",
    "TASK_NAME_TO_DATASET",
]

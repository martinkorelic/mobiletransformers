"""Shared YAML loading helper.

Consolidates the six near-identical ``load_config_from_file`` definitions scattered across the
legacy business modules (``trainer/builder.py``, ``inference/builder.py``, ``artifact/onnx_builder.py``,
``trainer/validator.py``, ``trainer/merge_validator.py``, ``inference/validator.py``). New code should
import this; the in-module copies are replaced as each module migrates (later plans) — note that
``trainer/builder.py`` currently pre-indexes into ``config[TRAIN_CONFIG]``, so its call sites must be
updated together with that swap, not by silently repointing to this raw loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config_from_file(config_file: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (raw ``yaml.safe_load`` of the whole document)."""
    with open(config_file) as f:
        return yaml.safe_load(f)

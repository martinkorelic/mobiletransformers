"""Shared YAML loading helper — the single ``load_config_from_file``.

The consolidation is **done** (2026-08-14). Four byte-identical private copies
(``artifacts/builder.py``, ``artifacts/validation.py``, ``training/merge_validators.py``,
``training/validators.py``) now import this one.

Two copies deliberately did not merge:

* ``export/training_export.py`` pre-indexed into ``config[TRAIN_CONFIG]``, so it returned the *train
  section* rather than the whole document. Silently repointing it here would have changed what every
  call site receives, which is what ``00_code_plans/02``'s deferral note warned about. It is renamed
  ``load_train_config_from_file`` and is now a thin wrapper over this loader.
* ``inference/builder.py`` keeps its own copy: it is the vendored GenAI graph builder, treated as
  upstream (and allow-listed as such in ``tests/unit/test_guards.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config_from_file(config_file: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (raw ``yaml.safe_load`` of the whole document)."""
    with open(config_file) as f:
        return yaml.safe_load(f)

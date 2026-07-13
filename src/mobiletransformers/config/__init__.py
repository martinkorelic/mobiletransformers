"""MobileTransformers config package: settings (secrets), constants, and the precedence helper.

Effective value for any tunable resolves in this strict order (highest first):

1. CLI flag              (owned by ``cli/*.py``)
2. Environment variable  (via ``settings.get_settings()``)
3. ``config/config.yml`` value (via ``utils.yaml.load_config_from_file``)
4. Package default       (``constants.py``)

Secrets skip ranks 1/3/4 — they live only at rank 2 (env via ``Settings``). YAML never holds secrets.
"""

from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def resolve(
    cli_value: _T | None, env_value: _T | None, yaml_value: _T | None, default: _T | None
) -> _T | None:
    """Return the first non-``None`` value in CLI > env > YAML > default order."""
    for value in (cli_value, env_value, yaml_value, default):
        if value is not None:
            return value
    return None


__all__ = ["resolve"]

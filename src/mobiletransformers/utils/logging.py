"""Structured, module-level logging for library code.

Use ``logger = get_logger(__name__)`` per module and log at levels — never ``print()`` inside the
library (user-facing CLI output stays in ``cli/``). The library attaches a ``NullHandler`` to its
root logger (PEP-recommended) so importing it never configures the application's logging; apps opt in
via ``configure_logging()`` or their own handlers.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "mobiletransformers"

# PEP 282 / library best practice: a NullHandler so we never emit unless the app configures logging.
logging.getLogger(_ROOT_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``mobiletransformers`` namespace.

    Pass ``__name__``; a non-package name is namespaced under ``mobiletransformers.`` so all library
    loggers share one configurable root.
    """
    if name == "__main__" or not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Opt-in convenience: attach a basic stream handler to the library root (for CLI/dev use)."""
    root = logging.getLogger(_ROOT_NAME)
    if not any(not isinstance(h, logging.NullHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)


__all__ = ["get_logger", "configure_logging"]

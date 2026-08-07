"""DEPRECATED compatibility shim — the contents moved into the package (Migration Map S1).

``tools/utils.py`` was a grab bag: filesystem helpers, chat-template rendering, dataset preparation and
a Trainer callback. It was split by concern so the dependency-free parts stay importable in the core
environment:

============================  ==================================================
old name                      new home
============================  ==================================================
``move_onnx_model``           ``mobiletransformers.utils.paths``
``move_files_excluding``      ``mobiletransformers.utils.paths``
``delete_directory``          ``mobiletransformers.utils.paths``
``create_chat_input``         ``mobiletransformers.utils.templating``
``render_template``           ``mobiletransformers.utils.templating``
``load_and_save_dataset``     ``mobiletransformers.training.data``
``trim_dataset``              ``mobiletransformers.training.data``
``save_as_jsonl``             ``mobiletransformers.training.data``
``preload_dataset``           ``mobiletransformers.training.data``
``MemoryLoggerCallback``      ``mobiletransformers.training.callbacks``
============================  ==================================================

Removed once every caller uses the package paths (Migration Map S9).
"""

import warnings

from mobiletransformers.training.data import (  # noqa: F401
    load_and_save_dataset,
    preload_dataset,
    save_as_jsonl,
    trim_dataset,
)
from mobiletransformers.utils.paths import (  # noqa: F401
    delete_directory,
    move_files_excluding,
    move_onnx_model,
)
from mobiletransformers.utils.templating import create_chat_input, render_template  # noqa: F401

warnings.warn(
    "tools.utils moved into mobiletransformers.{utils.paths,utils.templating,training.data,"
    "training.callbacks}; the shim will be removed.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    """Resolve ``MemoryLoggerCallback`` lazily — it subclasses a transformers type."""
    if name == "MemoryLoggerCallback":
        from mobiletransformers.training.callbacks import MemoryLoggerCallback

        return MemoryLoggerCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemoryLoggerCallback",
    "create_chat_input",
    "delete_directory",
    "load_and_save_dataset",
    "move_files_excluding",
    "move_onnx_model",
    "preload_dataset",
    "render_template",
    "save_as_jsonl",
    "trim_dataset",
]

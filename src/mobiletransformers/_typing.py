"""Shared typing aliases used across the package.

Kept small and dependency-free. The ``py.typed`` marker (sibling file) exports these types to
downstream consumers per PEP 561.
"""

from __future__ import annotations

import os
from typing import Any

# A filesystem path accepted by the library (str or os.PathLike). PEP 604 union (runtime, py>=3.10).
PathLike = str | os.PathLike[str]

# An ONNX tensor/initializer name.
TensorName = str

# A decoded JSON object.
JsonDict = dict[str, Any]

__all__ = ["PathLike", "TensorName", "JsonDict"]

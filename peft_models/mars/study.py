"""DEPRECATED shim — moved to ``mobiletransformers.peft.mars.study`` (Migration Map S3)."""

import warnings

from mobiletransformers.peft.mars.study import (  # noqa: F401
    factorize,
    find_best_shape,
    reshape_to_higher_order,
    sequential_svd,
    tensor_train_contract,
    tensor_train_decomposition,
    tt_tensor_elements,
)

warnings.warn(
    "peft_models.mars.study moved to mobiletransformers.peft.mars.study.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "factorize",
    "find_best_shape",
    "reshape_to_higher_order",
    "sequential_svd",
    "tensor_train_contract",
    "tensor_train_decomposition",
    "tt_tensor_elements",
]

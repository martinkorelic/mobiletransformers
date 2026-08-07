"""DEPRECATED shim — moved into the package (Migration Map S4).

``add_cls_pooling`` -> ``mobiletransformers.export.embedding_export``
``add_concatenation`` -> ``mobiletransformers.export.embedding_export``
``add_max_pooling`` -> ``mobiletransformers.export.embedding_export``
``add_mean_pooling`` -> ``mobiletransformers.export.embedding_export``
``add_mean_sqrt_len_pooling`` -> ``mobiletransformers.export.embedding_export``
``add_pooling_operations`` -> ``mobiletransformers.export.embedding_export``
``add_pooling_to_onnx_model`` -> ``mobiletransformers.export.embedding_export``
``load_pooling_config_from_hub`` -> ``mobiletransformers.export.embedding_export``
``print_pooling_summary`` -> ``mobiletransformers.export.embedding_export``
"""

import warnings

from mobiletransformers.export.embedding_export import (  # noqa: F401
    add_cls_pooling,
    add_concatenation,
    add_max_pooling,
    add_mean_pooling,
    add_mean_sqrt_len_pooling,
    add_pooling_operations,
    add_pooling_to_onnx_model,
    load_pooling_config_from_hub,
    print_pooling_summary,
)

warnings.warn(
    "trainer.embedding_builder moved into mobiletransformers.*; the shim will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "add_cls_pooling",
    "add_concatenation",
    "add_max_pooling",
    "add_mean_pooling",
    "add_mean_sqrt_len_pooling",
    "add_pooling_operations",
    "add_pooling_to_onnx_model",
    "load_pooling_config_from_hub",
    "print_pooling_summary",
]

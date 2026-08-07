"""DEPRECATED shim — moved into the package (Migration Map S4).

``OnnxInferenceWrapper`` -> ``mobiletransformers.export.training_export``
``OnnxTrainerWrapper`` -> ``mobiletransformers.export.training_export``
``add_peft_type`` -> ``mobiletransformers.export.training_export``
``apply_metadata`` -> ``mobiletransformers.export.training_export``
``check_extra_options`` -> ``mobiletransformers.export.training_export``
``compare_weights`` -> ``mobiletransformers.export.training_export``
``count_trainable_parameters`` -> ``mobiletransformers.export.training_export``
``ensure_training_mode_input`` -> ``mobiletransformers.export.training_export``
``get_layers_with_grad`` -> ``mobiletransformers.export.training_export``
``inspect_weights`` -> ``mobiletransformers.export.training_export``
``load_config_from_file`` -> ``mobiletransformers.export.training_export``
``onnx_dynamic_quantization`` -> ``mobiletransformers.export.training_export``
``optimum_hf_export`` -> ``mobiletransformers.export.training_export``
``parse_argument_list`` -> ``mobiletransformers.export.training_export``
``parse_arguments`` -> ``mobiletransformers.export.training_export``
``parse_extra_options`` -> ``mobiletransformers.export.training_export``
``preprocess_model`` -> ``mobiletransformers.export.training_export``
``trim_initializers`` -> ``mobiletransformers.export.training_export``
"""

import warnings

from mobiletransformers.export.training_export import (  # noqa: F401
    OnnxInferenceWrapper,
    OnnxTrainerWrapper,
    add_peft_type,
    apply_metadata,
    check_extra_options,
    compare_weights,
    count_trainable_parameters,
    ensure_training_mode_input,
    get_layers_with_grad,
    inspect_weights,
    load_config_from_file,
    onnx_dynamic_quantization,
    optimum_hf_export,
    parse_argument_list,
    parse_arguments,
    parse_extra_options,
    preprocess_model,
    trim_initializers,
)

warnings.warn(
    "trainer.builder moved into mobiletransformers.*; the shim will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "OnnxInferenceWrapper",
    "OnnxTrainerWrapper",
    "add_peft_type",
    "apply_metadata",
    "check_extra_options",
    "compare_weights",
    "count_trainable_parameters",
    "ensure_training_mode_input",
    "get_layers_with_grad",
    "inspect_weights",
    "load_config_from_file",
    "onnx_dynamic_quantization",
    "optimum_hf_export",
    "parse_argument_list",
    "parse_arguments",
    "parse_extra_options",
    "preprocess_model",
    "trim_initializers",
]

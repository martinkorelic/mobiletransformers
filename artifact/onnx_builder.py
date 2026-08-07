"""DEPRECATED shim — moved to ``mobiletransformers.artifacts.builder`` (Migration Map S5)."""

import warnings

from mobiletransformers.artifacts.builder import (  # noqa: F401
    CausalLMCE,
    convert_pipeline,
    force_dequantize_external_and_save,
    gen_artifacts,
    gen_genai,
    get_all_metadata_from_onnx,
    get_layers_with_grad,
    load_config_from_file,
    onnx_checktrain,
    onnx_export_dummy_model,
    onnx_infer,
    onnx_segment_weights,
    onnx_transfer_trained_weights,
    parse_arguments,
    parse_extra_options,
)

warnings.warn(
    "artifact.onnx_builder moved to mobiletransformers.artifacts.builder.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CausalLMCE",
    "convert_pipeline",
    "force_dequantize_external_and_save",
    "gen_artifacts",
    "gen_genai",
    "get_all_metadata_from_onnx",
    "get_layers_with_grad",
    "load_config_from_file",
    "onnx_checktrain",
    "onnx_export_dummy_model",
    "onnx_infer",
    "onnx_segment_weights",
    "onnx_transfer_trained_weights",
    "parse_arguments",
    "parse_extra_options",
]

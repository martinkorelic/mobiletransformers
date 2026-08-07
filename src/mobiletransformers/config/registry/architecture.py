"""Architecture registry — single source of truth for per-architecture export/inference dispatch.

Replaces the parallel architectures-keyed if/elif chains in ``trainer/builder.py`` (training export,
Optimum ``*OnnxConfig``) and ``inference/builder.py`` (inference graph ``*Model`` builders).
Adding an architecture is a registry row — no new ``elif``.

Class bindings are **lazy dotted-path strings** (resolved via ``import_from_path`` only when an export
actually runs) so this registry imports cleanly in the core env without pulling optimum/torch. The
actual dispatch-site rewrites in the legacy builders ride with their migration plans (training export
#7, inference builder is gated by the Optimum/GenAI decision — see the restructure master plan); this
module provides the contract they consume.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from mobiletransformers.config.constants import TaskType
from mobiletransformers.exceptions import UnsupportedModelError


def import_from_path(dotted: str) -> Any:
    """Import ``pkg.mod.Name`` lazily and return the attribute (raises at call time, not load time)."""
    module_path, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_path), attr)


@dataclass(frozen=True)
class ArchitectureSpec:
    architecture: str  # config.architectures[0], e.g. "LlamaForCausalLM"
    onnx_config_class: str  # dotted path to the Optimum OnnxConfig (training export)
    # Export-path target modules, keyed (like this whole registry) by config.architectures[0].
    # NOT the same table as PEFT_TARGET_MODULES_BY_MODEL_TYPE in registry/peft.py, which is keyed by
    # config.model_type and covers the far wider set of PEFT-wrappable models (incl. encoders/seq2seq).
    target_modules: tuple[str, ...]
    inference_model_class: str | None = None  # dotted path to the genai inference builder; None = n/a
    attention_module_name: str = "self_attn"
    task: TaskType = TaskType.TEXT_GENERATION
    # Variant selection (e.g. Phi3 4K vs 128K keyed on config.max_position_embeddings).
    variant_key: str | None = None
    variant_values: dict[int, str] = field(default_factory=dict)  # {variant_value: inference dotted path}

    def load_onnx_config_class(self) -> Any:
        return import_from_path(self.onnx_config_class)

    def load_inference_model_class(self, variant_value: int | None = None) -> Any:
        if self.variant_key is not None and variant_value is not None:
            path = self.variant_values.get(variant_value)
            if path is None:
                raise UnsupportedModelError(
                    f"{self.architecture} has no inference variant for {self.variant_key}={variant_value}"
                )
            return import_from_path(path)
        if self.inference_model_class is None:
            raise UnsupportedModelError(f"{self.architecture} has no inference builder yet")
        return import_from_path(self.inference_model_class)


_OC = "optimum.exporters.onnx.model_configs"
_INF = "inference.builder"  # legacy genai builder module (moves in the inference migration)

ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = {
    "LlamaForCausalLM": ArchitectureSpec(
        "LlamaForCausalLM", f"{_OC}.LlamaOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.LlamaModel"
    ),
    "GemmaForCausalLM": ArchitectureSpec(
        "GemmaForCausalLM", f"{_OC}.GemmaOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.GemmaModel"
    ),
    "Gemma2ForCausalLM": ArchitectureSpec(
        "Gemma2ForCausalLM", f"{_OC}.GemmaOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.Gemma2Model"
    ),
    # Gemma3 export is supported (GemmaOnnxConfig); inference is the FunctionGemma gate (#37).
    "Gemma3ForCausalLM": ArchitectureSpec(
        "Gemma3ForCausalLM", f"{_OC}.GemmaOnnxConfig", ("q_proj", "v_proj"), None
    ),
    "Phi3ForCausalLM": ArchitectureSpec(
        "Phi3ForCausalLM",
        f"{_OC}.Phi3OnnxConfig",
        ("qkv_proj", "o_proj"),
        variant_key="max_position_embeddings",
        variant_values={4096: f"{_INF}.Phi3Mini4KModel", 131072: f"{_INF}.Phi3Mini128KModel"},
    ),
    "Qwen2ForCausalLM": ArchitectureSpec(
        "Qwen2ForCausalLM", f"{_OC}.Qwen2OnnxConfig", ("q_proj", "v_proj"), f"{_INF}.QwenModel"
    ),
    "OPTForCausalLM": ArchitectureSpec(
        "OPTForCausalLM",
        f"{_OC}.OPTOnnxConfig",
        ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
    ),
    "BertModel": ArchitectureSpec(
        "BertModel",
        f"{_OC}.BertOnnxConfig",
        ("query", "value"),
        attention_module_name="attention",
        task=TaskType.FEATURE_EXTRACTION,
    ),
}


def resolve_architecture(config: Any) -> ArchitectureSpec:
    """Look up the spec for a HF config's first architecture. Fail closed on unknown."""
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise UnsupportedModelError("model config has no `architectures`")
    name = architectures[0]
    spec = ARCHITECTURE_REGISTRY.get(name)
    if spec is None:
        raise UnsupportedModelError(f"unsupported architecture: {name}")
    return spec


__all__ = ["ArchitectureSpec", "ARCHITECTURE_REGISTRY", "resolve_architecture", "import_from_path"]

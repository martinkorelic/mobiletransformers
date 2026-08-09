"""Architecture registry — single source of truth for per-architecture export/inference dispatch.

Replaces the parallel architectures-keyed if/elif chains in ``trainer/builder.py`` (training export,
Optimum ``*OnnxConfig``) and ``inference/builder.py`` (inference graph ``*Model`` builders).
Adding an architecture is a registry row — no new ``elif``.

Class bindings are **lazy dotted-path strings** (resolved via ``import_from_path`` only when an export
actually runs) so this registry imports cleanly in the core env without pulling optimum/torch.

The registry now covers **every** branch of ``inference/builder.py``'s 14-branch architecture ladder,
including the three that did more than pick a class — those side effects are data on the row
(``option_overrides`` / ``extra_option_overrides`` / ``config_overrides`` / ``warnings``) rather than
statements in a chain. Adding an architecture is a registry row; no new ``elif``.

**The dispatch site now consumes this** (S6): ``inference/builder.py``'s ladder is gone, the module
moved into the package, and all 15 branches were verified to resolve to the same class objects the
chain constructed. ``tests/unit/test_guards.py``'s ``DISPATCH_ALLOWLIST`` is empty as a result.
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
    # Dotted path to the Optimum OnnxConfig (training export). None for inference-only architectures:
    # PhiMoE, Phi3Small, Phi3V and ChatGLM have a genai inference builder but NO Optimum config exists
    # for them, so they can be exported for inference and never trained through the Optimum path.
    # Verified against optimum-onnx's model_configs, not assumed.
    onnx_config_class: str | None
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

    # --- Side effects the legacy ladder performed inline (the one design gap in #6's registry plan) ---
    #
    # Three of `inference/builder.py`'s 14 branches did more than pick a class: they mutated the export
    # request or the HF config before constructing it. A registry row that only names a class cannot
    # replace those branches, so the mutations become data too.

    #: Export-option forcings, e.g. `{"execution_provider": "cuda", "precision": "int4"}` for PhiMoE,
    #: whose `MoE` op ORT implements for CUDA only. Applied over the caller's request.
    option_overrides: dict[str, Any] = field(default_factory=dict)

    #: `extra_options` forcings, e.g. `{"exclude_embeds": True}` for Phi3V (text component only).
    extra_option_overrides: dict[str, Any] = field(default_factory=dict)

    #: HF-config attribute forcings applied before the builder reads them, e.g. ChatGLM's
    #: `hidden_act = "swiglu"` (its config declares an activation the builder does not map).
    config_overrides: dict[str, Any] = field(default_factory=dict)

    #: Operator-facing warnings the legacy branches `print`ed. Kept as data so a caller can log them
    #: through the normal logger rather than to stdout.
    warnings: tuple[str, ...] = ()

    def load_onnx_config_class(self) -> Any:
        if self.onnx_config_class is None:
            raise UnsupportedModelError(
                f"{self.architecture} has no Optimum OnnxConfig — it is inference-only and cannot be "
                "exported for training through the Optimum path."
            )
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
_INF = "mobiletransformers.inference.builder"  # S6: in the package — the wheel is self-contained

ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = {
    "LlamaForCausalLM": ArchitectureSpec(
        "LlamaForCausalLM", f"{_OC}.LlamaOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.LlamaModel"
    ),
    "GemmaForCausalLM": ArchitectureSpec(
        "GemmaForCausalLM", f"{_OC}.GemmaOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.GemmaModel"
    ),
    # Gemma2/Gemma3 bind their OWN Optimum configs. Both rows previously pointed at `GemmaOnnxConfig`,
    # which is a different architecture — Gemma2 adds alternating sliding-window attention and logit
    # soft-capping, Gemma3 differs again — so the generic config would have described the wrong graph
    # (KV-cache layout and attention in particular). Verified present in the pinned optimum-onnx 0.1.0.
    # NOT exercised end to end here: no profile in this checkout has optimum installed, and the dotted
    # paths resolve lazily, so this is a correctness fix that the export profile still has to confirm.
    "Gemma2ForCausalLM": ArchitectureSpec(
        "Gemma2ForCausalLM", f"{_OC}.Gemma2OnnxConfig", ("q_proj", "v_proj"), f"{_INF}.Gemma2Model"
    ),
    # Gemma3 export is supported; inference is the FunctionGemma gate (#37).
    "Gemma3ForCausalLM": ArchitectureSpec(
        "Gemma3ForCausalLM", f"{_OC}.Gemma3OnnxConfig", ("q_proj", "v_proj"), None
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
    # --- The #6 remainder: the last 7 rows of `inference/builder.py`'s 14-branch ladder -------------
    "MistralForCausalLM": ArchitectureSpec(
        "MistralForCausalLM", f"{_OC}.MistralOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.MistralModel"
    ),
    "PhiForCausalLM": ArchitectureSpec(
        "PhiForCausalLM", f"{_OC}.PhiOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.PhiModel"
    ),
    # MoE is CUDA-only in ORT and this builder only emits a quantized graph, so the legacy branch
    # OVERRODE whatever the caller asked for. Encoded as data rather than lost.
    "PhiMoEForCausalLM": ArchitectureSpec(
        "PhiMoEForCausalLM",
        None,
        ("q_proj", "v_proj"),
        variant_key="max_position_embeddings",
        variant_values={131072: f"{_INF}.Phi3MoE128KModel"},
        option_overrides={"execution_provider": "cuda", "precision": "int4"},
        warnings=(
            "PhiMoE runs on CUDA only (ORT implements `MoE` for CUDA); forcing execution_provider=cuda.",
            "PhiMoE is supported in quantized form only; forcing precision=int4.",
        ),
    ),
    "Phi3SmallForCausalLM": ArchitectureSpec(
        "Phi3SmallForCausalLM",
        None,
        ("query_key_value", "dense"),
        variant_key="max_position_embeddings",
        variant_values={8192: f"{_INF}.Phi3Small8KModel", 131072: f"{_INF}.Phi3Small128KModel"},
    ),
    "Phi3VForCausalLM": ArchitectureSpec(
        "Phi3VForCausalLM",
        None,
        ("qkv_proj", "o_proj"),
        f"{_INF}.Phi3VModel",
        extra_option_overrides={"exclude_embeds": True},
        warnings=("Phi3V export covers the TEXT component only; forcing exclude_embeds=true.",),
    ),
    "NemotronForCausalLM": ArchitectureSpec(
        "NemotronForCausalLM", f"{_OC}.NemotronOnnxConfig", ("q_proj", "v_proj"), f"{_INF}.NemotronModel"
    ),
    # Two architecture strings for one model: a quantized ChatGLM declares
    # `ChatGLMForConditionalGeneration`, the HF model declares `ChatGLMModel`. Both rows point at the
    # same builder — the legacy branch `or`-ed them, and a registry keyed by architectures[0] needs both.
    "ChatGLMForConditionalGeneration": ArchitectureSpec(
        "ChatGLMForConditionalGeneration",
        None,
        ("query_key_value", "dense"),
        f"{_INF}.ChatGLMModel",
        config_overrides={"hidden_act": "swiglu"},
    ),
    "ChatGLMModel": ArchitectureSpec(
        "ChatGLMModel",
        None,
        ("query_key_value", "dense"),
        f"{_INF}.ChatGLMModel",
        config_overrides={"hidden_act": "swiglu"},
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
    # --- Encoder classification (#33) ---
    #
    # Separate rows from `BertModel` because the architecture key IS the head: a checkpoint loaded as
    # `AutoModelForSequenceClassification` reports `BertForSequenceClassification`, and it is the head
    # that decides the task, the label shape and the loss. `BertModel` stays feature-extraction (the
    # RAG embedder), and neither row has to know about the other.
    #
    # Targets are the BERT attention projection names (`query`/`value`), the encoder equivalent of the
    # decoders' `q_proj`/`v_proj` — the LoRA convention of adapting Wq and Wv.
    "BertForSequenceClassification": ArchitectureSpec(
        "BertForSequenceClassification",
        f"{_OC}.BertOnnxConfig",
        ("query", "value"),
        attention_module_name="attention",
        task=TaskType.SEQUENCE_CLASSIFICATION,
    ),
    "RobertaForSequenceClassification": ArchitectureSpec(
        "RobertaForSequenceClassification",
        f"{_OC}.RobertaOnnxConfig",
        ("query", "value"),
        attention_module_name="attention",
        task=TaskType.SEQUENCE_CLASSIFICATION,
    ),
    "DistilBertForSequenceClassification": ArchitectureSpec(
        "DistilBertForSequenceClassification",
        f"{_OC}.DistilBertOnnxConfig",
        # DistilBERT names its projections q_lin/v_lin, not query/value — exactly the per-architecture
        # difference this registry exists to hold as data.
        ("q_lin", "v_lin"),
        attention_module_name="attention",
        task=TaskType.SEQUENCE_CLASSIFICATION,
    ),
}


def resolve_architecture(config: Any, *, architecture: str | None = None) -> ArchitectureSpec:
    """Look up the spec for a HF config's first architecture. Fail closed on unknown.

    ``architecture`` overrides the lookup key and should be ``type(model).__name__`` whenever the
    model has actually been loaded. **The head is part of the architecture identity**, and
    ``config.architectures`` describes the *checkpoint*, not what was loaded from it: a
    sentence-transformers encoder declares ``["BertModel"]`` even when loaded through
    ``AutoModelForSequenceClassification`` as a ``BertForSequenceClassification``. Keying off the
    config alone therefore resolves an encoder fine-tune to the un-headed, untrainable row.

    For every already-supported path the two agree (`AutoModelForCausalLM` on a Llama checkpoint gives
    `LlamaForCausalLM`, which is also `architectures[0]`), so passing the loaded class is strictly more
    accurate rather than a behaviour change.
    """
    name = architecture
    if not name:
        architectures = getattr(config, "architectures", None) or []
        if not architectures:
            raise UnsupportedModelError("model config has no `architectures`")
        name = architectures[0]
    spec = ARCHITECTURE_REGISTRY.get(name)
    if spec is None:
        raise UnsupportedModelError(f"unsupported architecture: {name}")
    return spec


__all__ = ["ArchitectureSpec", "ARCHITECTURE_REGISTRY", "resolve_architecture", "import_from_path"]

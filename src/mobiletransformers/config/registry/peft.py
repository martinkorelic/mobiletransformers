"""PEFT method registry — single source of truth for PEFT config setup + adapter mapping + merger tag.

Replaces the train_method-keyed if/elif config-setup and adapter-mapping dispatch
(``trainer/builder.py``) and the bespoke ``create_mars_adapter_mapping`` / ``create_lora_mapping``
key hardcoding (``trainer/utils.py``). Adding a method is a registry row + a ``PEFTMethod`` enum member.

The ``component_schema`` order is the source of truth the tensor codec (#8,
``TrainableTensorCodec.from_peft_mapping``) consumes for deterministic tensor naming. ``config_class``
is a lazy dotted path (resolved only when a real PEFT wrap runs, in the training-export migration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mobiletransformers.config.constants import MergerVariant, PEFTMethod
from mobiletransformers.config.registry.architecture import import_from_path
from mobiletransformers.exceptions import UnsupportedModelError


@dataclass(frozen=True)
class AdapterComponent:
    role: str  # "shared_A" | "intermediate" | "adapter_A" | "adapter_B" | ...
    search_pattern: str  # how to locate the tensor in the PEFT-wrapped module


@dataclass(frozen=True)
class PEFTMethodSpec:
    method: PEFTMethod
    config_class: str | None  # dotted path to LoraConfig | MarsConfig | ...; None for all/nolora
    component_schema: tuple[AdapterComponent, ...]  # ORDER is the codec's source of truth
    merger_variant_fp: MergerVariant | None  # merger variant when output is not quantized
    merger_variant_q: MergerVariant | None  # merger variant when output is quantized
    builds_mapping: bool = True
    #: Lazy dotted path to this method's adapter-mapping builder (same lazy-import convention as
    #: ``config_class``). ``None`` when ``builds_mapping`` is False. The builders differ in substance
    #: — MARS tracks shared-module identity and qkv/mlp adapter indices, LoRA is a flat module scan —
    #: so this dispatches to them rather than pretending one generic walk covers both.
    mapping_builder: str | None = None


#: PEFT target modules keyed by HF ``config.model_type`` — the SINGLE source for what MARS/ablation
#: wrap when the user names no explicit ``target_modules``. Previously duplicated byte-for-byte across
#: ``peft/mars/utils.py`` and ``peft/ablation/utils.py`` under two different dict names,
#: which is exactly the drift #6 exists to remove; both now re-export this table.
#:
#: NOTE this is keyed by ``model_type`` (``llama``, ``t5``, ...) and deliberately NOT collapsed onto
#: :attr:`ArchitectureSpec.target_modules`, which is keyed by ``config.architectures[0]``
#: (``LlamaForCausalLM``, ...) and covers only the 8 architectures the ONNX export path supports.
#: The two key spaces are different and this one is far wider — folding them would silently drop
#: PEFT support for t5/mt5/bart/gpt2/bloom/gptj/gpt_neo(x)/bert/roberta/deberta(-v2)/gpt_bigcode.
PEFT_TARGET_MODULES_BY_MODEL_TYPE: dict[str, list[str]] = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "gpt_bigcode": ["c_attn"],
    "deberta": ["in_proj"],
    "qwen2": ["q_proj", "v_proj"],
}


_LORA_COMPONENTS = (
    AdapterComponent("adapter_A", "lora_A"),
    AdapterComponent("adapter_B", "lora_B"),
)
_MARS_COMPONENTS = (
    AdapterComponent("shared_A", "mars_shared_A"),
    AdapterComponent("adapter_B", "mars_B"),
)

PEFT_REGISTRY: dict[PEFTMethod, PEFTMethodSpec] = {
    PEFTMethod.LORA: PEFTMethodSpec(
        PEFTMethod.LORA,
        "peft.LoraConfig",
        _LORA_COMPONENTS,
        MergerVariant.LORA,
        MergerVariant.LORA_Q,
        mapping_builder="mobiletransformers.peft.mapping.create_lora_mapping",
    ),
    PEFTMethod.LORA_XS: PEFTMethodSpec(
        PEFTMethod.LORA_XS,
        "peft.LoraConfig",
        _LORA_COMPONENTS,
        MergerVariant.LORA,
        MergerVariant.LORA_Q,
        # LoRA-XS reparameterizes an existing LoRA wrap, so it shares LoRA's module layout.
        mapping_builder="mobiletransformers.peft.mapping.create_lora_mapping",
    ),
    PEFTMethod.MARS: PEFTMethodSpec(
        PEFTMethod.MARS,
        "mobiletransformers.peft.mars.config.MarsConfig",
        _MARS_COMPONENTS,
        MergerVariant.MARS_Q,
        MergerVariant.MARS_Q,
        mapping_builder="mobiletransformers.peft.mapping.create_mars_adapter_mapping",
    ),
    # "all" (all linear layers trainable) and "nolora" produce no adapter mapping and no merger.
    PEFTMethod.ALL: PEFTMethodSpec(PEFTMethod.ALL, None, (), None, None, builds_mapping=False),
    PEFTMethod.NOLORA: PEFTMethodSpec(PEFTMethod.NOLORA, None, (), None, None, builds_mapping=False),
}


def get_peft_spec(method: PEFTMethod) -> PEFTMethodSpec:
    """Look up the spec for a PEFT method. Fail closed on unknown."""
    spec = PEFT_REGISTRY.get(method)
    if spec is None:
        raise UnsupportedModelError(f"unsupported PEFT method: {method}")
    return spec


def build_adapter_mapping(method: PEFTMethod, model: Any, **kwargs: Any) -> dict[str, Any]:
    """Build the base-layer -> adapter-tensor mapping for ``method`` (#6 A3).

    The ONE entry point for adapter mapping: callers pass the resolved :class:`PEFTMethod` and never
    branch on a method string themselves. It returns ``{}`` for methods that produce no adapters
    (``all`` / ``nolora``) and fails closed on a method whose spec claims to build a mapping but
    registers no builder.

    ``**kwargs`` are forwarded to the resolved builder (e.g. MARS's ``shared_qkv`` /
    ``shared_mlp_enabled``); a builder that does not accept them will say so.
    """
    spec = get_peft_spec(method)
    if not spec.builds_mapping:
        return {}
    if spec.mapping_builder is None:
        raise UnsupportedModelError(
            f"PEFT method {method.value!r} declares builds_mapping but registers no mapping_builder"
        )
    return dict(import_from_path(spec.mapping_builder)(model, **kwargs))


__all__ = [
    "AdapterComponent",
    "PEFTMethodSpec",
    "PEFT_REGISTRY",
    "get_peft_spec",
    "build_adapter_mapping",
]

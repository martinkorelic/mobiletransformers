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

from mobiletransformers.config.constants import MergerVariant, PEFTMethod
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
    ),
    PEFTMethod.LORA_XS: PEFTMethodSpec(
        PEFTMethod.LORA_XS,
        "peft.LoraConfig",
        _LORA_COMPONENTS,
        MergerVariant.LORA,
        MergerVariant.LORA_Q,
    ),
    PEFTMethod.MARS: PEFTMethodSpec(
        PEFTMethod.MARS,
        "peft_models.mars.config.MarsConfig",
        _MARS_COMPONENTS,
        MergerVariant.MARS_Q,
        MergerVariant.MARS_Q,
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


__all__ = ["AdapterComponent", "PEFTMethodSpec", "PEFT_REGISTRY", "get_peft_spec"]

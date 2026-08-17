from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from peft.config import PeftConfig


class AblationVariant(Enum):
    """
    - Variant 0 - Normal LoRA
    - Variant A - LoRA + intermediate layer
    - Variant B - Input vector + frozen downprojection + up projection
    - Variant C - Frozen downprojection + intermediate + up projection
    - Variant D - Shared frozen downprojection + intermediate + up projection
    - Variant E - Mid training random rank pruning
    - Variant F - Mid training least L1 dimensions rank pruning
    - Variant G - Dynamic quantized backbone (int8)
    - Variant H - Dynamic quantized backbone (int4)
    """

    VARIANT_0 = "0"
    VARIANT_A = "A"
    VARIANT_B = "B"
    VARIANT_C = "C"
    VARIANT_D = "D"
    VARIANT_E = "E"
    VARIANT_F = "F"
    VARIANT_G = "G"
    VARIANT_H = "H"


@dataclass
class AblationConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`AblationModel`].

    Args:
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply Vera to. Only linear layers are supported.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        modules_to_save (`List[str]`):
            List of modules apart from Vera layers to be set as trainable and saved in the final checkpoint.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    variant: str = field(
        default="0",
        metadata={
            "help": (
                "Variants to choose from:"
                "- Variant 0 - Normal LoRA"
                "- Variant A - LoRA + intermediate layer"
                "- Variant B - Input vector + frozen downprojection + up projection"
                "- Variant C - Frozen downprojection + intermediate + up projection"
                "- Variant D - Shared frozen downprojection + intermediate + up projection"
                "- Variant E - Mid training random rank pruning"
                "- Variant F - Mid training least L1 dimensions rank pruning"
                "- Variant G - Dynamic quantized backbone (int8)"
                "- Variant H - Dynamic quantized backbone (int4)"
            )
        },
    )
    ### VARIANT SPECIFIC SETTINGS ###
    share_weights: bool = field(
        default=False,
        metadata={"help": "Set this to True if you want to share weights in downprojection layer."},
    )
    metric_tracking: bool = field(
        default=False,
        metadata={"help": "Track metrics during training."},
    )
    track_n: int = field(default=100, metadata={"help": "Average and store metrics every n steps."})
    target_modules: list[str] | str | None = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            )
        },
    )
    alpha: int = field(default=8, metadata={"help": "Scaling factor, computed as alpha/rank."})
    init_weight: str = field(
        default="kaiming",
        metadata={"help": ("Initialization of the adapter weights.")},
    )
    seed: int = field(default=42, metadata={"help": "Seed for initializing layers."})
    bias: str = field(
        default="none", metadata={"help": "Bias type for Ablation. Can be 'none', 'all' or 'ablation_only'"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Vera layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: list[int] | int | None = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: list[str] | str | None = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )

    def __post_init__(self):
        # super().__post_init__()
        # PEFT type
        self.peft_type = "ABLATION"

        # Convert target_modules to list instead of set to avoid potential issues
        if isinstance(self.target_modules, list):
            self.target_modules = list(set(self.target_modules))  # Remove duplicates but keep as list
        elif isinstance(self.target_modules, set):
            self.target_modules = list(self.target_modules)  # Convert set to list

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError(
                "When `layers_pattern` is specified, `layers_to_transform` must also be specified. "
            )

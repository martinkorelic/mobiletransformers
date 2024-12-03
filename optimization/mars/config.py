from __future__ import annotations

import warnings
from dataclasses import dataclass

from peft.config import PeftConfig
from peft.utils import PeftType

from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class MarsConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`MarsModel`].

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

    ranks: Optional[Union[list[int]]] = field(
        default_factory=[64, 32],
        metadata={
            "help": (
                "List of ranks to create the adapter rank latent subspaces. Mixture matrices will be computed from these ranks."
            )
        },
    )
    lora_alphas: Optional[Union[list[int]]] = field(
        default_factory=[1, 1],
        metadata={
            "help": (
                "List of alphas for each of these latent subspace outputs, matches the index for each of the ranks in `ranks`."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Vera."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Mars. Can be 'none', 'all' or 'mars_only'"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Vera layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
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
        #super().__post_init__()
        # PEFT type
        self.peft_type = "MARS"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
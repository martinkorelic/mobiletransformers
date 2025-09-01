from __future__ import annotations

import warnings
from dataclasses import dataclass

from peft.config import PeftConfig
from peft.utils import PeftType

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

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

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    shared_r: Optional[int] = field(default=None, metadata={"help": "Shared rank attention dimension"})
    optimization_level: int = field(default=0, metadata={
        "help": (
            "Optimization level to enable with other configurations:"
            "0 - fully trainable all layers and no quantization"
            "1 - partial trainable layers (frozen and fused down projection layers) with no quantization"
            "2 - fully trainable layers with partial quantization (specified in `modules_to_quantize`)"
            "3 - fully trainable layers with full quantization"
            "4 - partial trainable layers (frozen and fused down projection layers) with full quantization"
            )
        }
    )
    target_modules: Optional[Union[list[str], str]] = field(
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
    enabled_qkv: Optional[Tuple[str, ...]] = field(default=("q", "k", "v"), metadata={"help": "Which QKV projections to enable in the shared QKV adapter. Please select between 'q', 'k' and 'v'."})
    enabled_mlp: bool = field(
        default=True,
        metadata={"help": "Set this to True if we should have a shared down_proj and gate_proj down projection layers."},
    )
    mixture: bool = field(
        default=False,
        metadata={"help": "Set this to True if the adapter layer should include a mixture layer."},
    )
    modules_to_quantize: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules in which we apply dynamic quantization on base layers."
                "Choose between the following options: 'q', 'k', 'v', 'gate', 'down', 'up', 'o'."
            )
        },
    )
    modules_to_preserve_errors: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules in which preserve the quantization error."
                "If we have shared QKV or shared MLP enabled, we can only have in max of those modules errors preserved."
                "Choose between the following options: 'q', 'k', 'v', 'gate', 'down', 'up', 'o'."
            )
        },
    )
    orthogonal_init: bool = field(
        default=False,
        metadata={"help": "Whether to enable orthogonal initialization in intermediate matrices."},
    )
    quant_n_bits: int = field(default=8, metadata={"help": "Quantization type (bits for quantized weights) for MARS. Can be either '8' or '4'."})
    use_bnb: bool = field(
        default=True,
        metadata={"help": "Whether to use BitsAndBytes to quantize base layers."},
    )
    onnx_export: bool = field(
        default=False,
        metadata={"help": "Whether to prepare the model for ONNX export (disable quantization)."},
    )
    alpha: int = field(default=8, metadata={"help": "Scaling factor, computed as alpha/rank."})
    seed: int = field(default=42, metadata={"help": "Seed for initializing layers."})
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
        
        # Convert target_modules to list instead of set to avoid potential issues
        if isinstance(self.target_modules, list):
            self.target_modules = list(set(self.target_modules))  # Remove duplicates but keep as list
        elif isinstance(self.target_modules, set):
            self.target_modules = list(self.target_modules)  # Convert set to list
            
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
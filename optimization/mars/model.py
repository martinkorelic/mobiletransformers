import os
import warnings
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
import torch
from torch.nn.modules import Module
from safetensors.torch import save_file
from research.experiments import generate_complementary_matrices

# TODO: Separate base tuners
#from .layer import Linear, MarsLayer
#from .layerv2 import Linear, MarsLayer
#from .layerv3 import Linear, MarsLayer
#from .layerv4 import Linear, MarsLayer
#from .layerv5 import Linear, MarsLayer
from .layerv6 import Linear, MarsLayer
from .utils import TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING

class MarsModel(BaseTuner):

    prefix: str = "mars"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str = "mars", low_cpu_mem_usage: bool = False) -> None:
        
        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage)        

    def _pre_injection_hook(self, model: Module, config: PeftConfig, adapter_name: str) -> None:
        self.shared_weights = torch.nn.ModuleDict({})
        
        self.gen1 = torch.Generator()
        self.gen1.manual_seed(config.seed)

        self.gen2 = torch.Generator()
        self.gen2.manual_seed(config.seed + 1)

    def _create_and_replace(self, mars_config, adapter_name: str, target, target_name: str, parent, current_key: str, **kwargs) -> None:

        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        
        bias = hasattr(target, "bias") and target.bias is not None

        # TODO: Shape matching with ranks check

        if isinstance(target, Linear):
            target.update_layer(
                target,
                adapter_name,
                mars_config.r,
                mars_config.subspace,
                mars_config.mixture
            )

            if mars_config.share_weights:
                target = self._shared_and_store_weights(target, mars_config)

        else:
            new_module = self._create_new_module(mars_config, adapter_name, target, mars_config.r, mars_config.subspace, mars_config.mixture, **kwargs)

            if mars_config.share_weights:
                new_module = self._shared_and_store_weights(new_module, mars_config)

            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)

            self._replace_module(parent, target_name, new_module, target)


    def _shared_and_store_weights(self, new_module, mars_config):
        # Ensure weight sharing in new_module.down_project
        down_project_shape = f"{new_module.in_features}x{mars_config.r}"  # Use a string key for ParameterDict

        if down_project_shape in self.shared_weights:
            # Reuse existing shared weights
            new_module.down_project[self.active_adapter] = self.shared_weights[down_project_shape]
        else:
            # Generate new shared weights
            A1, A2 = generate_complementary_matrices(m=new_module.in_features, n=mars_config.subspace[0], gen1=self.gen1, gen2=self.gen2)

            A = A1 if mars_config.subspace[1] == 0 else A2[:, :mars_config.r]

            self.shared_weights[down_project_shape] = torch.nn.Linear(new_module.in_features, mars_config.r, bias=False)
            self.shared_weights[down_project_shape].weight.data = A.T.contiguous()
            self.shared_weights[down_project_shape].weight.requires_grad = False

            new_module.down_project[self.active_adapter] = self.shared_weights[down_project_shape]

        return new_module

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "mars" in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    @staticmethod
    def _create_new_module(mars_config, adapter_name, target, rank, subspace, mixture, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if "fan_in_fan_out" in kwargs:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = False
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`"
            )
        
        new_module = Linear(
            base_layer=target,
            adapter_name=adapter_name,
            r=rank,
            subspace=subspace,
            mixture=mixture,
            fan_in_fan_out=mars_config.fan_in_fan_out,
            **kwargs
        )

        return new_module
    
    @staticmethod
    def _check_target_module_exists(mars_config, key):
        return check_target_module_exists(mars_config, key)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
    
    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "mars_only":
                for m in model.modules():
                    if isinstance(m, MarsLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
    
    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)
    
    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, MarsLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name
    
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True) -> None:
        """
        Saves the trainable adapter weights of the MarsModel in safetensors format.

        Args:
            save_directory (str): Directory where the adapter model and configuration files will be saved.
            safe_serialization (bool, optional): Whether to save in safetensors format. Defaults to True.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # Collect trainable adapter weights
        adapter_weights = {
            name: param.clone().detach().cpu()
            for name, param in self.model.named_parameters()
            if self.prefix in name
        }

        if not adapter_weights:
            warnings.warn("No trainable Mars adapters found. Nothing to save.")

        # Save weights
        file_path = os.path.join(save_directory, "adapter_model.safetensors")
        if safe_serialization:
            save_file(adapter_weights, file_path, metadata={"format": "pt"})
        else:
            torch.save(adapter_weights, file_path.replace(".safetensors", ".pt"))

        # Save adapter configuration
        for adapter_name, config in self.peft_config.items():
            config.save_pretrained(save_directory)

        print(f"Mars adapters saved to {save_directory}")
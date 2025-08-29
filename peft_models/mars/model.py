import os
import warnings
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
import torch
from torch.nn.modules import Module
from safetensors.torch import save_file

from .layer import Linear, MarsLayer, SharedAttentionAdapter, SharedMLPAdapter
from .utils import TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING

class MarsModel(BaseTuner):
    prefix: str = "mars"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str = "mars", low_cpu_mem_usage: bool = False) -> None:

        # Pre-initialization
        if peft_config[adapter_name].shared_r is None:
             peft_config[adapter_name].shared_r =  peft_config[adapter_name].r

        self.trainable_down = True
        self.optimization_level =  peft_config[adapter_name].optimization_level
        self.only_export = peft_config[adapter_name].onnx_export
        self.quant_n_bits = peft_config[adapter_name].quant_n_bits

        # Based on optimization level set configurations
        if  peft_config[adapter_name].optimization_level == 0:
            self.trainable_down = True
        elif  peft_config[adapter_name].optimization_level == 1:
            self.trainable_down = False
        elif  peft_config[adapter_name].optimization_level == 2:
            self.trainable_down = True
        elif  peft_config[adapter_name].optimization_level == 3:
            self.trainable_down = True
        elif  peft_config[adapter_name].optimization_level == 4:
            self.trainable_down = False

        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage)

    def _pre_injection_hook(self, model: Module, config: PeftConfig, adapter_name: str) -> None:

        enabled_qkv = getattr(config, "enabled_qkv", ("q", "k", "v"))

        # Map enabled projections to indices in tuple
        enabled_list = list(enabled_qkv)

        # TODO: Check modules if they are even in target_modules
        any_mlp = any([ 'gate_proj' in tm or 'up_proj' in tm for tm in config.target_modules])
        any_qkv = any([ 'q_proj' in tm or 'k_proj' in tm or 'v_proj' in tm for tm in config.target_modules])
 
        # Register hooks for each attention layer
        for name, module in model.named_modules():

            # TODO: Here we assume the attention layer is named "self_attn"
            if isinstance(module, type(model.model.layers[0].self_attn)) and any_qkv:

                # Create a separate shared adapter for each attention layer
                module.shared_qkv = SharedAttentionAdapter(
                    hidden_size=model.config.hidden_size,
                    rank=config.r,
                    shared_rank=config.shared_r,
                    alpha=config.alpha,
                    enabled=enabled_list
                )
                
                # Compute shared outputs once and store them
                def compute_shared_qkv(module, args, kwargs):
                    
                    # TODO: Could have args or kwargs where hidden states are, this might depend on architecture
                    
                    # Compute shared outputs only once
                    qkv_outputs = module.shared_qkv(kwargs['hidden_states'])
                    # Store them in the module for the projection layers to use

                    module.shared_qkv._shared_outputs = qkv_outputs
                    return None
                
                # Register the hook on the attention layer to compute shared outputs once
                module.register_forward_pre_hook(compute_shared_qkv, with_kwargs=True)

                def pass_qkv_inputs(module, args):
                    if module is None or not hasattr(module, 'shared_qkv'):
                        return args

                    shared_outputs = getattr(module.shared_qkv, "_shared_outputs", None)
                    if shared_outputs is None:
                        return args

                    # Get the specific output for this projection type
                    if module.projection_type not in shared_outputs:
                        return args

                    shared_output = shared_outputs[module.projection_type]
                    
                    # Delete the specific key to free memory
                    del module.shared_qkv._shared_outputs[module.projection_type]
                    
                    # Optional: Clean up the entire dict when empty
                    if not module.shared_qkv._shared_outputs:
                        del module.shared_qkv._shared_outputs

                    # Return original input paired with shared output
                    return (shared_output,) + args
                
                # Helper to assign proj attrs and register pre-hook if enabled
                def register_proj_hook(proj_name, proj_type):
                    proj = getattr(module, proj_name, None)
                    if proj is None:
                        return
                    if proj_type not in enabled_qkv:
                        return
                    proj.projection_type = proj_type
                    proj.register_forward_pre_hook(pass_qkv_inputs)
                
                
                # Register hooks for projections only if enabled
                register_proj_hook('q_proj', 'q')
                register_proj_hook('k_proj', 'k')
                register_proj_hook('v_proj', 'v')

            elif isinstance(module, type(model.model.layers[0].mlp)) and any_mlp:
                module.shared_mlp = SharedMLPAdapter(
                    hidden_size=model.config.hidden_size,
                    rank=config.r,
                    shared_rank=config.shared_r,
                    alpha=config.alpha
                )

                # Compute shared outputs once and store them
                def compute_shared_mlp(module, args, kwargs):
                    
                    # TODO: Could have args or kwargs where hidden states are, this might depend on architecture

                    # Compute shared outputs only once
                    gate_out, up_out = module.shared_mlp(args[0])
                    # Store them in the module for the projection layers to use
                    module.shared_mlp._shared_outputs = {
                        'gate': gate_out,
                        'up': up_out
                    }
                    return None
                
                # Register the hook on the attention layer to compute shared outputs once
                module.register_forward_pre_hook(compute_shared_mlp, with_kwargs=True)

                # Register forward pre-hooks for each projection to pass both inputs
                def pass_mlp_inputs(module, args):

                    # Get the appropriate shared output based on the projection type
                    if module.projection_type == 'gate':
                        shared_output = module.shared_mlp._shared_outputs['gate']
                        del module.shared_mlp._shared_outputs['gate']
                    elif module.projection_type == 'up':
                        shared_output = module.shared_mlp._shared_outputs['up']
                        del module.shared_mlp._shared_outputs['up']
                    else:
                        return args
                    
                    # Return modified args and kwargs
                    return (shared_output,) + args
                
                # Register the pre-hooks on the projection layers
                module.gate_proj.register_forward_pre_hook(pass_mlp_inputs)
                module.up_proj.register_forward_pre_hook(pass_mlp_inputs)

    def _create_and_replace(
        self,
        mars_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **kwargs
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        
        # TODO: Add tqdm to this function and class
        # Print out what will be needed for the layer creation (creating quantization, preserving errors...)

        # Get rank and alpha from config
        r = mars_config.r
        alpha = mars_config.alpha

        projection_type = None
        quantize_base = False
        preserve_errors = False
        is_standalone = True

        if 'q_proj' in target_name:
            projection_type = 'q'
        elif 'k_proj' in target_name:
            projection_type = 'k'
        elif 'v_proj' in target_name:
            projection_type = 'v'
        elif 'gate_proj' in target_name:
            projection_type = 'gate'
        elif 'up_proj' in target_name:
            projection_type = 'up'
        elif 'o_proj' in target_name:
            projection_type = 'o'
        elif 'down_proj' in target_name:
            projection_type = 'down'

        self.validate_preserve_errors(mars_config)

        # Determine if adapter is shared or standalone
        if projection_type in mars_config.enabled_qkv:
            is_standalone = False
        elif mars_config.enabled_mlp and projection_type in ['gate', 'up']:
            is_standalone = False

        # Determine if adapter needs base layer quantization or not
        # By default optimization level has full quantization of base layers
        if self.optimization_level > 1:
            quantize_base = True
            if mars_config.modules_to_preserve_errors and projection_type in mars_config.modules_to_preserve_errors:
                preserve_errors = True
        # Else if partial quantization only quantize those layers specified
        elif self.optimization_level == 1:
            if mars_config.modules_to_quantize and projection_type in mars_config.modules_to_quantize: 
                quantize_base = True
                if mars_config.modules_to_preserve_errors and projection_type in mars_config.modules_to_preserve_errors:
                    preserve_errors = True

        module_config = {}

        module_config['target_name'] = target_name
        module_config['is_standalone'] = is_standalone
        module_config['shared_rank'] = mars_config.shared_r
        module_config['preserve_errors'] = preserve_errors
        module_config['quantize_base'] = quantize_base
        module_config['trainable_down'] = self.trainable_down
        module_config['onnx_export'] = self.only_export
        module_config['quant_n_bits'] = self.quant_n_bits

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                r,
                alpha,
                projection_type,
                **module_config
            )
        else:
            new_module = self._create_new_module(
                mars_config,
                adapter_name,
                target,
                r,
                alpha,
                projection_type,
                **module_config
            )

            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)

            self._replace_module(parent, target_name, new_module, target)
    
    @staticmethod
    def _replace_module(parent, child_name, new_module, child):

        projection_type = None

        if any(proj in child_name for proj in ['q_proj', 'k_proj', 'v_proj']):
            projection_type = 'qkv'
        elif any(proj in child_name for proj in ['gate_proj', 'up_proj']):
            projection_type = 'mlp'

        forward_hooks = {}
        forward_pre_hooks = {}

        if hasattr(child, '_forward_hooks'):
            forward_hooks = child._forward_hooks.copy()
            child._forward_hooks.clear()  # Remove hooks from original module

        if hasattr(child, '_forward_pre_hooks'):
            forward_pre_hooks = child._forward_pre_hooks.copy()
            child._forward_pre_hooks.clear()  # Remove hooks from original module

        setattr(parent, child_name, new_module)
        
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
            

        # Transfer hooks to the new module
        if hasattr(new_module, '_forward_hooks'):
            new_module._forward_hooks.update(forward_hooks)
        if hasattr(new_module, '_forward_pre_hooks'):
            new_module._forward_pre_hooks.update(forward_pre_hooks)

        # Set up shared QKV reference
        if projection_type == 'qkv' and hasattr(parent, 'shared_qkv'):
            new_module.shared_qkv = parent.shared_qkv

            # If dequantized module, then update the shared QKV
            if new_module.preserve_errors:
                new_module.shared_qkv._update_layer(new_module.svd_U, new_module.svd_S, new_module.projection_type)
                new_module.clear_svd_components()
        
        # Set up shared MLP reference
        if projection_type == 'mlp' and hasattr(parent, 'shared_mlp'):
            new_module.shared_mlp = parent.shared_mlp
            
            # If dequantized module, then update the shared MLP
            if new_module.preserve_errors:
                new_module.shared_mlp._update_layer(new_module.svd_U, new_module.svd_S, new_module.projection_type)
                new_module.clear_svd_components()

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "mars" in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    @staticmethod
    def _create_new_module(mars_config, adapter_name, target, rank, alpha, projection_type, **kwargs):
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
            alpha=alpha,
            projection_type=projection_type,
            mixture=mars_config.mixture,
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
    
    def validate_preserve_errors(self, mars_config):
        if mars_config.modules_to_preserve_errors is None:
            return

        qkv_errors = {'q', 'k', 'v'}
        mlp_errors = {'gate', 'down'}

        # Validate QKV
        if mars_config.enabled_qkv:
            present_qkv = [x for x in mars_config.modules_to_preserve_errors if x in qkv_errors]
            if len(present_qkv) > 1:
                raise ValueError(
                    f"Only one of ['q', 'k', 'v'] can be in `modules_to_preserve_errors` when shared QKV is enabled. Found: {present_qkv}"
                )

        # Validate MLP
        if mars_config.enabled_mlp:
            present_mlp = [x for x in mars_config.modules_to_preserve_errors if x in mlp_errors]
            if len(present_mlp) > 1:
                raise ValueError(
                    f"Only one of ['gate', 'down'] can be in `modules_to_preserve_errors` when shared MLP is enabled. Found: {present_mlp}"
                )

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        """Mark only adapter parameters as trainable."""
        for n, p in model.named_parameters():
            
            # If no adapter prefix in name
            if (self.prefix not in n):
                p.requires_grad = False

            # if we don't want trainable down projection
            if not self.trainable_down and self.prefix in n and any([m_name in n for m_name in ['down_project', 'shared_qkv', 'shared_mlp']]):
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
    
    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, MarsLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

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

    def _set_adapter_layers(self, enabled=True):
        """Set the enabled state of all adapter layers."""
        for module in self.model.modules():
            if isinstance(module, MarsLayer):
                module.disable_adapters = not enabled

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True) -> None:
        """Save the trainable adapter weights of the MarsModel.
        
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
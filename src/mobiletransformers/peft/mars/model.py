import os
import warnings

import torch
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from safetensors.torch import save_file
from torch.nn.modules import Module

from mobiletransformers.config.registry.architecture import ArchitectureSpec, resolve_architecture
from mobiletransformers.exceptions import UnsupportedModelError
from mobiletransformers.utils.logging import get_logger

from .layer import Linear, MarsLayer, SharedAttentionAdapter, SharedMLPAdapter
from .utils import TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING

logger = get_logger(__name__)

#: Roles whose projections share one :class:`SharedAttentionAdapter`, and one
#: :class:`SharedMLPAdapter`, respectively. Roles — not module names: the names are per-architecture
#: data on :class:`~mobiletransformers.config.registry.architecture.ArchitectureSpec`.
_QKV_ROLES = ("q", "k", "v")
_MLP_ROLES = ("gate", "up")


def _hidden_states_from(module: Module, args: tuple, kwargs: dict) -> torch.Tensor:
    """Extract the hidden-states input of an attention/MLP block's forward.

    A decoder's ``LlamaAttention.forward`` is called with ``hidden_states=`` as a keyword, which is
    what this code used to assume unconditionally (``kwargs["hidden_states"]``). ``BertAttention``
    and ``BertSelfAttention`` take it **positionally**, so the old form raised ``KeyError`` on every
    encoder forward. Accept either, and fail closed naming the module rather than letting a wrong
    tensor through.
    """
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and args:
        hidden_states = args[0]
    if not isinstance(hidden_states, torch.Tensor):
        raise UnsupportedModelError(
            f"{type(module).__name__}: MARS could not locate the hidden-states input of this module's "
            "forward (neither a `hidden_states` keyword nor a tensor first positional argument). "
            "The shared adapter cannot be applied without it."
        )
    return hidden_states


def _owns_any_child(module: Module, names: set[str]) -> bool:
    """True when ``module`` has a direct child submodule with one of ``names``."""
    return any(child in names for child, _ in module.named_children())


def _is_within(module_path: str, block_name: str) -> bool:
    """True when ``block_name`` is one of ``module_path``'s dotted components (or the leaf itself).

    ``model.layers.0.self_attn`` is within ``self_attn``; ``bert.encoder.layer.0.attention.self`` is
    within ``attention``. Component-wise, not substring: ``attention_probs`` must not match
    ``attention``.
    """
    return block_name in module_path.split(".")


def _compute_shared_qkv(module: Module, args: tuple, kwargs: dict) -> None:
    """Compute the block's shared QKV outputs once, for its projections to consume."""
    module.shared_qkv._shared_outputs = module.shared_qkv(_hidden_states_from(module, args, kwargs))
    return None


def _pass_qkv_inputs(module: Module, args: tuple) -> tuple:
    if module is None or not hasattr(module, "shared_qkv"):
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


def _compute_shared_mlp(module: Module, args: tuple, kwargs: dict) -> None:
    gate_out, up_out = module.shared_mlp(_hidden_states_from(module, args, kwargs))
    module.shared_mlp._shared_outputs = {"gate": gate_out, "up": up_out}
    return None


def _pass_mlp_inputs(module: Module, args: tuple) -> tuple:
    projection_type = getattr(module, "projection_type", None)
    if projection_type not in ("gate", "up"):
        return args
    shared_outputs = getattr(module.shared_mlp, "_shared_outputs", None)
    if shared_outputs is None or projection_type not in shared_outputs:
        return args
    shared_output = shared_outputs.pop(projection_type)
    return (shared_output,) + args


class MarsModel(BaseTuner):
    """
    PEFT model implementing the MARS (Multi-Adapter Rank Sharing) adapter technique on base models.
    """

    prefix: str = "mars"

    def __init__(
        self,
        model,
        peft_config: PeftConfig | dict[str, PeftConfig],
        adapter_name: str = "mars",
        low_cpu_mem_usage: bool = False,
    ) -> None:

        # Pre-initialization
        if peft_config[adapter_name].shared_r is None:
            peft_config[adapter_name].shared_r = peft_config[adapter_name].r

        self.trainable_down = True
        self.optimization_level = peft_config[adapter_name].optimization_level
        self.only_export = peft_config[adapter_name].onnx_export
        self.quant_n_bits = peft_config[adapter_name].quant_n_bits
        self.use_bnb = peft_config[adapter_name].use_bnb

        # Based on optimization level set configurations
        if peft_config[adapter_name].optimization_level == 0:
            self.trainable_down = True
        elif peft_config[adapter_name].optimization_level == 1:
            self.trainable_down = False
        elif peft_config[adapter_name].optimization_level == 2:
            self.trainable_down = True
        elif peft_config[adapter_name].optimization_level == 3:
            self.trainable_down = True
        elif peft_config[adapter_name].optimization_level == 4:
            self.trainable_down = False

        # Which module names carry which projection role is per-architecture DATA, resolved from the
        # class that was actually loaded (the head is part of the architecture identity — the same
        # reason `export/training_export.py` passes `architecture=type(model).__name__`). Resolved
        # BEFORE `super().__init__`, because BaseTuner's constructor runs the injection that consumes
        # it. Fails closed on an unknown architecture rather than silently applying decoder naming to
        # a model that does not use it — which is precisely the failure this replaces.
        self._arch_spec: ArchitectureSpec = resolve_architecture(
            model.config, architecture=type(model).__name__
        )

        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage)

    def _pre_injection_hook(self, model: Module, config: PeftConfig, adapter_name: str) -> None:

        enabled_qkv = getattr(config, "enabled_qkv", ("q", "k", "v"))

        # Map enabled projections to indices in tuple
        enabled_list = list(enabled_qkv)

        spec = self._arch_spec
        qkv_names = {n for n in (spec.module_name_for_role(r) for r in _QKV_ROLES) if n}
        mlp_names = {n for n in (spec.module_name_for_role(r) for r in _MLP_ROLES) if n}

        # Whether the user's target set actually reaches shared-adapter projections, decided by ROLE
        # rather than by the decoder-only literals `q_proj`/`gate_proj` this used to test for.
        any_qkv = any(spec.role_for_module(tm) in _QKV_ROLES for tm in config.target_modules)
        any_mlp = any(spec.role_for_module(tm) in _MLP_ROLES for tm in config.target_modules)

        # The anchor for a shared adapter is the module that DIRECTLY OWNS the projections, because
        # that is both where the hidden states arrive and the `parent` that `_replace_module` reads
        # `shared_qkv`/`shared_mlp` off. For a decoder that is `self_attn` itself; for BERT the
        # projections live one level deeper, in `attention.self`, so anchoring on the module named
        # `attention_module_name` would have attached the adapter to the wrong parent and silently
        # never wired it up. `attention_module_name` still scopes the search, so an unrelated module
        # that happens to own a `query`/`value` child is not mistaken for attention.
        qkv_anchors = [
            (name, module)
            for name, module in model.named_modules()
            if any_qkv and _owns_any_child(module, qkv_names) and _is_within(name, spec.attention_module_name)
        ]
        mlp_anchors = [
            (name, module)
            for name, module in model.named_modules()
            if any_mlp and _owns_any_child(module, mlp_names)
        ]
        if any_qkv and not qkv_anchors:
            raise UnsupportedModelError(
                f"MARS found no attention module owning any of {sorted(qkv_names)} under a "
                f"{spec.attention_module_name!r} block in {spec.architecture}. The shared QKV adapter "
                "would be a silent no-op; fix `projection_names`/`attention_module_name` for this "
                "architecture in the registry instead."
            )
        logger.info(
            "MARS shared adapters for %s: %d attention anchor(s), %d mlp anchor(s)",
            spec.architecture,
            len(qkv_anchors),
            len(mlp_anchors),
        )

        # --- shared QKV adapters, one per attention block -------------------------------------
        for _name, module in qkv_anchors:
            module.shared_qkv = SharedAttentionAdapter(
                hidden_size=model.config.hidden_size,
                rank=config.r,
                shared_rank=config.shared_r,
                alpha=config.alpha,
                enabled=enabled_list,
            )
            module.register_forward_pre_hook(_compute_shared_qkv, with_kwargs=True)

            for role in _QKV_ROLES:
                if role not in enabled_qkv:
                    continue
                proj_name = spec.module_name_for_role(role)
                proj = getattr(module, proj_name, None) if proj_name else None
                if proj is None:
                    continue
                proj.projection_type = role
                proj.register_forward_pre_hook(_pass_qkv_inputs)

        # --- shared MLP adapters -----------------------------------------------------------------
        for _name, module in mlp_anchors:
            module.shared_mlp = SharedMLPAdapter(
                hidden_size=model.config.hidden_size,
                rank=config.r,
                shared_rank=config.shared_r,
                alpha=config.alpha,
            )
            module.register_forward_pre_hook(_compute_shared_mlp, with_kwargs=True)

            for role in _MLP_ROLES:
                proj_name = spec.module_name_for_role(role)
                proj = getattr(module, proj_name, None) if proj_name else None
                if proj is None:
                    continue
                proj.projection_type = role
                proj.register_forward_pre_hook(_pass_mlp_inputs)

    def _create_and_replace(
        self, mars_config, adapter_name, target, target_name, parent, current_key, **kwargs
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # TODO: Add tqdm to this function and class
        # Print out what will be needed for the layer creation (creating quantization, preserving errors...)

        # Get rank and alpha from config
        r = mars_config.r
        alpha = mars_config.alpha

        quantize_base = False
        preserve_errors = False
        is_standalone = True

        # Registry data, not a literal ladder: `query` -> "q" on BERT, `q_proj` -> "q" on Llama.
        # When this returns None the module gets a standalone adapter — which is correct for a target
        # that genuinely has no shared role, and used to happen to EVERY encoder module because the
        # ladder tested decoder names only.
        projection_type = self._arch_spec.role_for_module(target_name)

        self.validate_preserve_errors(mars_config)

        # Determine if adapter is shared or standalone
        if projection_type in mars_config.enabled_qkv:
            is_standalone = False
        elif mars_config.enabled_mlp and projection_type in ["gate", "up"]:
            is_standalone = False

        # Determine if adapter needs base layer quantization or not
        # By default optimization level has full quantization of base layers
        if self.optimization_level > 1:
            quantize_base = True
            if (
                mars_config.modules_to_preserve_errors
                and projection_type in mars_config.modules_to_preserve_errors
            ):
                preserve_errors = True
        # Else if partial quantization only quantize those layers specified
        elif self.optimization_level == 1:
            if mars_config.modules_to_quantize and projection_type in mars_config.modules_to_quantize:
                quantize_base = True
                if (
                    mars_config.modules_to_preserve_errors
                    and projection_type in mars_config.modules_to_preserve_errors
                ):
                    preserve_errors = True

        module_config = {}

        module_config["target_name"] = target_name
        module_config["is_standalone"] = is_standalone
        module_config["shared_rank"] = mars_config.shared_r
        module_config["preserve_errors"] = preserve_errors
        module_config["quantize_base"] = quantize_base
        module_config["trainable_down"] = self.trainable_down
        module_config["onnx_export"] = self.only_export
        module_config["quant_n_bits"] = self.quant_n_bits
        module_config["use_bnb"] = self.use_bnb

        if isinstance(target, Linear):
            target.update_layer(adapter_name, r, alpha, projection_type, **module_config)
        else:
            new_module = self._create_new_module(
                mars_config, adapter_name, target, r, alpha, projection_type, **module_config
            )

            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)

            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):

        # Was a @staticmethod with the decoder module names inlined; it is only ever called from
        # `_create_and_replace` above, so binding it to the instance is what gives it access to the
        # architecture spec. Grouping is now by ROLE.
        role = self._arch_spec.role_for_module(child_name)
        projection_type = None
        if role in _QKV_ROLES:
            projection_type = "qkv"
        elif role in _MLP_ROLES:
            projection_type = "mlp"

        forward_hooks = {}
        forward_pre_hooks = {}

        if hasattr(child, "_forward_hooks"):
            forward_hooks = child._forward_hooks.copy()
            child._forward_hooks.clear()  # Remove hooks from original module

        if hasattr(child, "_forward_pre_hooks"):
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
        if hasattr(new_module, "_forward_hooks"):
            new_module._forward_hooks.update(forward_hooks)
        if hasattr(new_module, "_forward_pre_hooks"):
            new_module._forward_pre_hooks.update(forward_pre_hooks)

        # Set up shared QKV reference
        if projection_type == "qkv" and hasattr(parent, "shared_qkv"):
            new_module.shared_qkv = parent.shared_qkv

            # If dequantized module, then update the shared QKV
            if new_module.preserve_errors:
                new_module.shared_qkv._update_layer(
                    new_module.svd_U, new_module.svd_S, new_module.projection_type
                )
                new_module.clear_svd_components()

        # Set up shared MLP reference
        if projection_type == "mlp" and hasattr(parent, "shared_mlp"):
            new_module.shared_mlp = parent.shared_mlp

            # If dequantized module, then update the shared MLP
            if new_module.preserve_errors:
                new_module.shared_mlp._update_layer(
                    new_module.svd_U, new_module.svd_S, new_module.projection_type
                )
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
            **kwargs,
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

        qkv_errors = {"q", "k", "v"}
        mlp_errors = {"gate", "down"}

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
            if self.prefix not in n:
                p.requires_grad = False

            # if we don't want trainable down projection
            if (
                not self.trainable_down
                and self.prefix in n
                and any([m_name in n for m_name in ["down_project", "shared_qkv", "shared_mlp"]])
            ):
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
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
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

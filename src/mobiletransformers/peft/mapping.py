"""Base-layer -> adapter-tensor mapping builders (MARS and LoRA).

Migrated from ``trainer/utils.py`` (Migration Map S4). These are the two builders the #6 registry
resolves through :func:`mobiletransformers.config.registry.peft.build_adapter_mapping` — callers pass a
``PEFTMethod`` and never choose between them directly.

They are genuinely different walks, not one function with a flag: MARS tracks shared-module identity
and qkv/mlp adapter indices, LoRA is a flat ``named_modules`` scan.
"""

from __future__ import annotations

from mobiletransformers.config.registry.architecture import ArchitectureSpec, resolve_architecture


def _arch_spec_for(model) -> ArchitectureSpec:
    """The architecture spec behind a PEFT-wrapped model.

    Prefers the one the MARS tuner already resolved (``peft_model.base_model`` is the ``MarsModel``),
    so the mapping and the wrap can never disagree about which naming applies. Falls back to
    resolving from the wrapped HF model, which fails closed on an unknown architecture.
    """
    tuner = getattr(model, "base_model", None)
    spec = getattr(tuner, "_arch_spec", None)
    if isinstance(spec, ArchitectureSpec):
        return spec
    inner = getattr(tuner, "model", None) or model
    return resolve_architecture(inner.config, architecture=type(inner).__name__)


def _projection_role(model, base_layer_name: str) -> str | None:
    """``...attention.self.value.base_layer`` -> ``"v"``; ``...self_attn.v_proj.base_layer`` -> ``"v"``."""
    parent = base_layer_name.rsplit(".base_layer", 1)[0]
    return _arch_spec_for(model).role_for_module(parent)


def create_mars_adapter_mapping(model, shared_qkv=["q", "k", "v"], shared_mlp_enabled=True):
    """
    Create a JSON mapping of base layers to their corresponding adapters.
    Handles shared modules by their object identity and deduplicates them.

    NOTE: There could be problems with mapping with this function, depending on model architectures and position of the layers.

    Args:
        model: PyTorch model with PEFT adapters

    Returns:
        dict: Mapping of base layer names to their adapter configurations
    """
    mapping = {}

    # Track unique modules by their object id to handle shared modules
    module_id_to_name = {}

    def register_unique_module(module, full_name):
        """Register a module and return the canonical name for shared modules"""
        module_id = id(module)
        if module_id in module_id_to_name:
            # This module is shared, return the canonical name
            return module_id_to_name[module_id]
        else:
            # First time seeing this module
            module_id_to_name[module_id] = full_name
            return full_name

    # First pass: collect all modules and their paths
    all_modules = {}
    for name, module in model.named_modules():
        all_modules[name] = module

    # Find all base layers and their parent contexts
    base_layer_contexts = {}
    for name, module in all_modules.items():
        if "base_layer" in name:
            # Get parent path (everything before .base_layer)
            parent_path = name.rsplit(".base_layer", 1)[0]
            base_layer_contexts[name] = parent_path

    current_shared_mlp_name = None
    shared_mlp_counter = 0
    current_inter_mlp_name = None
    current_shared_qkv_name = None
    shared_qkv_counter = 0
    current_inter_qkv_name = None

    # For each base layer, find its adapters
    for base_layer_name, parent_path in base_layer_contexts.items():
        adapters = {}

        # Prefix renaming if needed
        if base_layer_name.startswith("base_model.model.model."):
            base_layer_name = base_layer_name.replace("base_model.model.model.", "backbone.model.")

        # Look for adapters in the parent context
        for module_name, module in all_modules.items():
            # Skip if not in the same parent context
            if not module_name.startswith(parent_path + "."):
                continue

            # Prefix renaming if needed
            if module_name.startswith("base_model.model.model."):
                module_name.replace("base_model.model.model.", "backbone.model.")

            # Get the relative path from parent
            relative_path = module_name[len(parent_path) + 1 :]

            # Apply categorization rules based on path patterns
            # Rule 1: shared_*.mars_down_* -> "shared_A"

            if relative_path.startswith("shared_") and ".mars_down_" in relative_path:
                canonical_name = register_unique_module(module, module_name)
                adapters["shared_A"] = canonical_name

                if relative_path.startswith("shared_mlp"):
                    current_shared_mlp_name = module_name

                elif relative_path.startswith("shared_qkv"):
                    current_shared_qkv_name = module_name

            # Rule 2: shared_*.mars -> "intermediate" (direct mars in shared)
            elif relative_path.startswith("shared_") and relative_path.endswith(".mars"):
                canonical_name = register_unique_module(module, module_name)
                adapters["intermediate"] = canonical_name

                if relative_path.startswith("shared_mlp"):
                    current_inter_mlp_name = module_name
                    shared_mlp_counter = 0
                    adapters["adapter_index"] = 0
                    shared_mlp_counter += 1
                elif relative_path.startswith("shared_qkv"):
                    current_inter_qkv_name = module_name
                    shared_qkv_counter = 0
                    adapters["adapter_index"] = 0
                    shared_qkv_counter += 1

            # Rule 3: up_project.mars -> "adapter_B"
            elif relative_path == "up_project.mars":
                canonical_name = register_unique_module(module, module_name)
                adapters["adapter_B"] = canonical_name

                if hasattr(module, "rank"):
                    adapters["rank"] = int(module.rank)
                else:
                    print(f"[WARNING] Could not find rank in {module_name}")
                if hasattr(module, "alpha"):
                    adapters["alpha"] = float(module.alpha)

            # Rule 4: down_project.mars -> "adapter_A"
            elif relative_path == "down_project.mars":
                canonical_name = register_unique_module(module, module_name)
                adapters["adapter_A"] = canonical_name

        # Check if we need to add pointer to shared or intermediate layer.
        #
        # `named_modules()` yields a shared object ONCE, under the first path that reaches it, so only
        # one projection per attention block ever finds `shared_qkv` in its own subtree. The others
        # need this back-pointer — without it the codec never learns that they share the tensor.
        #
        # Which projection this is used to be decided by `"q_proj" in base_layer_name`, a decoder-only
        # literal that matched nothing on an encoder: BERT's `value` silently lost its `shared_A`,
        # `intermediate` and `adapter_index` while `query` kept them, so the two projections of a
        # layer disagreed about whether they shared anything. Now resolved by ROLE through the
        # architecture registry, exactly as `peft/mars/model.py` does.
        base_role = _projection_role(model, base_layer_name)
        if "shared_A" not in adapters and "adapter_A" not in adapters:
            if base_role in shared_qkv:
                adapters["shared_A"] = current_shared_qkv_name
                adapters["intermediate"] = current_inter_qkv_name
                adapters["adapter_index"] = shared_qkv_counter
                shared_qkv_counter += 1
            elif shared_mlp_enabled and base_role in ("gate", "up"):
                adapters["shared_A"] = current_shared_mlp_name
                adapters["intermediate"] = current_inter_mlp_name
                adapters["adapter_index"] = shared_mlp_counter
                shared_mlp_counter += 1

        if adapters:
            mapping[base_layer_name] = adapters

    # with open('base_mapping.json', 'w') as f:
    #    json.dump(mapping, f)

    return mapping


def create_lora_mapping(peft_model) -> dict:
    """
    Creates a mapping from base layer names to their corresponding LoRA adapter layer names
    within a PEFT LoRA model.

    Args:
        peft_model (PeftModel): An instance of a PEFT LoRA model with applied LoRA adapters.

    Returns:
        dict: A dictionary where:
              - Keys are the full path names of the base layers with LoRA adapters.
              - Values are dictionaries containing the full path names to their
                corresponding 'lora_A' and 'lora_B' adapter modules.
    """
    from peft.tuners.lora import LoraLayer  # noqa: PLC0415

    peft_mapping = {}

    for module_path, module in peft_model.named_modules():
        # Identify modules that are LoRA-enabled layers
        if isinstance(module, LoraLayer):
            base_layer_name = module_path

            # Iterate through all adapter names for this LoRA layer (e.g., 'default')
            for adapter_name in module.lora_A.keys():
                lora_a_full_path = f"{base_layer_name}.lora_A.{adapter_name}"
                lora_b_full_path = f"{base_layer_name}.lora_B.{adapter_name}"

                # Prioritize 'default' adapter or use the first one found
                if adapter_name == "default" or base_layer_name not in peft_mapping:
                    peft_mapping[base_layer_name] = {
                        "adapter_A": lora_a_full_path,
                        "adapter_B": lora_b_full_path,
                    }

    return peft_mapping

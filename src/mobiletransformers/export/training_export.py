"""
Script that fetches the Huggingface LLM model and converts it into a ONNX graph compatible for artifact training generation.
"""

import argparse
import gc
import json
import textwrap
from pathlib import Path

import numpy as np
import onnx
import torch
import yaml
from onnx import TensorProto, helper, numpy_helper
from optimum.exporters.onnx import export
from peft import LoraConfig, PeftModel, PeftType, get_peft_model

# peft renamed its PeftType -> tuner-class registry in 0.15 (`PEFT_TYPE_TO_MODEL_MAPPING` ->
# `PEFT_TYPE_TO_TUNER_MAPPING`). The `ort-training-local` group floats `peft>=0.13` while
# `third_party/onnxruntime/manifest.json` records the tested pairing as 0.13.2, so a fresh resolve picks
# up a much newer peft and this module stopped importing at all — which fails the whole training stage
# before it does any work. Accept both spellings rather than pinning the profile to a 2024 peft.
try:  # peft < 0.15
    from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
except ImportError:  # peft >= 0.15
    from peft.peft_model import PEFT_TYPE_TO_TUNER_MAPPING as PEFT_TYPE_TO_MODEL_MAPPING
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# OnnxConfigWithLoss was REMOVED in optimum 2.1 (the optimum-onnx split; verified by
# spikes/optimum_migration/check_symbols.py). export() survives, so we keep the training-graph export
# on it with a VENDORED OnnxConfigWithLoss. Per-architecture *OnnxConfig classes now resolve via the
# architecture registry (#6/#9) — the old architectures[0] ladder is gone.
# PEFT registry (#6) — the old `train_method == "..."` chain is gone: the wire value is parsed ONCE
# into the PEFTMethod enum (fail-closed on an unknown method) and every branch keys off that member.
from mobiletransformers.config.constants import PEFTMethod, TaskType
from mobiletransformers.config.registry.architecture import resolve_architecture
from mobiletransformers.config.registry.peft import build_adapter_mapping
from mobiletransformers.export.embedding_export import add_pooling_to_onnx_model
from mobiletransformers.export.onnx_config_with_loss import OnnxConfigWithLoss
from mobiletransformers.export.registry import choose_task, supported_onnx_tasks
from mobiletransformers.peft.mars.config import MarsConfig
from mobiletransformers.peft.mars.model import MarsModel


def add_peft_type(name, value):
    """Dynamically add a new value to the PeftType enum."""
    setattr(PeftType, name, value)
    PeftType._value2member_map_[value] = name


# Add custom PEFT type dynamically
add_peft_type("MARS", "MARS")
PEFT_TYPE_TO_MODEL_MAPPING[PeftType("MARS")] = MarsModel

# All operators supported for training should be in https://onnx.ai/onnx/operators/index.html
from dotenv import load_dotenv
from onnxruntime.quantization import QuantType, quantize_dynamic

from mobiletransformers.peft.lora_xs.initialization_utils import find_and_initialize

load_dotenv()

from mobiletransformers.config.constants import TRAIN_CONFIG
from mobiletransformers.config.settings import get_settings


def get_layers_with_grad(model):
    """
    Collects layers with required grad and frozen parameter layers.
    """
    layers_with_grad = []
    layers_with_no_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layers_with_grad.append(name)
        else:
            layers_with_no_grad.append(name)
    return layers_with_grad, layers_with_no_grad


def ensure_training_mode_input(graph):
    """
    Add training mode boolean input to the graph for conditional flow.
    """

    training_mode_exists = any(input.name == "training_mode" for input in graph.input)
    if not training_mode_exists:
        # Add 'training_mode' input to the graph as a boolean tensor
        training_mode_input = helper.make_tensor_value_info("training_mode", TensorProto.BOOL, [1])
        graph.input.append(training_mode_input)


class OnnxInferenceWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model
        self.config = model.config
        self.training = False
        self.backbone.use_cache = True

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )


class OnnxTrainerWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model
        self.config = model.config
        self.training = True

    def forward(self, input_ids, attention_mask, position_ids, labels):
        return self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
        )


def compare_weights(model_path1, model_path2):
    """
    Compares weights of two models based on their initializers.
    """

    onnx_model = onnx.load(model_path1)
    INTIALIZERS = onnx_model.graph.initializer
    onnx_weights_1 = {}

    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights_1[initializer.name] = W

    del onnx_model
    onnx_model = onnx.load(model_path2)
    INTIALIZERS = onnx_model.graph.initializer
    onnx_weights_2 = {}

    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights_2[initializer.name] = W

        if initializer.name not in onnx_weights_1 or initializer.name not in onnx_weights_2:
            print(f"MISMATCH IN INITIALIZERS - missing {initializer.name}")
            continue

        are_equal = np.array_equal(onnx_weights_1[initializer.name], onnx_weights_2[initializer.name])

        if not are_equal:
            # print(are_equal)
            print("Not equal")
            print(initializer.name)
            # print(onnx_weights_1[initializer.name])
            # print(onnx_weights_2[initializer.name])
            # print(onnx_weights_1[initializer.name].shape)
            # print(onnx_weights_2[initializer.name].shape)

    onnx.save(onnx_model, "model.onnx", location="model.onnx_data", save_as_external_data=True)


def trim_initializers(model_path1):
    """
    Removes all the layers of initializers that start with:
    - ONNX basic nodes with "/"
    - ONNX nodes with "onnx::"
    """

    onnx_model = onnx.load(model_path1)
    INTIALIZERS = onnx_model.graph.initializer
    onnx_weights_1 = {}

    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)

        onnx_weights_1[initializer.name] = W

        if initializer.name.startswith("/") or initializer.name.startswith("onnx::"):
            print("Removed:")
            print(initializer.name)
            onnx_model.graph.initializer.remove(initializer)
        else:
            print("Not removed:")
            print(initializer.name)

    onnx.save(onnx_model, "model.onnx", location="model.onnx_data", save_as_external_data=True)


def inspect_weights(model_path, only_trainable=False):
    """
    Inspects the weights of the model provided.
    """

    onnx_model = onnx.load(model_path)
    INTIALIZERS = onnx_model.graph.initializer

    for param in INTIALIZERS:
        print(f"Layer name: {param.name}")


def apply_metadata(model_path, model_id):
    """
    Load ONNX model, apply metadata to both model and graph, and resave it (replacing original files).

    Args:
        model_path (Path): Path to the .onnx model file
        model_id (str): Model ID to add as metadata

    Returns:
        Path: Path to the updated model file
    """
    # Load the model
    model = onnx.load(str(model_path))

    # Remove existing model_id metadata from model if it exists
    to_remove = []
    for i, prop in enumerate(model.metadata_props):
        if prop.key == "model_id":
            to_remove.append(i)

    # Remove in reverse order to maintain indices
    for i in reversed(to_remove):
        del model.metadata_props[i]

    # Add metadata to model level
    model_metadata_entry = onnx.StringStringEntryProto()
    model_metadata_entry.key = "model_id"
    model_metadata_entry.value = str(model_id)
    model.metadata_props.append(model_metadata_entry)

    # Remove existing model_id metadata from graph if it exists
    graph_to_remove = []
    for i, prop in enumerate(model.graph.metadata_props):
        if prop.key == "model_id":
            graph_to_remove.append(i)

    # Remove in reverse order to maintain indices
    for i in reversed(graph_to_remove):
        del model.graph.metadata_props[i]

    # Add metadata to graph level
    graph_metadata_entry = onnx.StringStringEntryProto()
    graph_metadata_entry.key = "model_id"
    graph_metadata_entry.value = str(model_id)
    model.graph.metadata_props.append(graph_metadata_entry)

    return model


def preprocess_model(model: torch.nn.Module, epsilon_high=1e-8, epsilon_low=1e-10):
    """
    Add a really small epsilon to the model parameters if they are all zeroes.
    This is to prevent the ONNX from not saving the extra weights, as they need to be included as initializers.
    """
    for _, param in model.named_parameters():
        if torch.all(param.data == 0):
            random_values = torch.rand_like(param.data)
            random_values = (epsilon_high - epsilon_low) * random_values + epsilon_low
            param.data += random_values

    return model


def optimum_hf_export(
    model_id,
    model_output="onnx_models",
    training_mode=False,
    train_method="lora",
    lora_target=["q_proj", "k_proj"],
    lora_rank=4,
    lora_alpha=4,
    quantize=True,
    weight_type=QuantType.QInt8,
    peft_config={},
    specific_peft_config={},
    postprocess=False,
    exclude_extra_layers=["embed_head"],
    exclude_specific=False,
    exclude_specific_layers=[],
    opset=20,
    task_type="text-generation",
    add_pooling=False,
):
    """
    Exports the model from Huggingface to an ONNX model representation.
    """

    if task_type == "text-generation":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, token=get_settings().hf_token
        )
    else:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=get_settings().hf_token)
    config = AutoConfig.from_pretrained(model_id, token=get_settings().hf_token)

    # Registry-driven dispatch (no architectures[0] ladder): the architecture registry maps the HF
    # architecture -> its Optimum OnnxConfig class, and choose_task routes task selection. Unknown
    # architectures fail closed inside resolve_architecture. Adding one is a registry entry, not an elif.
    spec = resolve_architecture(config)
    resolved_task = choose_task(supported_onnx_tasks(config.model_type), override=task_type)
    onnx_config_class = spec.load_onnx_config_class()
    if spec.task == TaskType.FEATURE_EXTRACTION:
        ocl = onnx_config_class(config, task=resolved_task)
    else:
        ocl = onnx_config_class(
            config, task=resolved_task, use_past=not training_mode, use_past_in_inputs=not training_mode
        )

    lora_config = None
    lora_model = None

    if training_mode:
        ocl = OnnxConfigWithLoss(ocl)

    onnx_path = Path(f"{model_output}/model.onnx")

    # #6: one fail-closed parse at the boundary; PEFTMethod(...) raises ValueError on an unknown
    # method rather than silently falling through every branch and leaving `lora_model` unbound.
    peft_method = PEFTMethod(train_method)

    if training_mode and peft_method is PEFTMethod.LORA:
        # Apply LoRA to the model
        lora_config = LoraConfig(
            r=lora_rank, target_modules=lora_target, task_type="CAUSAL_LM", **peft_config
        )
        lora_model = PeftModel(model, lora_config, adapter_name="lora")
    elif training_mode and peft_method is PEFTMethod.LORA_XS:
        # TODO: Add specific PEFT config
        lora_config = LoraConfig(
            r=lora_rank, target_modules=lora_target, task_type="CAUSAL_LM", **peft_config
        )
        lora_model = get_peft_model(model, lora_config)
        adapter_name = "default"
        peft_config_dict = {}
        reconstruct_dict = {
            "reconstruction_type": "svd",
            "reconstr_mode": "separated",
            "half_init_dec": False,
            "replacement_module_random_init": False,
            "r_squared": True,
            "svd": {"rank": lora_rank, "n_iter": 10, "random_state": 42},
        }
        peft_config_dict[adapter_name] = lora_config
        find_and_initialize(model, peft_config_dict, adapter_name, "svd", reconstruct_dict, None)
    elif training_mode and peft_method is PEFTMethod.MARS:
        mars_config = MarsConfig(
            peft_type="MARS",
            r=lora_rank,
            alpha=lora_alpha,
            onnx_export=True,  # always needs to be True for export
            target_modules=lora_target,  # Target specific model layers
            task_type=None,
            **specific_peft_config,
        )

        lora_model = get_peft_model(model, mars_config, adapter_name="mars")
    elif training_mode and peft_method is PEFTMethod.ALL:
        # Make only linear layers trainable
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear)):
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        lora_model = model
    elif not training_mode or peft_method is PEFTMethod.NOLORA:
        lora_model = model

    # #6 A3: the ONE adapter-mapping entry point — the registry resolves each method's builder, so
    # this no longer re-derives "which mapping function" from the method string.
    mapping = {}
    if training_mode:
        mapping_kwargs = (
            {"shared_qkv": mars_config.enabled_qkv, "shared_mlp_enabled": mars_config.enabled_mlp}
            if peft_method is PEFTMethod.MARS
            else {}
        )
        mapping = build_adapter_mapping(peft_method, lora_model, **mapping_kwargs)

    if training_mode:
        if peft_method is not PEFTMethod.ALL:
            my_model = OnnxTrainerWrapper(lora_model.base_model.model)
        else:
            my_model = OnnxTrainerWrapper(lora_model)
        my_model.train()
    elif task_type == "text-generation":
        my_model = OnnxInferenceWrapper(lora_model)
        my_model.eval()
    else:
        # Infer from the model
        my_model = lora_model
        my_model.eval()

    # Preprocessing methods
    if training_mode:
        my_model = preprocess_model(my_model)

    # Trainable count
    trainable_count = count_trainable_parameters(my_model)

    export(my_model, ocl, onnx_path, opset, do_constant_folding=not training_mode)

    # Apply some metadata to model
    onnx_model = apply_metadata(onnx_path, model_id)

    # Add pooling operations to the embedding model and save
    if task_type == "feature-extraction" and add_pooling:
        add_pooling_to_onnx_model(onnx_model, model_id, f"{model_output}/embedding_model.onnx")

    # Save gradient layer names
    if training_mode:
        # Get layers with gradients in the LoRA model
        grad_layers, no_grad_layers = get_layers_with_grad(my_model)
        with open(f"{model_output}/training_config.json", "w+", encoding="utf-8") as f:
            json.dump(
                {
                    "requires_grad": grad_layers,
                    "frozen_params": no_grad_layers,
                    "peft_mapping": mapping,
                    "trainable_parameter_count": trainable_count,
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "peft_target": lora_target,
                },
                f,
                ensure_ascii=False,
            )

    del my_model
    my_model = None
    gc.collect()

    # Apply dynamic quantization to non-trainable layers
    if quantize:
        lora_target = [] if not training_mode else lora_target
        onnx_dynamic_quantization(
            onnx_model,
            onnx_path.absolute().as_posix(),
            f"{model_output}/quant_model.onnx",
            # Keep the PEFT adapters OUT of quantization. This argument was commented out, so with
            # `--quant int4` the quantizer packed the LoRA A/B weights along with the frozen base: no
            # float trainable initializer survived, `requires_grad` resolved to only `*_quantized`
            # companions, and `generate_artifacts` died — first on "Cannot compute the partial
            # derivative for '…weight_quantized'", then, once those were correctly excluded, on an
            # empty trainable set (`IndexError` in optim.py). A quantized base with float adapters is
            # the project's premise (the base/trainable external split #8/#9 are built around); the
            # line above computes `lora_target` for exactly this and then discarded it.
            exclude_weights=lora_target,
            weight_type=weight_type,
            exclude_extra_layers=exclude_extra_layers,
            exclude_specific=exclude_specific,
            exclude_specific_layers=exclude_specific_layers,
        )

        # Add pooling operations to the quantized embedding model and save
        if task_type == "feature-extraction" and add_pooling:
            add_pooling_to_onnx_model(
                f"{model_output}/quant_model.onnx", model_id, f"{model_output}/embedding_quant_model.onnx"
            )


def onnx_dynamic_quantization(
    onnx_model,
    onnx_model_path,
    onnx_model_quant_output,
    weight_type=QuantType.QInt16,
    exclude_weights=[],
    exclude_extra_layers=[],
    exclude_specific=False,
    exclude_specific_layers=[],
):

    nodes_to_not_quantize = []

    # Exclude trainable nodes
    for param in onnx_model.graph.node:
        if not exclude_specific:
            if any((allowed_layer in param.name) for allowed_layer in exclude_weights):
                nodes_to_not_quantize.append(param.name)
        else:
            if any((allowed_layer in param.name) for allowed_layer in exclude_specific_layers):
                nodes_to_not_quantize.append(param.name)

        if any(allowed_layer in param.name for allowed_layer in exclude_extra_layers):
            nodes_to_not_quantize.append(param.name)
        for input_weight in param.input:
            if any(allowed_layer in input_weight for allowed_layer in exclude_extra_layers):
                nodes_to_not_quantize.append(param.name)
                break

    # Does not work
    # quant_pre_process(onnx_model_path, f"pre_{onnx_model_quant_output}", save_as_external_data=True, all_tensors_to_one_file=True, external_data_location=f"pre_{onnx_model_quant_output}")

    del onnx_model
    gc.collect()

    quantize_dynamic(
        extra_options={
            "ActivationSymmetric": False,  # True for inference speed. False may keep more accuracy.
            "WeightSymmetric": False,  # True for inference speed. False may keep more accuracy.
            "EnableSubgraph": False,  # True for more quant.
            "ForceQuantizeNoInputCheck": True,  # True for more quant.
            "MatMulConstBOnly": True,  # False for more quant. Sometime, the inference speed may get worse. Keep this True in case of training graph.
        },
        nodes_to_exclude=nodes_to_not_quantize,
        model_input=onnx_model_path,
        model_output=onnx_model_quant_output,
        per_channel=True,
        use_external_data_format=True,
        weight_type=weight_type,
        reduce_range=False,
    )


def count_trainable_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_extra_options(kv_pairs):
    if "exclude_extra_layers" in kv_pairs:
        op_types_to_quantize = ()
        for op_type in kv_pairs["exclude_extra_layers"].split("/"):
            op_types_to_quantize += (op_type,)
        kv_pairs["exclude_extra_layers"] = op_types_to_quantize
    if "exclude_specific_layers" in kv_pairs:
        op_types_to_quantize = ()
        for op_type in kv_pairs["exclude_specific_layers"].split("/"):
            op_types_to_quantize += (op_type,)
        kv_pairs["exclude_specific_layers"] = op_types_to_quantize


def parse_argument_list(targt):
    return targt.split("/")


def parse_extra_options(extra_options: list[str]) -> dict[str, str]:
    """
    Parse additional options in KEY=VALUE format into a dictionary.
    """
    options_dict = {}
    for option in extra_options:
        if "=" in option:
            key, value = option.split("=", 1)
            options_dict[key] = value
        else:
            raise ValueError(f"Invalid format for extra option '{option}'. Use KEY=VALUE format.")

    print(f"Extra options: {options_dict}")
    check_extra_options(options_dict)
    return options_dict


def load_config_from_file(config_file: str):
    """Load configurations from a YAML file into a dictionary."""
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config[TRAIN_CONFIG]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Exporting the HF model into a ONNX graph compatible for on-device training.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--model_id", type=str, help="Identifier for the model to be converted.")
    parser.add_argument("--output", type=str, help="Path to the model output location.")
    parser.add_argument(
        "--training_mode",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether the model is in training mode. Default is True.",
    )
    parser.add_argument(
        "--train_method",
        type=str,
        choices=["lora", "lora-xs", "mars", "nolora"],
        default="lora",
        help="The training method to use, such as LoRA. Default is 'lora'.",
    )
    parser.add_argument(
        "--lora_target",
        type=parse_argument_list,
        default=["q_proj", "k_proj"],
        help="Target layers for LoRA, provided as a list. Default is ['q_proj', 'k_proj'].",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Rank for the given LoRA method. Default is 16."
    )
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for the PEFT method.")
    parser.add_argument(
        "--quantize", type=bool, default=True, help="Whether to apply quantization. Default is True."
    )
    parser.add_argument(
        "--weight_type",
        type=lambda x: QuantType[x],
        choices=list(QuantType),
        default=QuantType.QUInt8,
        help="The quantization weight type, e.g., QUInt8. Default is QuantType.QUInt8. Recommended QInt8 so it stays in the same quantization domain as inference model.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["text-generation", "feature-extraction"],
        default="text-generation",
        help="Task type to build the model for.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments.",
    )
    parser.add_argument(
        "--extra_options",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            postprocess = False : Whether to try to do operator fusion after creating the graph. The applied fused operators should be supported by training.
            opset = 20 : Opset version for model operators.
            exclude_extra_layers = layer1/layer2... : Extra layers to further exclude from the quantization. Keywords should be separated by "/".
            """),
    )

    args = parser.parse_args()

    user_extra_options = {}
    default_extra_options = {
        "postprocess": False,
        "add_pooling": True,
        "opset": 20,
        "exclude_extra_layers": [],
        "exclude_specific": False,
        "exclude_specific_layers": [],
    }

    config_dict = None

    if args.config_file:
        config_dict = load_config_from_file(args.config_file)

        args.peft_config = config_dict["peft_config"]
        setattr(args, config_dict["train_method"], config_dict[config_dict["train_method"]])

        # Override any command-line argument with values from the config file
        for key, value in config_dict.items():
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)
        # Override any command-line argument with values from the config file
        for key, value in config_dict["extra_options"].items():
            default_extra_options[key] = value
        args.weight_type = QuantType[config_dict["weight_type"]]
    else:
        user_extra_options = parse_extra_options(args.extra_options)
    args.extra_options = {**default_extra_options, **user_extra_options}

    return args


if __name__ == "__main__":
    args = parse_arguments()

    method = getattr(args, "train_method", "lora")
    peft_config = getattr(args, "peft_config", None)
    specific_peft_config = getattr(args, method, None)

    print(f"{TRAIN_CONFIG} arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if peft_config:
        print("PEFT arguments:")
        for arg, value in peft_config.items():
            print(f"{arg}: {value}")

    if specific_peft_config:
        print("Extra specific PEFT arguments:")
        for arg, value in specific_peft_config.items():
            print(f"{arg}: {value}")

    optimum_hf_export(
        model_id=args.model_id,
        model_output=args.output,
        train_method=args.train_method,
        training_mode=args.training_mode,
        lora_target=args.lora_target,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        quantize=args.quantize,
        weight_type=args.weight_type,
        task_type=args.task_type,
        peft_config=peft_config,
        specific_peft_config=specific_peft_config,
        **args.extra_options,
    )

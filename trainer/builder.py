"""
Script that fetches the Huggingface LLM model and converts it into a ONNX graph compatible for artifact training generation.
"""

import argparse, yaml
import json, gc, os
import textwrap
from typing import Dict, List
import torch
from pathlib import Path
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from optimum.exporters.onnx import OnnxConfigWithLoss, export
from optimum.exporters.onnx.model_configs import LlamaOnnxConfig, GemmaOnnxConfig, Phi3OnnxConfig

from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel, LoraConfig, get_peft_model
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import PeftType
from optimization.mars.config import MarsConfig
from optimization.mars.modelv2 import MarsModel

from trainer.utils import create_mars_adapter_mapping, create_lora_mapping

def add_peft_type(name, value):
    """Dynamically add a new value to the PeftType enum."""
    setattr(PeftType, name, value)
    PeftType._value2member_map_[value] = name

# Add custom PEFT type dynamically
add_peft_type("MARS", "MARS")
PEFT_TYPE_TO_MODEL_MAPPING[PeftType("MARS")] = MarsModel

from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
#from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer, DefaultWeightOnlyQuantConfig

from onnxruntime.transformers.onnx_model import OnnxModel

from optimization.lora_xs.initialization_utils import find_and_initialize

# All operators supported for training should be in https://onnx.ai/onnx/operators/index.html
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization

from dotenv import load_dotenv

load_dotenv()

from tools.parser_config import TRAIN_CONFIG

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
        self.backbone.use_cache=True
    
    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=True)

class OnnxTrainerWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model
        self.config = model.config
        self.training = True
    
    def forward(self, input_ids, attention_mask, position_ids, labels):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)

def compare_weights(model_path1, model_path2):
    """
    Compares weights of two models based on their initializers.
    """

    onnx_model   = onnx.load(model_path1)
    INTIALIZERS  = onnx_model.graph.initializer
    onnx_weights_1 = {}

    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights_1[initializer.name] = W

    del onnx_model
    onnx_model   = onnx.load(model_path2)
    INTIALIZERS  = onnx_model.graph.initializer
    onnx_weights_2 = {}

    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights_2[initializer.name] = W

        if initializer.name not in onnx_weights_1 or initializer.name not in onnx_weights_2:
            print(f"MISMATCH IN INITIALIZERS - missing {initializer.name}")
            continue

        are_equal = np.array_equal(onnx_weights_1[initializer.name], onnx_weights_2[initializer.name])

        if not are_equal:
            #print(are_equal)
            print("Not equal")
            print(initializer.name)
            #print(onnx_weights_1[initializer.name])
            #print(onnx_weights_2[initializer.name])
            #print(onnx_weights_1[initializer.name].shape)
            #print(onnx_weights_2[initializer.name].shape)
    
    onnx.save(onnx_model, "model.onnx", location="model.onnx_data", save_as_external_data=True)


def trim_initializers(model_path1):
    """
    Removes all the layers of initializers that start with:
    - ONNX basic nodes with "/"
    - ONNX nodes with "onnx::"
    """

    onnx_model   = onnx.load(model_path1)
    INTIALIZERS  = onnx_model.graph.initializer
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

    onnx_model   = onnx.load(model_path)
    INTIALIZERS  = onnx_model.graph.initializer

    for param in INTIALIZERS:
        print(f"Layer name: {param.name}")

def postprocess_model(model):
    """
    TODO: Work in progress still if needed, avoid using this function.
    Postprocessing of the model, adding fusion.
    """
    
    onx_m = OnnxModel(model)
    updated_model = FusionLayerNormalization(onx_m)
    updated_model.apply()
    return updated_model.model

def preprocess_model(model : torch.nn.Module, epsilon_high=1e-8, epsilon_low=1e-10):
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

def optimum_hf_export(model_id,
                      model_output="onnx_models",
                      training_mode = False,
                      train_method = "lora",
                      lora_target=["q_proj", "k_proj"],
                      lora_rank=4,
                      quantize=True,
                      weight_type=QuantType.QInt8,
                      peft_config={},
                      specific_peft_config={},
                      postprocess=False,
                      exclude_extra_layers = ["embed_head"],
                      exclude_specific=False,
                      exclude_specific_layers=[],
                      opset=20):
    """
    Exports the model from Huggingface to an ONNX model representation.
    - `model_id` - model id of Huggingface model
    - `model_output` - path to model output directory
    - `training_mode`- create model for training or inference
    - `lora_target` - which layers to apply LoRA to
    - `quantize` - add dynamic quantization to layers which do not need gradient updates
    """

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=os.environ["HF_TOKEN"])
    config = AutoConfig.from_pretrained(model_id, token=os.environ["HF_TOKEN"])

    if config.architectures[0] == "LlamaForCausalLM":
        ocl = LlamaOnnxConfig(config, task="text-generation", use_past=not training_mode, use_past_in_inputs=not training_mode)
    elif config.architectures[0] == "GemmaForCausalLM" or config.architectures[0] == "Gemma2ForCausalLM":
        ocl = GemmaOnnxConfig(config, task="text-generation", use_past=not training_mode, use_past_in_inputs=not training_mode)
    elif config.architectures[0] == "Phi3ForCausalLM":
        ocl = Phi3OnnxConfig(config, task="text-generation", use_past=not training_mode, use_past_in_inputs=not training_mode)

    lora_config = None
    lora_model = None

    if training_mode:
        ocl = OnnxConfigWithLoss(ocl)

    onnx_path = Path(f"{model_output}/model.onnx")

    if training_mode and train_method == "lora":
        # Apply LoRA to the model
        lora_config = LoraConfig(
            r=lora_rank,
            target_modules=lora_target,
            task_type="CAUSAL_LM",
            **peft_config
        )
        lora_model = PeftModel(model, lora_config, adapter_name="lora")
    elif training_mode and train_method == "lora-xs":
        # TODO: Add specific PEFT config
        lora_config = LoraConfig(
            r=lora_rank,
            target_modules=lora_target,
            task_type="CAUSAL_LM",
            **peft_config
        )
        lora_model = get_peft_model(model, lora_config)
        adapter_name = "default"
        peft_config_dict = {}
        reconstruct_dict = {
            'reconstruction_type': "svd",
            'reconstr_mode': "separated",
            'half_init_dec': False,
            'replacement_module_random_init': False,
            'r_squared': True,
            'svd': {
                'rank': lora_rank,
                'n_iter': 10,
                'random_state': 42
            }
        }
        peft_config_dict[adapter_name] = lora_config
        find_and_initialize(model, peft_config_dict, adapter_name, "svd", reconstruct_dict, None)
    elif training_mode and train_method == "mars":
        mars_config = MarsConfig(
            peft_type="MARS",
            r=lora_rank,
            onnx_export=True, # always needs to be True for export
            target_modules=lora_target,  # Target specific model layers
            task_type=None,
            **specific_peft_config
        )

        lora_model = get_peft_model(model, mars_config, adapter_name="mars")
    elif not training_mode or train_method == "nolora":
        lora_model = model

    mapping = {}
    if train_method == "mars":
        mapping = create_mars_adapter_mapping(lora_model, mars_config.enabled_qkv, mars_config.enabled_mlp)
    elif train_method == "lora":
        mapping = create_lora_mapping(lora_model)

    if training_mode:
        my_model = OnnxTrainerWrapper(lora_model.base_model.model)
        my_model.train()
    else:
        my_model = OnnxInferenceWrapper(lora_model)
        my_model.eval()

    # Preprocessing methods
    if training_mode:
        my_model = preprocess_model(my_model)

    export(my_model, ocl, onnx_path, opset, do_constant_folding=not training_mode)

    # Save gradient layer names
    if training_mode:
        # Get layers with gradients in the LoRA model
        grad_layers, no_grad_layers = get_layers_with_grad(my_model)
        with open(f"{model_output}/training_config.json", "w+", encoding="utf-8") as f:
            json.dump({
                "requires_grad": grad_layers,
                "frozen_params": no_grad_layers,
                "peft_mapping": mapping
            }, f, ensure_ascii=False)

    # Post-processing
    if training_mode and postprocess:
        my_model = onnx.load(onnx_path)
        my_model = postprocess_model(my_model)
        onnx_path = Path(f"{model_output}/opt_model.onnx")
        my_model.save_model_to_file(output_path=onnx_path.absolute().as_posix(), use_external_data_format=True)
    
    del my_model
    my_model = None
    gc.collect()

    # Apply dynamic quantization to non-trainable layers
    if quantize:
        lora_target = [] if not training_mode else lora_target
        onnx_dynamic_quantization(onnx_path.absolute().as_posix(),
                                  f"{model_output}/quant_model.onnx",
                                  #exclude_weights=lora_target,
                                  weight_type=weight_type,
                                  exclude_extra_layers=exclude_extra_layers,
                                  exclude_specific=exclude_specific,
                                  exclude_specific_layers=exclude_specific_layers)

def onnx_dynamic_quantization(onnx_model_path,
                              onnx_model_quant_output,
                              weight_type=QuantType.QInt16,
                              exclude_weights=[],
                              exclude_extra_layers=[],
                              exclude_specific=False,
                              exclude_specific_layers=[]):

    onnx_model = onnx.load(onnx_model_path)

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
    #quant_pre_process(onnx_model_path, f"pre_{onnx_model_quant_output}", save_as_external_data=True, all_tensors_to_one_file=True, external_data_location=f"pre_{onnx_model_quant_output}")

    del onnx_model
    gc.collect()

    quantize_dynamic(
        extra_options={
                'ActivationSymmetric': False,         # True for inference speed. False may keep more accuracy.
                'WeightSymmetric': False,                 # True for inference speed. False may keep more accuracy.
                'EnableSubgraph': False,                   # True for more quant.
                'ForceQuantizeNoInputCheck': True,       # True for more quant.
                'MatMulConstBOnly': True                 # False for more quant. Sometime, the inference speed may get worse. Keep this True in case of training graph.
        },
        nodes_to_exclude=nodes_to_not_quantize,
        model_input=onnx_model_path,
        model_output=onnx_model_quant_output,
        per_channel=True,
        use_external_data_format=True,
        weight_type=weight_type,
        reduce_range=False
    )

def check_extra_options(kv_pairs):
    if "exclude_extra_layers" in kv_pairs:
        op_types_to_quantize = ()
        for op_type in kv_pairs["exclude_extra_layers"].split("/"):
            op_types_to_quantize += (op_type, )
        kv_pairs["exclude_extra_layers"] = op_types_to_quantize
    if "exclude_specific_layers" in kv_pairs:
        op_types_to_quantize = ()
        for op_type in kv_pairs["exclude_specific_layers"].split("/"):
            op_types_to_quantize += (op_type, )
        kv_pairs["exclude_specific_layers"] = op_types_to_quantize

def parse_argument_list(targt):
    return targt.split('/')

def parse_extra_options(extra_options: List[str]) -> Dict[str, str]:
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
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config[TRAIN_CONFIG]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Exporting the HF model into a ONNX graph compatible for on-device training.", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model_id",
        type=str,
        help="Identifier for the model to be converted."
    )
    parser.add_argument( 
        "--output",
        type=str,
        help="Path to the model output location."
    )
    parser.add_argument(
        "--training_mode",
        type=bool,
        default=True,
        help="Whether the model is in training mode. Default is True."
    )
    parser.add_argument(
        "--train_method",
        type=str,
        choices=["lora", "lora-xs", "mars", "nolora"],
        default="lora",
        help="The training method to use, such as LoRA. Default is 'lora'."
    )
    parser.add_argument(
        "--lora_target",
        type=parse_argument_list,
        default=["q_proj", "k_proj"],
        help="Target layers for LoRA, provided as a list. Default is ['q_proj', 'k_proj']."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank for the given LoRA method. Default is 16."
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=True,
        help="Whether to apply quantization. Default is True."
    )
    parser.add_argument(
        "--weight_type",
        type=lambda x: QuantType[x],
        choices=list(QuantType),
        default=QuantType.QUInt8,
        help="The quantization weight type, e.g., QUInt8. Default is QuantType.QUInt8. Recommended QInt4 so it stays in the same quantization domain as inference model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments."
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
            """
            )
    )

    args = parser.parse_args()

    user_extra_options = {}
    default_extra_options = {
        "postprocess" : False,
        "opset" : 20,
        "exclude_extra_layers": [],
        "exclude_specific": False,
        "exclude_specific_layers": []
    }
    
    config_dict = None

    if args.config_file:
        config_dict = load_config_from_file(args.config_file)

        setattr(args, "peft_config", config_dict["peft_config"])
        setattr(args, config_dict["train_method"], config_dict[config_dict["train_method"]])
        
        # Override any command-line argument with values from the config file
        for key, value in config_dict.items():
            
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)
        # Override any command-line argument with values from the config file
        for key, value in config_dict["extra_options"].items():
            default_extra_options[key] = value
        setattr(args, "weight_type", QuantType[config_dict["weight_type"]])
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
    
    print("PEFT arguments:")
    for arg, value in peft_config.items():
        print(f"{arg}: {value}")

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
        quantize=args.quantize,
        weight_type=args.weight_type,
        peft_config=peft_config,
        specific_peft_config=specific_peft_config,
        **args.extra_options
    )
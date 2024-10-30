import os, json, gc
import torch
from pathlib import Path
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from optimum.exporters.onnx import main_export, onnx_export_from_model, OnnxConfigWithLoss, export
from optimum.exporters.onnx.model_configs import LlamaOnnxConfig, GemmaOnnxConfig

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from onnxruntime.quantization import quantize_dynamic, QuantType

# TODO: IMPORTANT! If training add training=torch.onnx.TrainingMode.PRESERVE, to the onnx_export function at the end L#586
from optimum.exporters.onnx.convert import export_pytorch

from optimization.lora_xs.initialization_utils import find_and_initialize

model_id = "TinyLlama/TinyLlama_v1.1"#"TinyLlama/TinyLlama_v1.1"#"google/gemma-2b-it"#
model_name = model_id.split("/")[1]

# TODO: Add Huggingface token
#os.environ["HF_TOKEN"] = "TODO"

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

def static_quantization(onnx_path):
    """
    TODO Static quantization if needed
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ort_model = ORTQuantizer(onnx_path)
    qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    

def remap_graph(onnx_model_path):
    """
    Function that would remap graph inputs for past and present KV caching.
    Adding conditional input based on training_mode input which indicates if we are using model for training or inference.
    This is harder to implement as each architecture does KV concatenation differently and would affect the output.
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph

    ensure_training_mode_input(graph)

    concat_nodes = []

    for i, node in enumerate(graph.node):
        
        if node.op_type != "Concat":
            continue

        past_inputs = []
        other_node_inputs = []

        for input_name in node.input:
            if "past_key_values" in input_name:
                past_inputs.append(input_name)
            else:
                other_node_inputs.append(input_name)
        
        node_outputs = list(node.output)

        if len(past_inputs) > 0:
            concat_nodes.append(node)
        
        assert len(node_outputs) == 1

        present_node_outputs = []

        for n in graph.node:
            if any(ni == node_outputs[0] and "present" in ni for ni in n.input):
                present_node_outputs.append(n.name)

        if len(present_node_outputs) > 0:   
            add_conditional_node(graph, node, past_inputs, other_node_inputs, node_outputs, present_node_outputs, i)

    onnx.save(model, "test_name.onnx", save_as_external_data=True)
    del model
    onnx.checker.check_model("test_name.onnx", full_check=True)

def add_conditional_node(graph, node, past_inputs, other_node_inputs, node_outputs, present_node_outputs, num_node):
    """
    Conditional flow node "If".
    Problem with such a function is that different models implement KV caching differently,
    so this would need to be implemented differently for each model if needed.
    """

    # Training mode with no past
    then_branch = helper.make_graph(
        nodes=[
            # Identity node for using 'present' input as is
            helper.make_node("Identity", inputs=other_node_inputs, outputs=node_outputs)
        ],
        name=f"training_branch_{node_outputs[0]}",
        inputs=[],
        outputs=[helper.make_tensor_value_info(name, TensorProto.FLOAT, ("batch_size", 4, "past_sequence_length + 1", 64)) for name in node_outputs]
    )

    other_node_inputs.extend(past_inputs)

    # Inference with past
    else_branch = helper.make_graph(
        nodes=[
            # Use the concat node
            helper.make_node("Concat", inputs=other_node_inputs, outputs=node_outputs, axis=-2)
        ],
        name=f"inference_branch_{node_outputs[0]}",
        inputs=[], #+ past_inputs,
        outputs=[helper.make_tensor_value_info(name, TensorProto.FLOAT, ("batch_size", 4, "past_sequence_length + 1", 64)) for name in node_outputs]
    )

    if_node = helper.make_node(
                "If",
                inputs=["training_mode"],
                outputs=node_outputs,
                then_branch=then_branch,
                else_branch=else_branch
            )

    graph.node.remove(node)
    graph.node.insert(num_node, if_node)

    return graph

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

class OnnxPackagedWrapper(torch.nn.Module):
    """
    Model with conditional flow, with possibility of handling both training and inference.
    It is not possible due to limitations of torch scripting.

    TorchScript not a viable solution as it is not supported by transformers entirely.
    scripted_model = torch.jit.script(my_model)
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model
        self.config = model.config
        
    def forward(self, input_ids, attention_mask, position_ids, labels, past_key_values):
        # Do inference if all the labels are zero
        # This requires script based exporter with conditional flow.
        if torch.all(labels == 0):
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=True)
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

def preprocess_model(model : torch.nn.Module, epsilon_high=1e-8, epsilon_low=1e-10):
    """
    Add a really small epsilon to the model parameters if they are all zeroes.
    This is to prevent the ONNX from not saving the extra weights, as they need to be included as initializers. 
    """

    for name, param in model.named_parameters():
        if torch.all(param.data == 0):
            random_values = torch.rand_like(param.data)
            random_values = (epsilon_high - epsilon_low) * random_values + epsilon_low
            param.data += random_values
    
    return model

def optimum_hf_export(model_id, model_output="onnx_models", training_mode = False, train_method = "lora", lora_target=["q_proj", "k_proj"], lora_rank=4, quantize=True, weight_type=QuantType.QInt16, opset=18):
    """
    Exports the model from Huggingface to an ONNX model representation.
    - `model_id` - model id of Huggingface model
    - `model_output` - path to model output directory
    - `training_mode`- create model for training or inference
    - `lora_target` - which layers to apply LoRA to
    - `quantize` - add dynamic quantization to layers which do not need gradient updates
    """

    model = AutoModelForCausalLM.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    if config.architectures[0] == "LlamaForCausalLM":
        ocl = LlamaOnnxConfig(config, task="text-generation", use_past=not training_mode, use_past_in_inputs=not training_mode)
    elif config.architectures[0] == "GemmaForCausalLM":
        ocl = GemmaOnnxConfig(config, task="text-generation", use_past=not training_mode, use_past_in_inputs=not training_mode)

    if training_mode:
        ocl = OnnxConfigWithLoss(ocl)

    onnx_path = Path(f"{model_output}/model.onnx")

    lora_config = LoraConfig(
            r=lora_rank,
            target_modules=lora_target,
            task_type="CAUSAL_LM",
        )

    # TODO: Should make merged LoRA adapters for inference model

    if train_method == "lora":
        # Apply LoRA to the model
        lora_model = PeftModel(model, lora_config)
    elif train_method == "lora-xs":
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
    elif train_method == None:
        lora_model = model

    if training_mode:
        my_model = OnnxTrainerWrapper(lora_model)
        my_model.train()
    else:
        my_model = OnnxInferenceWrapper(lora_model)
        my_model.eval()

    # Get layers with gradients in the LoRA model
    grad_layers, no_grad_layers = get_layers_with_grad(my_model)

    # Preprocessing methods
    my_model = preprocess_model(my_model)

    export(my_model, ocl, onnx_path, opset, do_constant_folding=False)

    # Save gradient layer names
    if training_mode:
        with open(f"{model_output}/training_config.json", "w+", encoding="utf-8") as f:
            json.dump({
                "requires_grad": grad_layers,
                "frozen_params": no_grad_layers
            }, f, ensure_ascii=False)
    
    # Apply dynamic quantization to non-trainable layers
    if quantize:
        onnx_dynamic_quantization(onnx_path.absolute().as_posix(), f"{model_output}/quant_model.onnx", exclude_weights=lora_target, weight_type=weight_type)

def onnx_dynamic_quantization(onnx_model_path, onnx_model_quant_output, weight_type=QuantType.QInt16, exclude_weights=["q_proj", "k_proj"], exclude_extra_layers=["embed_tokens"]):

    onnx_model = onnx.load(onnx_model_path)

    nodes_to_not_quantize = []

    # Exclude trainable nodes
    for param in onnx_model.graph.node:
        if any((allowed_layer in param.name and param.name.endswith("Transpose")) for allowed_layer in exclude_weights):
            nodes_to_not_quantize.append(param.name)
        if any(allowed_layer in param.name for allowed_layer in exclude_extra_layers):
            nodes_to_not_quantize.append(param.name)

    # Does not work
    #quant_pre_process(onnx_model_path, f"pre_{onnx_model_quant_output}", save_as_external_data=True, all_tensors_to_one_file=True, external_data_location=f"pre_{onnx_model_quant_output}")

    del onnx_model
    gc.collect()

    quantize_dynamic(
        extra_options={
            "ForceQuantizeNoInputCheck": True
        },
        nodes_to_exclude=nodes_to_not_quantize,
        model_input=onnx_model_path,
        model_output=onnx_model_quant_output,
        use_external_data_format=True,
        weight_type=weight_type,
        #reduce_range=True
    )

if __name__ == "__main__":

    optimum_hf_export(model_id=model_id, training_mode=False, train_method=None, lora_target=["q_proj", "k_proj", "v_proj", "o_proj"], lora_rank=4, quantize=True, weight_type=QuantType.QUInt8)
    #inspect_weights("artifacts/inference_model.onnx")
    #compare_weights("artifacts/inference_model.onnx", "opt_tinyllama_inference_int16/quant_model.onnx")
    #trim_initializers("onnx_tinyllama_exported_inf/model.onnx")
"""
Script that creates the training and inference model artifacts which can be deployed to the device.
The models are utilized by the on-device application.
"""

import argparse
import textwrap
from typing import Dict, List
import subprocess
import os, json, gc, time, yaml

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from transformers import AutoTokenizer, AutoConfig

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.training import onnxblock, artifacts
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnx.external_data_helper import convert_model_to_external_data, write_external_data_tensors, set_external_data

import onnxruntime as rt
from onnxruntime import InferenceSession, SessionOptions
from inference.generator import generate_tokens_onnx
from tools.utils import move_files_excluding, delete_directory, load_and_save_dataset
from tools.parser_config import ARTIFACT_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG, TASK_NAME_TO_DATASET
from tools.tokenizer_export import export_tokenizer_config

from artifact.merger import create_mars_merger_model, create_lora_merger_model

def gen_artifacts(train_dir,
                  artifact_dir="artifacts",
                  model_name="quant_model.onnx",
                  train_cfg_file="training_config.json",
                  training_config={}):
    """
    Generates the training artifacts from the provided model and directory.
    Needs the training configuration provided along with the model in the same directory. 
    """
    onnx_model_path = os.path.join(train_dir, model_name)
    train_cfg_path = os.path.join(train_dir, train_cfg_file)

    onnx.checker.check_model(onnx_model_path, full_check=True)
    onnx_model = onnx.load(onnx_model_path)

    params = {}
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    requires_grad = []
    frozen_params = []

    for param in onnx_model.graph.initializer:
        if any(rqp in param.name for rqp in params["requires_grad"]):
            requires_grad.append(param.name)
        else:
            frozen_params.append(param.name)
    
    del onnx_model
    gc.collect()

    # Generate the training artifacts
    artifacts.generate_artifacts(onnx_model_path,
                                requires_grad = requires_grad,
                                frozen_params = frozen_params,
                                # We don't need to provide a loss function, as the loss is already
                                # computed from the PyTorch Transformer model
                                # In the case of inference model, we don't need it
                                #loss = CausalLMCE(),
                                optimizer = artifacts.OptimType.AdamW,
                                artifact_directory = artifact_dir
                                )
    
    # Export training configs
    with open(f'{artifact_dir}/{train_cfg_file}', "w", encoding="utf-8") as f:
        json.dump({
            "requires_grad": requires_grad,
            "peft_mapping": params["peft_mapping"],
            **training_config
            #"frozen_params": frozen_params
        }, f, ensure_ascii=False)

def onnx_checktrain(model_dir,
                    model_id,
                    export_inference=False,
                    test_inference=False,
                    test_evaluate=False,
                    transfer_weights=False,
                    inference_model_path="inference_model.onnx",
                    max_sequence_length=100):
    """
    Checks the model if the outputs are training correctly as well as evaluation.
    Exports model for inference if needed or transfers the weights to an already existing inference model.

   - `test_inference` - runs the model through an example prompt
   - `test_evaluate` - tests the model evaluation
   - `transfer_weights` - instead of exporting for inference, we only copy the subset of updated weights to an already created inference model after training
   - `inference_model_path` - model path of an already existing inference model or one to create
   - `max_sequence_length` - length of text generation sequence
    """
    
    state = CheckpointState.load_checkpoint(f"{model_dir}/checkpoint")

    sess_options = SessionOptions()    
    sess_options.enable_profiling = False
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

    model = Module(f"{model_dir}/training_model.onnx", state, f"{model_dir}/eval_model.onnx", session_options=sess_options)
    optimizer = Optimizer(f"{model_dir}/optimizer_model.onnx", model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

    # Create dummy input
    tokenizer.pad_token_id = 0
    inputs = tokenizer(["This is a test, hello from world.", "This is a test, hello to world."], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].numpy()
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    labels = inputs["input_ids"].clone().numpy()

    labels = np.copy(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Optionally, set the last token to -100 to ignore it in the loss

    inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
        "position_ids": position_ids,
        "labels": labels
    }

    start_train_time = time.time()
    model.train()
    forward = model(*inputs.values())
    optimizer.step()
    model.lazy_reset_grad()
    end_train_time = time.time()

    print(f"[INFO] Training loss result: {forward[0]}")
    print(f"[INFO] Training time: {end_train_time - start_train_time} s")

    # TODO: Doesn't work with inputs?
    if test_evaluate:
        model.eval()
        forward = model(input_ids, labels)
        print("Evaluation results:")
        print(forward[0])

    exclude_nodes = ["loss"]

    if transfer_weights:
        inference_model_path = os.path.join(model_dir, inference_model_path)
        onnx_transfer_trained_weights(state, inference_model_path)
        del model
        del state
        del optimizer
        gc.collect()
    elif export_inference:
        # Model inference: we want to get only logits and hidden states for decoding
        model.export_model_for_inferencing(f"{model_dir}/{inference_model_path}", [ out_name for out_name in model.output_names() if out_name not in exclude_nodes])

        del model
        del state
        del optimizer
        gc.collect()
        # Load and test inference if needed
        if test_inference:
            onnx_infer(model_id, f"{model_dir}/{inference_model_path}", with_past=False, max_length=max_sequence_length)

def force_dequantize_external_and_save(model, output_path, external_data_filename=None):
    """
    Force DequantizeLinear x_scale and x_zero_point tensors to be external and save the model.
    
    Args:
        model: Loaded ONNX model (onnx.ModelProto)
        output_path: Path where to save the modified model
        external_data_filename: Name of external data file (optional, defaults to model_name.onnx.data)
    
    Returns:
        int: Number of tensors that were forced to external
    """
    if external_data_filename is None:
        model_name = os.path.splitext(os.path.basename(output_path))[0]
        external_data_filename = f'{model_name}.onnx.data'
    
    forced_count = 0
    
    # Force DequantizeLinear tensors to external manually BEFORE converting everything else
    for initializer in model.graph.initializer:
        # Check if initializer name ends with x_scale or x_zero_point
        is_dequant_tensor = (initializer.name.endswith('weight_scale') or 
                           initializer.name.endswith('weight_zero_point'))
        
        if is_dequant_tensor and initializer.data_location != onnx.TensorProto.EXTERNAL:
            #print(f"Forcing DequantizeLinear tensor to external: {initializer.name}")
            
            # Convert tensor data to raw_data format first
            # This ensures the tensor has raw_data field that set_external_data expects
            if not initializer.HasField("raw_data"):
                # Convert using numpy helper to preserve exact data type and format
                tensor_array = onnx.numpy_helper.to_array(initializer)
                
                # Clear all existing data fields first
                initializer.ClearField("float_data")
                initializer.ClearField("int32_data") 
                initializer.ClearField("int64_data")
                initializer.ClearField("string_data")
                initializer.ClearField("uint64_data")
                initializer.ClearField("double_data")
                initializer.ClearField("raw_data")
                
                # Set raw_data with the binary representation
                initializer.raw_data = tensor_array.tobytes()
            
            # Now use the proper ONNX function to set external data
            onnx.external_data_helper.set_external_data(
                tensor=initializer,
                location=external_data_filename
            )
            forced_count += 1
            
            # Now use the proper ONNX function to set external data
            set_external_data(
                tensor=initializer,
                location=external_data_filename
            )
            
            forced_count += 1
    
    # Convert all OTHER tensors to external data (this won't affect already external ones)
    convert_model_to_external_data(
        model, 
        location=external_data_filename, 
        size_threshold=0,
        all_tensors_to_one_file=True
    )
    
    # Write external data to file
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = "."
    
    return write_external_data_tensors(model, output_dir)

def onnx_export_dummy_model(model_output="tokenizer.onnx"):
    """
    Creates a fake dummy model for the tokenization process with GenAI.
    """

    # Define the input and output tensor types
    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.FLOAT, [None, None])
    logits = helper.make_tensor_value_info('logits', TensorProto.FLOAT, [None, None])


    node = helper.make_node(
        "Identity",
        inputs=["input_ids"],
        outputs=["logits"]
    )

    graph = helper.make_graph(
        nodes=[node], 
        name="identity_graph",
        inputs=[input_ids],  # No inputs
        outputs=[logits]  # No outputs
    )

    # Create an empty model
    model = helper.make_model(
        graph,
        producer_name='onnx-empty-model',
        opset_imports=[helper.make_opsetid('', 14)]  # Adjust the opset version as needed
    )

    onnx.checker.check_model(model)

    onnx.save_model(model, model_output, save_as_external_data=False)

def onnx_infer(model_id, model_path="inf_model_onnx_gemma_nonq.onnx", with_past=False, max_length=100):
    """
    Test inference with the provided ONNX inference model. Model needs to have inputs:
    - input_ids
    - attention_mask
    - position_ids
    """

    session = InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    config = AutoConfig.from_pretrained(model_id)

    prompt = "Hello, this is a message for the world. How is your day?"

    print(generate_tokens_onnx(prompt, tokenizer, session, config, with_past=with_past, max_length=max_length))

def onnx_segment_weights(model_path, output_path):
    """
    Save the model with external data.
    """
    m = onnx.load(model_path)
    onnx.save(m, output_path, save_as_external_data=True, all_tensors_to_one_file=False)

def onnx_transfer_trained_weights(state : CheckpointState, inference_model):
    """
    Transfer the updated weights from the checkpoint traning session to the inference model and save it.
    """

    updated_weights = {}

    # Extract all updated parameters data
    for param_name, parameter in state.parameters:
        if parameter.requires_grad:
            print(param_name)
            updated_weights[param_name] = parameter.data

    # In on-device training scenario we would also save the checkpoint with gradients
    # and delete the training session from memory
    state.save_checkpoint(state)

    # Load the inference model
    onnx_inference_model = onnx.load(inference_model)

    # Overwrite the parameters that require gradient
    for i, initializer in enumerate(onnx_inference_model.graph.initializer):
        if initializer.name in updated_weights:

            W = numpy_helper.to_array(initializer)
            if not np.array_equal(W, updated_weights[initializer.name]):
                print(f"Overwriting {initializer.name}, weights changed...")
                new_tensor = numpy_helper.from_array(updated_weights[initializer.name], initializer.name)
                initializer.CopyFrom(new_tensor)
                new_numpy = numpy_helper.to_array(onnx_inference_model.graph.initializer[i])
                if np.array_equal(new_numpy, updated_weights[initializer.name]):
                    print("Copied successfully")
            else:
                print(f"Weights were not changed, but the training session was performed on these weights?\nParameter: {initializer.name}")

    onnx.save(onnx_inference_model)
    del onnx_inference_model
    gc.collect()

def gen_genai(model_id, 
              model_path,
              training_config,
              new_model_name,
              new_model_path,
              weight_input=True,
              include_metadata=True,
              large_model=False,
              test_generation=False,
              test_generation_config={},
              check_model=True,
              opset_version=18):
    """
    Creates a GenAI compatible ONNX graph or a custom inference graph.

    - `training_config` - training configuration json
    - `new_model_path` - path to new inference model
    - `large_model` - if model is larger than 2GB
    """

    model = onnx.load(model_path)
    model_trainable_weights = {}

    new_model_namepath = f'{new_model_path}/{new_model_name}.onnx'

    if weight_input:

        with open(f'{training_config}', 'r') as f:
            requires_grad_layers = json.load(f)["requires_grad"]
        
        # Extract initializers from the model
        initializers = {init.name: init for init in model.graph.initializer}

        # Create new input nodes for the specified initializers
        new_inputs = []
        for name in requires_grad_layers:
            if name in initializers:
                initializer = initializers[name]
                # Create a new input node
                new_input = helper.make_tensor_value_info(name, initializer.data_type, initializer.dims)
                new_inputs.append(new_input)

                # Store them for the testing generation input
                model_trainable_weights[name] = numpy_helper.to_array(initializer)

                # Remove initializer since it becomes the input
                model.graph.initializer.remove(initializer)
        
        # Create a new list of inputs (existing inputs + new inputs for specified initializers)
        new_graph_inputs = list(model.graph.input) + new_inputs
        
        # Remove the specified initializers from the model
        new_initializers = [init for init in model.graph.initializer if init.name not in requires_grad_layers]
        
        # Create a new graph with updated inputs and removed initializers
        new_graph = helper.make_graph(
            nodes=model.graph.node,
            name=model.graph.name,
            inputs=new_graph_inputs,
            outputs=model.graph.output,
            initializer=new_initializers
        )
        
        # Create a new model with the modified graph
        model = helper.make_model(new_graph, opset_imports=[helper.make_operatorsetid("", opset_version)])
    
    if include_metadata:
        print(f"[INFO] Adding metadata to the inference model...")
        config = AutoConfig.from_pretrained(model_id)

        num_kv_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
        head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        num_layers = config.num_hidden_layers

        # Add custom metadata
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="model_id", value=str(model_id))
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="max_context_length", value=str(config.max_position_embeddings))
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="head_dim", value=str(head_size))
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="num_kv_heads", value=str(num_kv_heads))
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="num_layers", value=str(num_layers))
        )

    # We set size threshold to 0 to force all tensors to be saved as externally and later to be replaced easily in inference session
    print("[INFO] Forcing external initializers...")
    model = force_dequantize_external_and_save(model, new_model_namepath)
    onnx.save(model, new_model_namepath, save_as_external_data=True, location=f'{new_model_name}.onnx.data', size_threshold=0)

    if large_model:
        # Wait so it finishes writing to disk
        time.sleep(20)
        print("[INFO] Writing large model to disk...")
        if check_model:
            onnx.checker.check_model(new_model_namepath, full_check=True)
    elif check_model:
        onnx.checker.check_model(model, full_check=True)

    print("[INFO] Saved GenAI inference model.")

    if test_generation:
        session = InferenceSession(new_model_namepath, providers=['CPUExecutionProvider'])
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
        config = AutoConfig.from_pretrained(model_id)
        generate_tokens_onnx(tokenizer, session, config, model_trainable_weights, with_past=True, with_weight_input=weight_input, **test_generation_config)

def get_layers_with_grad(model):
    """
    Function to get all layers that require gradients
    """
    layers_with_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layers_with_grad.append(name)
    return layers_with_grad

class CausalLMCE(onnxblock.Block):
    def __init__(self):
        super().__init__()
        # Assumes classes is the last dimension
        # e.g., predictions: (num_examples, num_classes) -> labels: (num_examples,)
        # or predictions: (batch_size, seq_len, vocab) -> labels: (batch_size, seq_len)
        self._loss1 = onnxblock.loss.CrossEntropyLoss()

    def build(self, logits, *args):
        return self._loss1(logits)

def convert_pipeline(model_id,
                    peft_method,
                    train_model_name,
                    train_dir,
                    inference_model_name,
                    inference_dir,
                    build_dir,
                    gen_train_artifacts = True,
                    gen_inference_artifacts = True,
                    test_training = True,
                    test_eval = True,
                    test_generation = True,
                    inference_export_config = {},
                    test_generation_config = {},
                    inference_config = {},
                    train_config = {},
                    export_tokenizer = True, 
                    export_dataset = True,
                    export_inference_config = True,
                    export_merger = True,
                    delete_models=False,
                    config_file_path="config.yml",
                    **kwargs):
    """
    ONNX conversion for training and inference artifacts.
    Creates a build folder with train and inference subfolders each with models needed for tasks.
    """

    # TODO: Infer from the given model
    large_model = True

    try:
        # Create the base directory if it doesn't exist
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        
        train_path = os.path.join(build_dir, 'train')
        inference_path = os.path.join(build_dir, 'inference')
        
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(inference_path, exist_ok=True)
    
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
    
    if gen_train_artifacts:

        # Add peft method to training config
        train_config['peftMethod'] = peft_method

        # Generate training artifacts
        gen_artifacts(train_dir=train_dir, artifact_dir=f'{build_dir}/train', model_name=train_model_name, training_config=train_config)
        print("[INFO] Generated training artifacts.")

    if test_training:
        onnx_checktrain(model_dir=f'{build_dir}/train',
                        model_id=model_id,
                        test_evaluate=test_eval)
        print("[INFO] Training check completed.")
    
    # Export generative AI model
    if gen_inference_artifacts and inference_config["type"] == "genai":
        gen_genai(model_id=model_id,
                  model_path=f'{inference_dir}/{inference_model_name}',
                  training_config=f'{train_dir}/training_config.json',
                  new_model_name=inference_export_config["output_inference_model"],
                  new_model_path=f'{build_dir}/inference',
                  large_model=large_model,
                  #test_generation=test_generation,
                  weight_input=inference_export_config["weight_input"],
                  include_metadata=inference_export_config["include_metadata"],
                  opset_version=inference_export_config["opset"],
                  #test_generation_config=test_generation_config,
                  check_model=inference_config["type"] == "native")
        print("[INFO] Generated the artifact inference model graph.")

        # Move the rest of the files
        move_files_excluding(inference_dir, f'{build_dir}/inference', exclude_files=[inference_model_name])
        print(f"[INFO] Moved the rest of generation configuration files to: {build_dir}/inference")

    # Export native inference model
    # NOTE: If the ONNX Runtime versions do not match, you need to use inference.builder to build inference model
    elif gen_inference_artifacts and inference_config["type"] == "native":

        # Export native inference model (only from config.yml configurations)
        gen_args = [
            "python", "-m", "inference.builder",
            "--config", config_file_path
        ]
        subprocess.run(gen_args, check=True)

    # Export generation configuration
    if export_inference_config:
        with open(f'{build_dir}/inference/generation_config.json', "w", encoding="utf-8") as f:
            json.dump(inference_config, f, ensure_ascii=False)

    # Export tokenizer if needed
    if export_tokenizer:
        export_tokenizer_config(model_id, build_dir, os.environ['HF_TOKEN'])

    if export_dataset:
        if "taskName" not in train_config or "trainFile" not in train_config:
            print(f"[WARNING] taskName or trainFile not defined in train_config!")
        elif train_config["taskName"] not in TASK_NAME_TO_DATASET:
            print(f"[WARNING] taskName unknown!")
        else:
            load_and_save_dataset(TASK_NAME_TO_DATASET[train_config["taskName"]], train_path, train_config["trainFile"], split='train', max_dataset_length=train_config['maxDatasetLength'])

    if export_merger:
        if peft_method == "lora":
            create_lora_merger_model(f'{build_dir}/train/lora_merger_model.onnx', quantized=True)
            create_lora_merger_model(f'{build_dir}/train/lora_merger_model.onnx', quantized=False)
        elif peft_method == "mars":

            # We need both quantized and non-quantized merger models
            if 'optimization_level' in kwargs["train_builder_config"][peft_method] and kwargs["train_builder_config"][peft_method]['optimization_level'] <= 1:
                create_mars_merger_model(f'{build_dir}/train/mars_merger_model.onnx', quantized=False)
            create_mars_merger_model(f'{build_dir}/train/mars_qmerger_model.onnx', quantized=True)

            # We also need basic LoRA merger models for this PEFT method
            create_lora_merger_model(f'{build_dir}/train/lora_qmerger_model.onnx', quantized=True)
            create_lora_merger_model(f'{build_dir}/train/lora_merger_model.onnx', quantized=False)

    # Clean the generated models if needed
    if delete_models:
        delete_directory(inference_dir)
        delete_directory(train_dir)
        print(f"[INFO] Deleted previously generated training and inference models.")

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
    return options_dict

def load_config_from_file(config_file: str):
    """Load configurations from a YAML file into a dictionary."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Converting the given ONNX models into a ONNX artifacts for on-device training and inference.", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model_id",
        type=str,
        help="Identifier for the model to be converted."
    )
    parser.add_argument(
        "--build_path",
        type=str,
        help="Path to convert the artifact models."
    )
    parser.add_argument(
        "--inference_model",
        type=str,
        help="Name of the inference model."
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        help="Path to the inference model directory."
    )
    parser.add_argument(
        "--training_model",
        type=str,
        help="Name of the training model."
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        help="Path to the training model directory."
    )
    parser.add_argument(
        "--gen_train_artifacts",
        type=bool,
        default=True,
        help="Whether to generate training artifacts. Default is True."
    )
    parser.add_argument(
        "--gen_inference_artifacts",
        type=bool,
        default=True,
        help="Whether to generate inference artifacts. Default is True."
    )
    parser.add_argument(
        "--test_training",
        type=bool,
        default=True,
        help="Whether to test training capabilities. Default is True."
    )
    parser.add_argument(
        "--test_eval",
        type=bool,
        default=True,
        help="Whether to test evaluation capabilities. Default is True."
    )
    #parser.add_argument(
    #    "--test_generation",
    #    type=bool,
    #    default=True,
    #    help="Whether to perform inference / generation test on the inference exported model."
    #)
    parser.add_argument(
        "--delete_models",
        type=bool,
        default=True,
        help="Deletes the previously generated models."
    )
    parser.add_argument(
        "--export_tokenizer",
        type=bool,
        default=True,
        help="Exports tokenizer files and config in seperate dir."
    )
    parser.add_argument(
        "--export_inference_config",
        type=bool,
        default=True,
        help="Exports inference configuration config in build dir."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments."
    )
    parser.add_argument(
        "--export_dataset",
        type=bool,
        default=True,
        help="Exports dataset file in train dir."
    )
    parser.add_argument(
        "--export_merger",
        type=bool,
        default=True,
        help="Exports the merger model for adapters."
    )
    parser.add_argument(
        "--inference_export_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ...
            """
            )
    )
    parser.add_argument(
        "--inference_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ... TODO add description
            """
            )
    )
    parser.add_argument(
        "--train_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ... TODO add description
            """
            )
    )
    #parser.add_argument(
    #    "--test_generation_config",
    #    type=str,
    #    nargs="*",
    #    metavar="KEY=VALUE",
    #    default=[],
    #    help=textwrap.dedent("""\
    #     Key value pairs for various options. Currently supports:
    #        prompt = Hello... : Prompt for test generation. If using chatpot template setting, please provide the chat template format as well.
    #        decode_between = true : Whether to decode the text while it's generating.
    #        max_length = 100 : Max length of test sequence to generate.
    #        sampling = top_k : Sampling method. Should support topk and topp
    #        temperature = 0.7 : Temperature for sampling
    #        top_k = 10 : Top K for sampling
    #        top_p = 0.3 : Top P for sampling
    #        """
    #        )
    #)
    args = parser.parse_args()

    user_inference_config = {}
    default_user_inference_config = {
        "type": "genai", # "normal", "genai"
        "weight_input" : False, # Whether to include trainable weights as model input
        "test_inference": True, # Whether to perform inference / generation test on the inference exported model
        "include_metadata": True, # Whether to include the model metadata
        "output_inference_name": "genai_inference", # The new model name
        "opset_version": 20,
        "gen_config_file": "genai_config.json"
    }

    #user_test_generation_config = {}
    #default_test_generation_config = {
    #   "prompt": "Hello, this is a message for the world. How is your day?", # Prompt for test generation
    #        "decode_between": True, # Whether to decode the text while it's generating
    #        "max_length" : 100, # Max length of test sequence to generate
    #       "sampling": "topk", # Sampling method
    #        "temperature": 0.7, # Temperature for sampling
    #        "top_k": 10, # Top K for sampling
    #}

    extra_args = {}

    config_dict = None

    if args.config:
        config_dict = load_config_from_file(args.config)

        # Specific
        setattr(args, "model_id", config_dict[TRAIN_CONFIG]["model_id"])
        setattr(args, "peft_method", config_dict[TRAIN_CONFIG]["train_method"])
        
        # Override any command-line argument with values from the config file
        for key, value in config_dict[ARTIFACT_CONFIG].items():
            
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)

        setattr(args, "training_dir",config_dict[TRAIN_CONFIG]["output"])
        setattr(args, "inference_dir", config_dict[INFERENCE_CONFIG]["output"])

        extra_args["train_builder_config"] = config_dict[TRAIN_CONFIG]
    else:
        user_inference_config = parse_extra_options(args.inference_config)
        args.inference_export_config = {**default_user_inference_config, **user_inference_config}
        #user_test_generation_config = parse_extra_options(args.test_generation_config)
        #args.test_generation_coinfig = {**default_test_generation_config, **user_test_generation_config}

    return args, extra_args

if __name__ == "__main__":

    args, extra_args = parse_arguments()

    print(f"{ARTIFACT_CONFIG} arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    convert_pipeline(
        model_id=args.model_id,
        peft_method=args.peft_method,
        train_model_name=args.training_model,
        train_dir=args.training_dir,
        inference_model_name=args.inference_model,
        inference_dir=args.inference_dir,
        build_dir=args.build_path,
        gen_inference_artifacts=args.gen_inference_artifacts,
        gen_train_artifacts=args.gen_train_artifacts,
        test_training=args.test_training,
        test_eval=args.test_eval,
        # We avoid testing generation in this script due to package conflicts
        #test_generation=args.test_generation,
        inference_export_config=args.inference_export_config,
        #test_generation_config=args.test_generation_config
        export_tokenizer=args.export_tokenizer,
        inference_config=args.inference_config,
        train_config=args.train_config,
        export_dataset=args.export_dataset,
        export_inference_config=args.export_inference_config,
        export_merger=args.export_merger,
        delete_models=args.delete_models,
        config_file_path=args.config,
        **extra_args
    )
import torch
import os, json, gc

import numpy as np
import netron
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.training import onnxblock, artifacts
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from onnxruntime.quantization.preprocess import quant_pre_process
from peft import PeftModel, LoraConfig
from onnxruntime import InferenceSession

# TODO: Add Huggingface token
#os.environ["HF_TOKEN"] = "TODO"

model_id = "TinyLlama/TinyLlama_v1.1"#"google/gemma-2b-it"#
model_name = model_id.split("/")[1]

def convert_onnx(perform_pass=True, quantize=True):
    """
    Previous experiments for converting Huggingface models to onnx artifacts for on device training.
    Use other functions for generation of artifacts.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    pre_model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

    # Custom ONNX CausalLM model
    model = OnnxCausalLM(model=pre_model)

    lora_config = LoraConfig(
        r=4,
        target_modules=["q_proj", "k_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the model
    lora_model = PeftModel(model, lora_config)

    # Get layers with gradients in the LoRA model
    layers_with_grad = get_layers_with_grad(lora_model)

    # Create dummy input for tracing
    inputs = tokenizer("This is a test, hello from world.", return_tensors="pt")
    labels = inputs["input_ids"].clone()

    # Define the export path
    onnx_model_path = f"onnx_models/lora_{model_name}.onnx"
    onnx_model_dir = "onnx_models/"
    onnx_model_quant_output = f"lora_{model_name}_q.onnx"

    # Export the model to ONNX
    torch.onnx.export(
        lora_model,                                       # Model to be exported
        (inputs["input_ids"], inputs["attention_mask"], labels),
        onnx_model_path,                                  # Path where the model will be saved
        input_names=['input_ids', 'attention_mask', 'labels'],      # Input names
        output_names=["loss", "logits"],                          # Output names
        dynamic_axes={                                    # Dynamic axes for variable length inputs
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            "labels": {0: "batch_size", 1: "sequence_length"},
            "logits ": {0: "batch_size", 1: "sequence_length"}
            #'output': {0: 'batch_size', 1: 'sequence_length'},
        },
        export_params=True,
        # Optimize model by folding constant nodes (not recommended for ONNX runtime)
        do_constant_folding=False,
        # Training mode for on device learning
        training=torch.onnx.TrainingMode.TRAINING
    )


    # Perform a forward pass
    if perform_pass:
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
            print(outputs)
    
    onnx.checker.check_model(onnx_model_path, full_check=True)
    onnx_model = onnx.load(onnx_model_path)

    if quantize:
        no_quantized_layers = ["q_proj", "k_proj"]
        nodes_to_not_quantize = []

        # Exclude trainable nodes
        for param in onnx_model.graph.node:
            if any((allowed_layer in param.name and param.name.endswith("Transpose")) for allowed_layer in no_quantized_layers):
                nodes_to_not_quantize.append(param.name)

        quantize_dynamic(
            extra_options={ 
                "ForceQuantizeNoInputCheck": True
            },
            model_input=onnx_model_path,
            model_output=onnx_model_quant_output,
            nodes_to_exclude=nodes_to_not_quantize,
            use_external_data_format=True,
            weight_type=QuantType.QInt8,
            reduce_range=True
        )

        onnx.checker.check_model(onnx_model_quant_output, full_check=True)
        onnx_model = onnx.load(onnx_model_quant_output)  

        requires_grad = []
        frozen_params = []
        for param in onnx_model.graph.initializer:
            if any(grad_param in param.name for grad_param in no_quantized_layers):
                requires_grad.append(param.name)
            else:
                frozen_params.append(param.name)
    else:
        requires_grad = layers_with_grad
        frozen_params = [
            param.name
            for param in onnx_model.graph.initializer
            if param.name not in requires_grad
        ]

    del onnx_model
    print("Requires grad layers:")
    print(layers_with_grad)
    print("Frozen parameter layers:")
    print(frozen_params)

    # Generate the training artifacts
    artifacts.generate_artifacts(onnx_model_quant_output if quantize else onnx_model_path,
                                requires_grad = requires_grad,
                                frozen_params = frozen_params,
                                # We don't need to provide a loss function, as the loss is already
                                # computed from the PyTorch Transformer model
                                #loss = CausalLMCE(),
                                optimizer = artifacts.OptimType.AdamW,
                                artifact_directory = onnx_model_dir)

def gen_artifacts(train_dir,
                  artifact_dir="artifacts",
                  model_name="quant_model.onnx",
                  train_cfg="training_config.json"):
    """
    Generates the training artifacts from the provided model and directory.
    Needs the training configuration provided along with the model in the same directory. 
    """
    onnx_model_path = os.path.join(train_dir, model_name)
    train_cfg_path = os.path.join(train_dir, train_cfg)

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
                                artifact_directory = artifact_dir)

def onnx_checktrain(model_dir="onnx_tinyllama_int8",
                    export_inference=True,
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
   - `inference_model_path` - model path of an already existing inference model
   - `max_sequence_length` - length of text generation sequence
    """

    state = CheckpointState.load_checkpoint(f"{model_dir}/checkpoint")

    model = Module(f"{model_dir}/training_model.onnx", state, f"{model_dir}/eval_model.onnx")
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

    model.train()
    forward = model(*inputs.values())

    print("Training result:")
    print(forward[0])
    
    optimizer.step()
    model.lazy_reset_grad()

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
        model.export_model_for_inferencing(f"{model_dir}/inference_model.onnx", [ out_name for out_name in model.output_names() if out_name not in exclude_nodes])

        del model
        del state
        del optimizer
        gc.collect()

    # Load and test inference if needed
    if test_inference:
        onnx_infer(f"{model_dir}/inference.onnx", with_past=False, max_length=max_sequence_length)
    
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

def onnx_infer(model_path="inf_model_onnx_gemma_nonq.onnx", with_past=False, max_length=100):
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

def view_model(onnx_model):
    """
    View the model with Netron.
    """
    netron.start(onnx_model)

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

def get_layers_with_grad(model):
    """
    Function to get all layers that require gradients
    """
    layers_with_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layers_with_grad.append(name)
    return layers_with_grad

def generate_tokens_onnx(prompt, tokenizer, model, config, with_past=False, output_name="logits", max_length=100, sampling="greedy", temperature=1.0, top_k=50):
    """
    Generates the tokens from the ONNX Inference sessions.
    Uses either "topk" or "greedy" sampling methods.
    """
    
    input_ids = tokenizer(prompt, return_attention_mask=True, return_tensors="np")

    num_kv_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
    head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

    # Initialize the generated sequence with the prompt
    generated_ids = input_ids["input_ids"]
    attention_mask = input_ids["attention_mask"]
    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids = np.clip(position_ids, 0, None)

    # If KV caching enabled:
    if with_past:
        past_key_values = {}
        for i in range(config.num_hidden_layers):
            past_key_values[f"past_key_values.{i}.key"] = np.zeros(shape=(1, num_kv_heads, 0, head_size), dtype=np.float32)
            past_key_values[f"past_key_values.{i}.value"] = np.zeros(shape=(1, num_kv_heads, 0, head_size), dtype=np.float32)

        present_keys = [pkv.replace("past_key_values", "present") for pkv in past_key_values.keys()]

    for _ in range(max_length):

        model_inputs = {
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        if with_past:
            model_inputs.update(past_key_values)

        # Forward pass to get logits
        output = model.run([output_name] + (present_keys if with_past else []), model_inputs)

        logits = output[0]

        # TODO: KV cache doesn't work (model always expects past_sequence_length to be 0)
        #if with_past:
        #new_shape = (1, 4, attention_mask.shape[-1]+1, 64)
        #present_kv = {}
        #for pkv, pkv_name in zip(output[1:], past_key_values.keys()):
        #    extended_array = np.zeros(new_shape, dtype=np.float32)
        #    extended_array[:, :, :attention_mask.shape[-1], :] = pkv
        #    present_kv[pkv_name] = extended_array
        #    print(pkv_name)
        #    print(extended_array.shape)
        #past_key_values = present_kv
        
        # Get the last token logits and apply temperature
        logits = logits[:, -1, :] / temperature

        if sampling=="topk":
            # Apply top-k filtering
            top_k_values, top_k_indices = np.partition(logits[0], -top_k)[-top_k:], np.argpartition(logits[0], -top_k)[-top_k:]
            filtered_logits = np.full_like(logits[0], -float('Inf'))
            filtered_logits[top_k_indices] = top_k_values

            # Sample from the filtered logits
            probabilities = np.exp(filtered_logits - np.max(filtered_logits))  # Stability improvement
            probabilities /= np.sum(probabilities)
            next_token_id = np.random.choice(len(probabilities), p=probabilities)
        elif sampling == "greedy":
            next_token_id = np.argmax(logits[0])

        # Append the sampled token to the sequence
        generated_ids = np.concatenate([generated_ids, [[next_token_id]]], axis=1)
        print(generated_ids)

        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(decoded)

        # Update the position_ids (increment by 1 for the new token)
        new_position_id = position_ids[0, -1] + 1
        position_ids = np.concatenate([position_ids, [[new_position_id]]], axis=1)
        
        # Update the attention_mask (add 1 for the new token)
        attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)

        
        # Stop if end-of-sequence token is generated
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode the generated tokens
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_tokens(prompt, tokenizer, model, max_length=50, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    # Initialize the generated sequence with the prompt
    generated_ids = input_ids

    for _ in range(max_length):
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits

        # Get the last token logits and apply temperature
        logits = logits[:, -1, :] / temperature
        print(logits)

        # Apply top-k filtering
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        filtered_logits = torch.full_like(logits, -float('Inf'))
        filtered_logits.scatter_(1, top_k_indices, top_k_values)

        # Sample from the filtered logits
        next_token_id = torch.multinomial(torch.nn.functional.softmax(filtered_logits, dim=-1).squeeze(), num_samples=1).item()

        # Append the sampled token to the sequence
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)

        # Stop if end-of-sequence token is generated
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode the generated tokens
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


class CausalLMCE(onnxblock.Block):
    def __init__(self):
        super().__init__()
        # Assumes classes is the last dimension
        # e.g., predictions: (num_examples, num_classes) -> labels: (num_examples,)
        # or predictions: (batch_size, seq_len, vocab) -> labels: (batch_size, seq_len)
        self._loss1 = onnxblock.loss.CrossEntropyLoss()

    def build(self, logits, *args):
        return self._loss1(logits)

class OnnxCausalLM(torch.nn.Module):
    def __init__(self, model):
        super(OnnxCausalLM, self).__init__()
        # Initialize the pre-trained model
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


if __name__ == "__main__":

    
    gen_artifacts(train_dir="opt_tinyllama_train_int16", model_name="quant_model.onnx", train_cfg="training_config.json")
    #onnx_segment_weights("opt_tinyllama_inference/model.onnx", "opt_tinyllama_train/model.onnx")
    # Get the supported opset version
    #onnx_infer(model_path="inference.onnx")
    #view_model("onnx_tinyllama_int8/training_model.onnx")
    onnx_checktrain("artifacts", test_inference=True, export_inference=False, transfer_weights=False)
    #convert_onnx()
    #convert_onnx_genai()
    #onnx_export_dummy_model("tokenizer.onnx")
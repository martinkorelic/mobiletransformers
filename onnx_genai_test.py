import json
import onnxruntime_genai as og
import numpy as np
import time
import onnx
from onnx import numpy_helper

# Build configuration only
# python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options config_only=true


def test_genai_model(path="./onnx_genai_config"):

    model = og.Model(path)
    params = og.GeneratorParams(model)
    tokenizer = og.Tokenizer(model)
    input_tokens = tokenizer.encode_batch(["Hello, this is a message for the world. How is your day?"]).astype(np.int64)
    #params.set_model_input("labels", label_tokens)
    #params.try_graph_capture_with_max_batch_size(1)

    params.input_ids = input_tokens
    
    start_time = time.time()

    output_tokens = model.generate(params)
    end_time = time.time()

    elapsed_time = end_time - start_time
    
    print(output_tokens)
    decoded_tokens = tokenizer.decode_batch(output_tokens)
    print(decoded_tokens)
    print(f"Elapsed time for generation: {elapsed_time:.4f}")
    #gen = og.Generator(model, params)

    #l = gen.get_output("logits")
    #gen.compute_logits()
    #t = gen.generate_next_token()
    #print(t)
    #gen.compute_logits()
    #t = gen.generate_next_token()
    #t = gen.get_sequence(0)
    #print(t)

    #output_tokens = model.generate(params)

def test_genai_model_with_inputs(path, artifact_dir, weight_path, load_inference_model=True):

    model = og.Model(path)
    params = og.GeneratorParams(model)
    tokenizer = og.Tokenizer(model)
    input_tokens = tokenizer.encode_batch(["Hello, this is a message for the world. How is your day?"]).astype(np.int64)
    
    # Here we load from the inference model initializers so we can test it in python
    # This is because I could not create an environment with onnxruntime-training and onnxruntime-genai to load from CheckpointState
    # In practice these updated initializers would be loaded from CheckpointState
    if load_inference_model:
        m = onnx.load(f'{weight_path}')

    # Load the list of names from the JSON file
    with open(f'{artifact_dir}/training_config.json', 'r') as f:
        requires_grad_layers = json.load(f)["requires_grad"]

    # Set the other inputs
    for initializer in m.graph.initializer:
        if initializer.name in requires_grad_layers:
            W = numpy_helper.to_array(initializer)
            W = np.copy(W)
            params.set_model_input(initializer.name, W)

    params.input_ids = input_tokens
    
    start_time = time.time()

    output_tokens = model.generate(params)
    end_time = time.time()

    elapsed_time = end_time - start_time
    
    print(output_tokens)
    decoded_tokens = tokenizer.decode_batch(output_tokens)
    print(decoded_tokens)
    print(f"Elapsed time for generation: {elapsed_time:.4f}")

if __name__ == "__main__":
    #test_genai_model("onnx_genai_test")
    test_genai_model_with_inputs("onnx_genai_test", "build/train", "/onnx_models/opt_tinyllama_inference_int16/quant_model.onnx")
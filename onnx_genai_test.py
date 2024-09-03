import onnxruntime_genai as og
import numpy as np
import time

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


if __name__ == "__main__":
    test_genai_model()
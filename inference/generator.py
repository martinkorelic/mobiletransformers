"""
Script for LLM generation loop using the provided inference model. 
"""

import time
import numpy as np

def generate_tokens_onnx(tokenizer,
                         model,
                         config,
                         model_train_weights=[],
                         with_past=False,
                         with_weight_input=False,
                         with_position_ids=True,
                         with_labels=False,
                         prompt="Hello, how is your day?",
                         output_name="logits",
                         max_length=100,
                         sampling={
                            "method": "greedy",
                            "temperature" : 1.0,
                            "topP" : 0.9,
                            "topK": 50
                         },
                         decode_between=True,
                         **kwargs):
    """
    Generates the tokens from the ONNX Inference sessions.
    Supports either greedy / top_k / top_p sampling methods.
    """

    input_ids = tokenizer(prompt, return_attention_mask=True, return_tensors="np")

    num_kv_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
    head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

    # Initialize the generated sequence with the prompt
    token_input_ids = input_ids["input_ids"]

    attention_mask = np.ones((1, token_input_ids.shape[1]), dtype=np.int64)
    position_ids = np.arange(token_input_ids.shape[1], dtype=np.int64).reshape(1, -1)
    generated_ids = np.array([[]], dtype=np.int64)

    # If KV caching enabled:
    if with_past:
        past_key_values = {}
        for i in range(config.num_hidden_layers):
            past_key_values[f"past_key_values.{i}.key"] = np.random.rand(*(1, num_kv_heads, 0, head_size)).astype(np.float32)
            past_key_values[f"past_key_values.{i}.value"] = np.random.rand(*(1, num_kv_heads, 0, head_size)).astype(np.float32)

        present_keys = [pkv.replace("past_key_values", "present") for pkv in past_key_values.keys()]

    if with_labels:
        labels = token_input_ids.copy()

    start_time = time.time()
    num_decode = 0
    for _ in range(max_length):

        model_inputs = {
            "input_ids": token_input_ids,
            "attention_mask": attention_mask,
        }

        if with_position_ids:
            model_inputs.update({"position_ids": position_ids})
        if with_past:
            model_inputs.update(past_key_values)
        if with_weight_input:
            model_inputs.update(model_train_weights)
        if with_labels:
            model_inputs.update({"labels": labels})

        # Forward pass to get logits
        output = model.run([output_name] + (present_keys if with_past else []), model_inputs)

        logits = output[0]

        if with_past:
            present_kv = {}
            for pkv, pkv_name in zip(output[1:], past_key_values.keys()):
                present_kv[pkv_name] = pkv
            past_key_values = present_kv
        
        # Get the last token logits and apply temperature
        logits = logits[:, -1, :] / sampling["temperature"]

        if sampling["method"]=="top_k":
            top_k = sampling["topK"]
            # Apply top-k filtering
            top_k_values, top_k_indices = np.partition(logits[0], -top_k)[-top_k:], np.argpartition(logits[0], -top_k)[-top_k:]
            filtered_logits = np.full_like(logits[0], -float('Inf'))
            filtered_logits[top_k_indices] = top_k_values

            # Sample from the filtered logits
            probabilities = np.exp(filtered_logits - np.max(filtered_logits))  # Stability improvement
            probabilities /= np.sum(probabilities)
            next_token_id = np.random.choice(len(probabilities), p=probabilities)
        elif sampling["method"] == "greedy":
            next_token_id = np.argmax(logits[0])
        elif sampling["method"] == "top_p":
            top_p = sampling["topP"]
            # Sort logits by probability
            sorted_logits = np.sort(logits[0])[::-1]
            sorted_indices = np.argsort(logits[0])[::-1]

            # Calculate cumulative probabilities
            cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)))
            cumulative_probs /= cumulative_probs[-1]  # Normalize to get cumulative probabilities

            # Determine where the cumulative probability surpasses p
            cutoff_index = np.searchsorted(cumulative_probs, top_p) + 1

            # Filter out tokens that fall outside the top-p cumulative probability
            filtered_logits = np.full_like(logits[0], -float('Inf'))
            filtered_logits[sorted_indices[:cutoff_index]] = sorted_logits[:cutoff_index]

            # Sample from the filtered logits
            probabilities = np.exp(filtered_logits - np.max(filtered_logits))  # Stability improvement
            probabilities /= np.sum(probabilities)
            next_token_id = np.random.choice(len(probabilities), p=probabilities)
        else:
            raise ValueError(f"Invalid type selected for sampling option '{sampling}'.")

        # Append the sampled token to the sequence
        generated_ids = np.concatenate([generated_ids, [[next_token_id]]], axis=1)

        if with_past:
            # Handle only for the next token
            token_input_ids = np.array([[next_token_id]], dtype=np.int64)
            position_ids = np.array([[token_input_ids.shape[-1] - 1]], dtype=np.int64)
        else:
            # Handle for the full sequence
            token_input_ids = np.concatenate([token_input_ids, np.array([[next_token_id]], dtype=np.int64)], axis=-1)
            position_ids = np.arange(token_input_ids.shape[-1], dtype=np.int64).reshape(1, -1)

        if decode_between:
            decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(decoded)
            print("-------------------------------------------------")

        num_decode += 1
        
        # Update the attention_mask (add 1 for the new token)
        new_ones = np.ones((attention_mask.shape[0], 1), dtype=np.int64)
        attention_mask = np.concatenate((attention_mask, new_ones), axis=-1)

        # Stop if end-of-sequence token is generated
        if next_token_id == tokenizer.eos_token_id:
            break
    
    end_time = time.time()
    print("\n")
    print(f"[INFO] Generation time: {(num_decode / (end_time - start_time)):.2f} token/s")

    # Decode the generated tokens
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
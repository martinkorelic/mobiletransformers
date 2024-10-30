import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, DynamicCache

def run_inference():
    # Load the TinyLlama model and tokenizer
    model_name = "TinyLlama/TinyLlama_v1.1"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()

    # Input text
    input_text = "Hello, this is a message for the world. How is your day?"
    input_ids = tokenizer.encode(input_text, return_attention_mask=True, return_tensors='pt')
   
    # Define position_ids and attention_mask
    past_key_values = DynamicCache()
    generated_ids = input_ids
    
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64)
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)  # Shape: (1, sequence_length)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.float32)  # Shape: (1, sequence_length)
    past_key_values = None

    num_passes = 60  # Define how many forward passes you want

    for pass_num in range(num_passes):
        print(f"\nForward Pass {pass_num + 1}:")
        print("Input IDs shape:", input_ids.shape)         
        print("Position IDs shape:", position_ids.shape)
        print("Cache position: ", cache_position.shape)
        print("Attention mask: ", attention_mask.shape)

        # Forward pass to get past key-values and present key-values
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            #cache_position=cache_position,
                            use_cache=True,
                            past_key_values=past_key_values)  # Use past key-values from the previous pass

        past_key_values = outputs.past_key_values

        print(outputs.past_key_values[0][0].shape)
        print(outputs.past_key_values[0][1].shape)

        input_ids = outputs.logits[:, -1:].argmax(-1)
        generated_ids = torch.cat([generated_ids, input_ids], dim=-1)

        print(tokenizer.batch_decode(generated_ids))
        
        #generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        
        next_position_id = position_ids[0][-1] + 1
        print(attention_mask)
        position_ids = torch.tensor([[next_position_id]])
        cache_position = cache_position[-1:] + 1

if __name__ == "__main__":
    run_inference()
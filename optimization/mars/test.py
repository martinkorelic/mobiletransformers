from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

from optimization.mars.config import MarsConfig
from optimization.mars.model import MarsModel
import torch

if __name__ == "__main__":

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mars_config = MarsConfig(
        peft_type="MARS",
        ranks=[32, 16],
        lora_alphas=[1],  # Scaling factor
        target_modules=["q_proj"],  # Target specific model layers
        task_type="CAUSAL_LM"
    )

    peft_model = MarsModel(base_model, mars_config)

    peft_model.train()
    trainable_layers = [name for name, param in peft_model.named_parameters() if param.requires_grad]
    print(trainable_layers)
    
    # Set parameters for generation
    max_length = 10  # Maximum length of the generated text
    num_return_sequences = 1  # Number of generated sequences (if you want multiple completions)
    eos_token_id = tokenizer.eos_token_id  # End-of-sequence token (if available)

    input_text = "Once upon a time in a land far, far away,"

    inputs = tokenizer(input_text, return_tensors="pt")

    # Begin generation loop
    generated_text = input_text
    input_ids = inputs["input_ids"]
    attention_ids = inputs["attention_mask"]

    peft_model.eval()
    with torch.no_grad():  # Disable gradients for inference
        for _ in range(max_length):
            # Perform a forward pass to predict the next token
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_ids)

            # Extract logits (predicted scores for the next token)
            logits = outputs.logits

            # Get the predicted token (argmax over the last token's logits)
            next_token_id = torch.argmax(logits[0, -1]).item()

            # Decode the predicted token into text
            predicted_token = tokenizer.decode(next_token_id)

            # Append the predicted token to the generated text
            generated_text += predicted_token

            print(generated_text)

            # Append the predicted token to the input_ids for the next iteration
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=input_ids.device)], dim=1)

            # Stop generating if the end-of-sequence token is generated
            if next_token_id == eos_token_id:
                break

# Print the generated text
print(generated_text)
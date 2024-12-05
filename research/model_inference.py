import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "TinyLlama/TinyLlama_v1.1"
MERGED_MODEL_PATH = "./merged_models/tinyllama-v1.1-hellaswag-loraxs-32"

def test_model_inference(model_id, merged_model_path, gen_config = None, chat_template=True, compare_to_original=True):

    prompt = f"""Hello how are you doing"""

    tokenizer = AutoTokenizer.from_pretrained(model_id)    
    model = AutoModelForCausalLM.from_pretrained(merged_model_path)
    
    if gen_config:
        gen_config = GenerationConfig(
            **gen_config
        )

    tokenized_input = None

    if chat_template:
        messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds to the user queries.",
        },
        {"role": "user", "content": "What is the meaning of 42?"},
        ]
        tokenized_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        tokenized_input = tokenizer(prompt, return_tensors="pt")
    
    generated = model.generate(input_ids=tokenized_input["input_ids"], attention_mask=tokenized_input["attention_mask"], max_new_tokens=256)
    
    print("Merged model generated:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    print("-------------------------------------")

    if compare_to_original:
        del model
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(model_id)
        generated = model.generate(input_ids=tokenized_input["input_ids"], attention_mask=tokenized_input["attention_mask"], max_new_tokens=256)
        print("Original model generated:")
        print(tokenizer.decode(generated[0], skip_special_tokens=True))

if __name__ == "__main__":
    test_model_inference(
        model_id=MODEL_ID,
        merged_model_path=MERGED_MODEL_PATH,
        chat_template=False
    )
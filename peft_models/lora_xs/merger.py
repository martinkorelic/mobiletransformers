from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import argparse
import torch
import os
import json
from pathlib import Path
from safetensors import safe_open
from .initialization_utils import find_and_initialize


def main(args):

    if args.peft_method == "lora":
        return load_and_merge_lora_model(model_name_or_path=args.base_model,
                                         peft_model_path=args.adapter,
                                         save_directory=args.output_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        # torch_dtype=torch.float16,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, device_map='auto')
    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        lora_config_dict = json.load(f)
    lora_config = LoraConfig(**lora_config_dict)

    model = get_peft_model(model, lora_config)

    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    # TODO: Hardcoded
    reconstr_config = {
        'reconstruction_type': "svd",
        'reconstr_mode': "separated",
        'half_init_dec': False,
        'replacement_module_random_init': False,
        'r_squared': True,
        'svd': {
            # TODO: hardcoded
            'rank': args.rank,
            'n_iter': 10,
            'random_state': 42
        }
    }

    reconstr_type = reconstr_config['reconstruction_type']

    # in order to accelerate model preparation, svd iterations will be set to 1.
    reconstr_config['svd']['n_iter'] = 1

    find_and_initialize(model, peft_config_dict, adapter_name=adapter_name, reconstr_type=reconstr_type, writer=None, reconstruct_config=reconstr_config)

    peft_model_weights = {}
    with safe_open(os.path.join(args.adapter, "adapter_model.safetensors"),
                   framework="pt", device="cpu") as f:
        for key in f.keys():
            peft_model_weights[key] = f.get_tensor(key)
    renamed_state_dict = {
        k.replace(
            "lora_A", "lora_A.default"
        ).replace(
            "lora_B", "lora_B.default"
        ).replace(
            "_lora_latent", ".default_lora_latent"): v
        for (k, v) in peft_model_weights.items() if "classifier.out_proj" not in k
    }
    model.load_state_dict(renamed_state_dict, strict=False)
    print("merging the LoRA into the base model.")
    model = model.merge_and_unload()
    print("Saving the merged model to disk.")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

def load_and_merge_lora_model(model_name_or_path, peft_model_path, save_directory):
    """
    Loads a base model, applies a LoRA adapter, merges the LoRA adapters into the model, 
    and saves the merged model and tokenizer.
    
    Args:
        model_name_or_path (str): Path or name of the base model.
        peft_model_path (str): Path to the directory containing the LoRA adapter.
        save_directory (str): Directory to save the merged model and tokenizer.
    """
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load PEFT config and initialize LoRA model
    peft_config = PeftConfig.from_pretrained(peft_model_path)
    model = PeftModel.from_pretrained(model, peft_model_path, config=peft_config)
    
    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    # Save the merged model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to {save_directory}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge Adapter to Base Model')
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--adapter', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--peft_method', type=str)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    main(args)

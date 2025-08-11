import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, GenerationConfig

def export_tokenizer_config(model_name_or_path, output_dir="build", hf_token=None, trust_remote_code=True):
    """
    Export tokenizer and config files from HuggingFace model.
    
    Args:
        model_name_or_path (str): HuggingFace model name or local path
        output_dir (str): Output directory (default: "build")
        hf_token (str): HuggingFace token for private models
        trust_remote_code (bool): Whether to trust remote code
    
    Returns:
        dict: The generated config dictionary
    """
    
    # Create output directories
    tokenizer_dir = Path(output_dir) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load tokenizer and config
        print(f"Loading tokenizer from {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            token=hf_token, 
            trust_remote_code=trust_remote_code
        )
        
        try:
            config = GenerationConfig.from_pretrained(model_name_or_path, token=hf_token, trust_remote_code=True)
        except:
            config = AutoConfig.from_pretrained(model_name_or_path, token=hf_token, trust_remote_code=True)
        
        # Save tokenizer files to build/tokenizer directory
        print(f"Saving tokenizer files to {tokenizer_dir}...")
        tokenizer.save_pretrained(tokenizer_dir)
        
        # Get model type
        model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'

        # Create the config structure
        ortmobile_config = {
            "model": {
                "bos_token_id": getattr(config, 'bos_token_id', tokenizer.bos_token_id or 1),
                "context_length": getattr(config, 'max_position_embeddings', 
                                        getattr(config, 'max_sequence_length', 2048)),
                "num_attention_heads": getattr(config, 'num_attention_heads', 12),
                "num_hidden_layers": getattr(config, 'num_hidden_layers', 12),
                "num_key_value_heads": getattr(config, 'num_key_value_heads', 
                                             getattr(config, 'num_attention_heads', 12)),
                "eos_token_id": config.eos_token_id,
                "pad_token_id": config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id,
                "type": model_type,
                "vocab_size": getattr(config, 'vocab_size', len(tokenizer.get_vocab()))
            }
        }
        
        # Save the main config file
        config_path = Path(output_dir) / "tokenizer" / "ortmobile_tokenizer_config.json"
        print(f"Saving main config to {config_path}...")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(ortmobile_config, f, indent=4, ensure_ascii=False)
        
        print("Export completed successfully!")
        print(f"Files saved:")
        print(f"  - Main config: {config_path}")
        print(f"  - Tokenizer files: {tokenizer_dir}")
        
        # List tokenizer files that were saved
        tokenizer_files = list(tokenizer_dir.glob("*.json"))
        for file in tokenizer_files:
            print(f"    - {file.name}")
        
        return ortmobile_config
        
    except Exception as e:
        print(f"Error exporting tokenizer: {str(e)}")
        raise

def export_tokenizer_config_advanced(model_name_or_path, output_dir="build", hf_token=None, 
                                   trust_remote_code=True, extra_config_overrides=None):
    """
    Advanced version with additional configuration options.
    
    Args:
        model_name_or_path (str): HuggingFace model name or local path
        output_dir (str): Output directory
        hf_token (str): HuggingFace token
        trust_remote_code (bool): Whether to trust remote code
        extra_config_overrides (dict): Additional config values to override
    
    Returns:
        dict: The generated config dictionary
    """
    
    config = export_tokenizer_config(model_name_or_path, output_dir, hf_token, trust_remote_code)
    
    # Apply any overrides
    if extra_config_overrides:
        for key, value in extra_config_overrides.items():
            if key in config["model"]:
                config["model"][key] = value
                print(f"Override applied: {key} = {value}")
        
        # Save updated config
        config_path = Path(output_dir) / "ortmobile_tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    return config
"""
Script that validates the generation / inference of the inference artifact model.
"""

import argparse, os, json
import textwrap
from typing import Dict, List
import yaml

from dotenv import load_dotenv
load_dotenv()

from inference.generator import generate_tokens_onnx
from tools.utils import create_chat_input
from tools.parser_config import TRAIN_CONFIG, ARTIFACT_CONFIG, ARTIFACT_VALIDATOR_CONFIG
from onnxruntime import InferenceSession
from transformers import AutoTokenizer, AutoConfig

def validate_generation(model_id, model_name, model_dir, use_gen_config_file, test_generation, test_generation_config):

    model_path = os.path.join(model_dir, model_name)
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    genai_config_path = os.path.join(model_dir, "genai_config.json")

    if use_gen_config_file:
        # Overwrite the generation config
        with open(genai_config_path, "r", encoding="utf-8") as infile:
            test_generation_config = json.load(infile)["search"]
    
    if test_generation:
        
        tokenizer_config = {}
        # Load the tokenizer configuration from the JSON file
        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)

        if "chat_template" in tokenizer_config:
            test_generation_config["prompt"] = create_chat_input(test_generation_config["prompt"], tokenizer_config)
            print("[INFO] Updated the prompt with chat template:")
            print(test_generation_config["prompt"])
        
        session = InferenceSession(model_path, providers=['CPUExecutionProvider'])
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
        config = AutoConfig.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

        input_names = [input_name.name for input_name in session.get_inputs()]

        generate_tokens_onnx(tokenizer,
                             session,
                             config,
                             with_past=any("past_key" in inpn for inpn in input_names),
                             with_position_ids=("position_ids" in input_names),
                             **test_generation_config)


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
    parser = argparse.ArgumentParser(description="Validator for exported ONNX graphs for on-device inference.", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model_id",
        type=str,
        help="Identifier for the model to be converted."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments."
    )
    parser.add_argument(
        "--inference_artifact_dir",
        type=str,
        help="Path to inference artifact directory."
    )
    parser.add_argument(
        "--inference_artifact_name",
        type=str,
        help="Name of the inference artifact model."
    )
    parser.add_argument(
        "--test_generation",
        type=bool,
        default=True,
        help="Whether to perform inference / generation test on the inference exported model."
    )
    parser.add_argument(
        "--use_gen_config_file",
        type=bool,
        default=False,
        help="Whether to use the provided genai config file or not."
    )
    parser.add_argument(
        "--test_generation_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            prompt = Hello... : Prompt for test generation. If using chatbot template setting, please provide in the chat template format as well.
            decode_between = true : Whether to decode the text while it's generating.
            max_length = 100 : Max length of test sequence to generate.
            sampling = top_k : Sampling method. Should support topk and topp
            temperature = 0.7 : Temperature for sampling
            top_k = 10 : Top K for sampling
            top_p = 0.3 : Top P for sampling
            """
            )
    )
    args = parser.parse_args()

    user_test_generation_config = {}
    default_test_generation_config = {
       "prompt": "Hello, this is a message for the world. How is your day?", # Prompt for test generation
        "decode_between": True, # Whether to decode the text while it's generating
        "max_length" : 100, # Max length of test sequence to generate
        "sampling": "topk", # Sampling method
        "temperature": 0.7, # Temperature for sampling
        "top_k": 10, # Top K for sampling
    }

    config_dict = None

    if args.config_file:
        config_dict = load_config_from_file(args.config_file)

        # Specific
        setattr(args, "model_id", config_dict[TRAIN_CONFIG]["model_id"])
        setattr(args, "inference_artifact_dir", os.path.join(config_dict[ARTIFACT_CONFIG]["build_path"], "inference"))
        setattr(args, "inference_artifact_name", f'{config_dict[ARTIFACT_CONFIG]["inference_config"]["output_inference_model"]}.onnx')
        
        # Override any command-line argument with values from the config file
        for key, value in config_dict[ARTIFACT_VALIDATOR_CONFIG].items():
            
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)

    else:
        user_test_generation_config = parse_extra_options(args.test_generation_config)
        args.test_generation_config = {**default_test_generation_config, **user_test_generation_config}

    return args


if __name__ == "__main__":
    args = parse_arguments()

    print(f"{ARTIFACT_VALIDATOR_CONFIG} arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    validate_generation(
        model_id=args.model_id,
        model_name=args.inference_artifact_name,
        model_dir=args.inference_artifact_dir,
        use_gen_config_file=args.use_gen_config_file,
        test_generation=args.test_generation,
        test_generation_config=args.test_generation_config
    )
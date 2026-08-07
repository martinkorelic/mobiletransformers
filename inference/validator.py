"""
Script that validates the generation / inference of the inference artifact model.
"""

import argparse, os, json
import textwrap
from typing import Dict, List
import numpy as np
import yaml

from dotenv import load_dotenv
load_dotenv()

from inference.generator import generate_tokens_onnx
from tools.utils import create_chat_input
from tools.parser_config import TRAIN_CONFIG, ARTIFACT_CONFIG, ARTIFACT_VALIDATOR_CONFIG

import onnxruntime as rt
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer, AutoConfig
from mobiletransformers.config.settings import get_settings

def validate_generation(model_id, model_name, model_dir, test_generation, test_generation_config, load_merged_weights=False, **kwargs):

    model_path = os.path.join(model_dir, model_name)

    tokenizer_config_path = kwargs.get('tokenizer_dir', model_dir)
    tokenizer_config_path = os.path.join(tokenizer_config_path, "tokenizer_config.json")

    if test_generation_config['type'] == 'genai':
        genai_config_path = os.path.join(model_dir, "genai_config.json")
        # Overwrite the generation config
        with open(genai_config_path, "r", encoding="utf-8") as infile:
            test_generation_config = json.load(infile)["search"]
    elif test_generation_config['type'] == 'native':
        genai_config_path = os.path.join(model_dir, "generation_config.json")
        with open(genai_config_path, "r", encoding="utf-8") as infile:
            file_generation_config = json.load(infile)
        
        # Overwrite from test_generation_config
        for key, v in file_generation_config.items():
            test_generation_config[key] = v
    
    if test_generation:

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=get_settings().require_hf_token())
        
        if test_generation_config["hf_tokenizer"]:
            if tokenizer.chat_template is not None:
                messages = [
                    {"role": "user", "content": test_generation_config["prompt"]}
                ]
                test_generation_config["prompt"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            tokenizer_config = {}
            # Load the tokenizer configuration from the JSON file
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)

            if "chat_template" in tokenizer_config:
                test_generation_config["prompt"] = create_chat_input(test_generation_config["prompt"], tokenizer_config)
                print("[INFO] Updated the prompt with chat template:")
                print(test_generation_config["prompt"])

        sess_options = SessionOptions()    
        sess_options.enable_profiling = False
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Optionally enable other optimizations
        # e.g. enable CPU memory arena for faster allocation
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        external_initializers = []

        if load_merged_weights:

            merged_weights_dir = kwargs.get('temp_weights_dir', './build/train/temp_weights/')

            print(f"[INFO] Loading external initializers from {merged_weights_dir}")
            for fname in os.listdir(merged_weights_dir):
                if fname.endswith(".npz"):
                    npz_path = os.path.join(merged_weights_dir, fname)
                    weights = np.load(npz_path)
                    base_layer_name = os.path.splitext(fname)[0]

                    for key in weights.files:
                        arr = weights[key]
                        full_key_name = f"{base_layer_name}.{key}"
                        initializer = rt.OrtValue.ortvalue_from_numpy(arr)
                        external_initializers.append((full_key_name, initializer))
                        print(f"[DEBUG] External initializer: {full_key_name} shape={arr.shape}")

        if external_initializers:
            # convert to ONNX ExternalInitializer format
            # this is basically a dict of names to OrtValues
            external_init_map = {name: val for name, val in external_initializers}
            session = InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider'],
                external_initializers=external_init_map
            )
            print("[DEBUG] Session initializers:")
            for init in session.get_overridable_initializers():
                print(f"  - {init}")
        else:
            session = InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

        config = AutoConfig.from_pretrained(model_id, token=get_settings().require_hf_token())

        input_names = [input_name.name for input_name in session.get_inputs()]

        generate_tokens_onnx(tokenizer,
                             session,
                             config,
                             with_past=any("past_key" in inpn for inpn in input_names),
                             with_position_ids=("position_ids" in input_names),
                             with_labels=any("labels" in inpn for inpn in input_names),
                             **test_generation_config)

class MobileTransformerGenerator:
    """
    A reusable class for ONNX model generation that loads configuration once
    and allows multiple generation calls.
    """
    
    def __init__(self, model_id, model_name, model_dir, generation_config = {'type': 'native'}, 
                 load_merged_weights=False, merged_weights_dir = None, **kwargs):
        """
        Initialize the ONNX model generator.
        
        Args:
            model_id (str): HuggingFace model ID
            model_name (str): Name of the model file
            model_dir (str): Directory containing the model
            generation_config (dict): Generation configuration
            load_merged_weights (bool): Whether to load merged weights
            **kwargs: Additional configuration parameters
        """
        self.model_id = model_id
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, model_name)
        self.kwargs = kwargs
        self.merged_weights_dir = merged_weights_dir
        
        # Load and process generation config
        self.generation_config = self._load_generation_config(generation_config)
        
        # Initialize tokenizer
        self.tokenizer = self._initialize_tokenizer()
        
        # Initialize ONNX session
        self.session = self._initialize_session(load_merged_weights)
        
        # Load model config
        self.config = AutoConfig.from_pretrained(model_id, token=os.environ.get('HF_TOKEN'))
        
        # Get input configuration
        self.input_names = [input_name.name for input_name in self.session.get_inputs()]
        self.input_config = self._determine_input_config()
        
        print(f"[INFO] MobileTransformerGenerator initialized successfully")
        print(f"[INFO] Model: {model_name}")
    
    def _load_generation_config(self, test_generation_config):
        """Load and process generation configuration."""
        config = test_generation_config.copy()
        
        if config['type'] == 'genai':
            genai_config_path = os.path.join(self.model_dir, "genai_config.json")
            with open(genai_config_path, "r", encoding="utf-8") as infile:
                config = json.load(infile)["search"]
        elif config['type'] == 'native':
            genai_config_path = os.path.join(self.model_dir, "generation_config.json")
            with open(genai_config_path, "r", encoding="utf-8") as infile:
                file_generation_config = json.load(infile)
            
            # Merge configurations
            for key, v in file_generation_config.items():
                config[key] = v
        
        return config
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer with chat template support."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            token=os.environ.get('HF_TOKEN')
        )
        
        # Store tokenizer configuration for chat template
        tokenizer_config_path = self.kwargs.get('tokenizer_dir', self.model_dir)
        tokenizer_config_path = os.path.join(tokenizer_config_path, "tokenizer_config.json")
        
        self.tokenizer_config = {}
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r') as f:
                self.tokenizer_config = json.load(f)
        
        return tokenizer
    
    def _initialize_session(self, load_merged_weights):
        """Initialize ONNX runtime session with optional external initializers."""
        sess_options = SessionOptions()    
        sess_options.enable_profiling = False
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        #sess_options.log_severity_level = 0  # Enable verbose logging
        #sess_options.log_verbosity_level = 0
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        if load_merged_weights:
            external_names, external_values = self._load_external_initializers()
        
        if load_merged_weights:
            sess_options.add_external_initializers(external_names, external_values)
            session = InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            print(f"[INFO] Loaded {len(external_names)} external initializers")
        else:
            session = InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
        
        return session
    
    def _load_external_initializers(self):
        """Load external initializers from merged weights directory."""
        
        external_initializers = []
        
        if not os.path.exists(self.merged_weights_dir):
            print(f"[WARNING] Merged weights directory not found: {self.merged_weights_dir}")
            return external_initializers
        
        print(f"[INFO] Loading external initializers from {self.merged_weights_dir}")
        
        external_names = []
        external_values = []

        for fname in os.listdir(self.merged_weights_dir):
            if fname.endswith(".npz"):
                npz_path = os.path.join(self.merged_weights_dir, fname)
                weights = np.load(npz_path)
                base_layer_name = os.path.splitext(fname)[0]
                
                for key in weights.files:
                    arr = weights[key]
                    full_key_name = f"{base_layer_name}.{key}"

                    # Renaming conventions
                    full_key_name = full_key_name.replace("self_attn", "attn")
                    full_key_name = full_key_name.replace("base_layer", "MatMul")
                    full_key_name = full_key_name.replace("backbone.model", "model")

                    initializer = rt.OrtValue.ortvalue_from_numpy(arr)
                    external_values.append(initializer)
                    external_names.append(full_key_name)
                    print(f"[DEBUG] External initializer: {full_key_name} shape={arr.shape}")
        
        return external_names, external_values
    
    def _determine_input_config(self):
        """Determine input configuration based on model inputs."""
        return {
            'with_past': any("past_key" in inpn for inpn in self.input_names),
            'with_position_ids': "position_ids" in self.input_names,
            'with_labels': any("labels" in inpn for inpn in self.input_names),
        }
    
    def _prepare_prompt(self, prompt):
        """Prepare prompt with chat template if available."""
        if self.generation_config.get("hf_tokenizer", True):
            if self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
        else:
            if "chat_template" in self.tokenizer_config:
                return create_chat_input(prompt, self.tokenizer_config)
        
        return prompt
    
    def generate(self, 
                 prompt="Hello, how is your day?",
                 max_length=100,
                 sampling=None,
                 output_name="logits",
                 decode_between=False,
                 use_chat_template=False,
                 **generation_kwargs):
        """
        Generate text using the loaded ONNX model.
        
        Args:
            prompt (str): Input prompt for generation
            max_length (int): Maximum length of generated text
            sampling (dict): Sampling configuration
            output_name (str): Name of the output tensor
            decode_between (bool): Whether to decode between generation steps
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text or tokens based on configuration
        """
        # Default sampling configuration
        if sampling is None:
            sampling = self.generation_config["sampling"]
        
        # Prepare the prompt with chat template if needed
        if use_chat_template:
            prompt = self._prepare_prompt(prompt)
        
        if decode_between:
            print(f"[INFO] Generating with prompt:\n{prompt}")
        
        # Call the generation function with all necessary parameters
        return generate_tokens_onnx(
            tokenizer=self.tokenizer,
            model=self.session,
            config=self.config,
            prompt=prompt,
            max_length=max_length,
            sampling=sampling,
            output_name=output_name,
            decode_between=decode_between,
            **self.input_config,
            **generation_kwargs
        )
    
    def batch_generate(self, prompts, **generation_kwargs):
        """
        Generate text for multiple prompts.
        
        Args:
            prompts (list): List of input prompts
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"[INFO] Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt=prompt, **generation_kwargs)
            results.append(result)
        return results
    
    def update_generation_config(self, new_config):
        """
        Update generation configuration.
        
        Args:
            new_config (dict): New configuration parameters
        """
        self.generation_config.update(new_config)
        print(f"[INFO] Generation configuration updated")
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_id': self.model_id,
            'model_path': self.model_path,
            'input_names': self.input_names,
            'input_config': self.input_config,
            'vocab_size': len(self.tokenizer) if self.tokenizer else None,
            'generation_config': self.generation_config
        }

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
    parser = argparse.ArgumentParser(description="Validator for exported ONNX artifacts for on-device inference.", formatter_class=argparse.RawTextHelpFormatter)

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
        "--load_merged_weights",
        type=bool,
        default=True,
        help="Whether to load the merged weights into inference model."
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
            hf_tokenizer = False : Whether to use HF tokenizer or tokenizer from local files
            """
            )
    )
    args = parser.parse_args()

    user_test_generation_config = {}
    default_test_generation_config = {
       "prompt": "Hello, this is a message for the world. How is your day?", # Prompt for test generation
        "decode_between": True, # Whether to decode the text while it's generating
        "max_length" : 100, # Max length of test sequence to generate
        "sampling": "top_k", # Sampling method
        "temperature": 0.7, # Temperature for sampling
        "top_k": 10, # Top K for sampling,
        "hf_tokenizer": False
    }

    config_dict = None

    extra_args = {}

    if args.config_file:
        config_dict = load_config_from_file(args.config_file)

        # Specific
        setattr(args, "model_id", config_dict[TRAIN_CONFIG]["model_id"])
        setattr(args, "inference_artifact_dir", os.path.join(config_dict[ARTIFACT_CONFIG]["build_path"], "inference"))
        setattr(args, "inference_artifact_name", f'{config_dict[ARTIFACT_CONFIG]["inference_export_config"]["output_inference_model"]}.onnx')

        extra_args['tokenizer_dir'] = os.path.join(config_dict[ARTIFACT_CONFIG]["build_path"], "tokenizer")
        extra_args['temp_weights_dir'] = os.path.join(config_dict[ARTIFACT_CONFIG]["build_path"], "train", "temp_weights")
        
        # Override any command-line argument with values from the config file
        for key, value in config_dict[ARTIFACT_VALIDATOR_CONFIG].items():
            
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)
            if key in extra_args:
                extra_args[key] = value
            

    else:
        user_test_generation_config = parse_extra_options(args.test_generation_config)
        args.test_generation_config = {**default_test_generation_config, **user_test_generation_config}

    return args, extra_args


if __name__ == "__main__":
    args, extra_args = parse_arguments()

    print(f"{ARTIFACT_VALIDATOR_CONFIG} arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    print(extra_args)

    validate_generation(
        model_id=args.model_id,
        model_name=args.inference_artifact_name,
        model_dir=args.inference_artifact_dir,
        test_generation=args.test_generation,
        test_generation_config=args.test_generation_config,
        load_merged_weights=args.load_merged_weights,
        **extra_args
    )
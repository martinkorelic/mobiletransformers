from __future__ import annotations
from safetensors import safe_open
import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
from peft_models.ablation.config import AblationConfig
from peft_models.ablation.model import AblationModel
from peft_models.lora_xs.initialization_utils import find_and_initialize
from peft_models.mars.config import MarsConfig
from peft_models.mars.model import MarsModel

from peft import PeftModel, PeftConfig, get_peft_model
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import PeftType
from peft.tuners.lora import LoraConfig

from research.utils import load_mars_adapters

def add_peft_type(name, value):
    """Dynamically add a new value to the PeftType enum."""
    setattr(PeftType, name, value)
    PeftType._value2member_map_[value] = name

# Add custom PEFT type dynamically
add_peft_type("MARS", "MARS")
add_peft_type("ABLATION", "ABLATION")

PEFT_TYPE_TO_MODEL_MAPPING[PeftType("MARS")] = MarsModel
PEFT_TYPE_TO_MODEL_MAPPING[PeftType("ABLATION")] = AblationModel


from deepeval.models import DeepEvalBaseLLM
from safetensors.torch import load_file

class CustomPeftModel(DeepEvalBaseLLM):
    def __init__(self, adapter_path, model_name="PEFT model", adapter_name="lora", base_model=None, device="cuda"):
        """
        Custom LLM class that loads a base model and applies a PEFT adapter if available.
        
        Parameters:
            base_model_path (str): Path to the base model (Hugging Face model ID or local directory).
            adapter_path (str, optional): Path to PEFT adapter folder (contains `adapter_model.safetensors`).
        """
        self.name = model_name
        is_adapter_model = False
        self.adapter_name = adapter_name

        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        adapter_tensors = os.path.join(adapter_path, "adapter_model.safetensors")
        base_config_path = os.path.join(adapter_path, "config.json")
        base_tensors = os.path.join(adapter_path, "model.safetensors")
        base_model_name = base_model
        config = None

        # Check if base model exists
        if os.path.exists(base_tensors):
            
            model_path = base_tensors
        elif os.path.exists(adapter_tensors):

            is_adapter_model = True
            
            model_path = adapter_tensors
        else:
            raise FileNotFoundError("No 'adapter_config.json' or 'config.json' and their corresponding weights found!")

        if os.path.exists(base_config_path):

            config_path = base_config_path
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_model_name = json.load(f)["_name_or_path"]
            config = AutoConfig.from_pretrained(config_path)

        elif os.path.exists(adapter_config_path):
            config_path = adapter_config_path
            # Define which config we are loading in
            if adapter_name in ["lora", "qlora", "loraq4", "loraq8"]:
                config = PeftConfig.from_pretrained(adapter_path)
            elif adapter_name in ["mars", "qmars"]:
                mars_config = {}
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    mars_config = json.load(f)
                config = MarsConfig(**mars_config)
            elif adapter_name == "ablation":
                ablation_config = {}
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    ablation_config = json.load(f)
                config = AblationConfig(**ablation_config)
            elif adapter_name == "lora_xs":
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    lora_xs_config = json.load(f)
                config = LoraConfig(**lora_xs_config)

            # Load the model config to get base model path
            base_model_name = config.base_model_name_or_path

        if is_adapter_model:
            print(f"Loading PEFT adapters from {adapter_path}...")

            if adapter_name in ["qlora", "qmars", "loraq4", "loraq8"]:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if adapter_name in ["qlora", "qmars"]:
                        # Original 4-bit quantization for QLoRA/QMARS
                        quantization_config = BitsAndBytesConfig(
                            # Load the model with 4-bit quantization
                            load_in_4bit=True,
                            # Use double quantization
                            bnb_4bit_use_double_quant=True,
                            # Use 4-bit Normal Float for storing the base model weights in GPU memory
                            bnb_4bit_quant_type="nf4",
                            # De-quantize the weights to 32-bit float before the forward/backward pass
                            bnb_4bit_compute_dtype=torch.float32
                        )
                        print("Using 4-bit quantization for QLoRA/QMARS")
                        
                    elif adapter_name == "loraq4":
                        # 4-bit quantization with int4 for LoRA
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="fp4",  # Using fp4 for int4-like quantization
                            bnb_4bit_compute_dtype=torch.float32,
                        )
                        print("Using 4-bit int4 quantization for LoRAQ4")
                        
                    elif adapter_name == "loraq8":
                        # 8-bit quantization for LoRA with optimized settings
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,  # Default threshold for outlier detection
                            llm_int8_enable_fp32_cpu_offload=False,  # Set to True if you need CPU offloading
                            # llm_int8_skip_modules can be added if certain modules need to be skipped
                        )
                        print("Using 8-bit int8 quantization for LoRAQ8")
                        
                except ImportError as e:
                    print(f"Warning: BitsAndBytesConfig not available for quantization: {e}")
                    print("Loading model without quantization")
                    quantization_config = None
                    
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, 
                    quantization_config=quantization_config
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

            if adapter_name in ["lora", "qlora", "loraq4", "loraq8"]:
                self.model = PeftModel.from_pretrained(base_model, adapter_path, config=config)
            elif adapter_name == "mars" or adapter_name == "qmars":
                
                # Create PeftModel
                model = get_peft_model(base_model, config, adapter_name="mars", autocast_adapter_dtype=False)

                print(f"Model dtype before adapter loading: {next(model.parameters()).dtype}")

                # Load adapters
                self.model = load_mars_adapters(model, adapter_tensors)

                print(f"Loaded MARS adapters from {adapter_tensors}.")
            elif adapter_name == "ablation":
                
                # Create PeftModel
                model = get_peft_model(base_model, config, adapter_name="ablation", autocast_adapter_dtype=False)

                # Load adapters
                self.model = load_mars_adapters(model, adapter_tensors)

                print(f"Loaded Ablation adapters from {adapter_tensors}.")
            elif adapter_name == "lora_xs":            

                self.model = get_peft_model(base_model, config)

                adapter_name = "default"
                peft_config_dict = {adapter_name: config}

                reconstr_config = {
                    'reconstruction_type': "svd",
                    'reconstr_mode': "separated",
                    'half_init_dec': False,
                    'replacement_module_random_init': False,
                    'r_squared': True,
                    'svd': {
                        'rank': config.r,
                        'n_iter': 10,
                        'random_state': 42
                    }
                }

                reconstr_type = reconstr_config['reconstruction_type']

                # in order to accelerate model preparation, svd iterations will be set to 1.
                reconstr_config['svd']['n_iter'] = 1

                find_and_initialize(self.model, peft_config_dict, adapter_name=adapter_name, reconstr_type=reconstr_type, writer=None, reconstruct_config=reconstr_config)

                peft_model_weights = {}
                with safe_open(adapter_tensors, framework="pt", device="cpu") as f:
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
                self.model.load_state_dict(renamed_state_dict, strict=False)
        else:
            self.model = AutoModelForCausalLM.from_config(config)
            state_dict = load_file(model_path)
            #for param_name in state_dict.keys():
            #    print(param_name)
            self.model.load_state_dict(state_dict, strict=False)
        
        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load generation config if available
        try:
            self.generation_config = GenerationConfig.from_pretrained(base_model_name)

            self.model.generation_config = self.generation_config

            print("Loaded generation config from base model.")

            # Custom generation config
            self.generation_config.early_stopping = True
            
            self.generation_config.max_new_tokens = 3

        except Exception as e:
            print("No generation config found. Using default settings.")
            self.generation_config = self.model.generation_config

        # Move to device
        self.model.to(device)
        self.device = device
    
    def set_generation_config(self, early_stopping=True, max_new_tokens=20):
        self.generation_config.early_stopping = early_stopping
        self.generation_config.max_new_tokens = max_new_tokens

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the model using text-generation pipeline.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(self.device)

        input_length = inputs["input_ids"].shape[1]  # Length of the input prompt

        with torch.no_grad():
            # Use autocast for QMARS to handle dtype mismatches
            if hasattr(self, 'adapter_name') and self.adapter_name == "qmars":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)

        generated_tokens = outputs[0][input_length:] 
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.name
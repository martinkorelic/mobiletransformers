import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
from peft_models.ablation.config import AblationConfig
from peft_models.ablation.model import AblationModel
from peft_models.mars.config import MarsConfig
from peft_models.mars.model import MarsModel

from peft import PeftModel, PeftConfig, get_peft_model
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import PeftType

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
            if adapter_name == "lora" or adapter_name == "qlora":
                config = PeftConfig.from_pretrained(adapter_path)
            elif adapter_name == "mars":
                mars_config = {}
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    mars_config = json.load(f)
                config = MarsConfig(**mars_config)
            elif adapter_name == "ablation":
                ablation_config = {}
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    ablation_config = json.load(f)
                config = AblationConfig(**ablation_config)

            # Load the model config to get base model path
            base_model_name = config.base_model_name_or_path

        if is_adapter_model:
            # TODO: Which adapter model, maybe merge?
            print(f"Loading PEFT adapters from {adapter_path}...")

            if adapter_name == "qlora":
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        # Load the model with 4-bit quantization
                        load_in_4bit=True,
                        # Use double quantization
                        bnb_4bit_use_double_quant=True,
                        # Use 4-bit Normal Float for storing the base model weights in GPU memory
                        bnb_4bit_quant_type="nf4",
                        # De-quantize the weights to 32-bit float before the forward/backward pass
                        bnb_4bit_compute_dtype=torch.float32,
                    )
                    print("Using 4-bit quantization for QLoRA")
                    
                except ImportError as e:
                    print(f"Warning: BitsAndBytesConfig not available for quantization: {e}")
                    print("Loading model without quantization")
                    quantization_config = None
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quantization_config)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

            if adapter_name == "lora" or adapter_name == "qlora":
                self.model = PeftModel.from_pretrained(base_model, adapter_path, config=config)
            elif adapter_name == "mars":
                
                # Create PeftModel
                model = get_peft_model(base_model, config, adapter_name="mars")

                # Load adapters
                self.model = load_mars_adapters(model, adapter_tensors)

                print(f"Loaded MARS adapters from {adapter_tensors}.")
            elif adapter_name == "ablation":
                
                # Create PeftModel
                model = get_peft_model(base_model, config, adapter_name="ablation")

                # Load adapters
                self.model = load_mars_adapters(model, adapter_tensors)

                print(f"Loaded Ablation adapters from {adapter_tensors}.")
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

            #if adapter_name == "lora":
            self.model.generation_config = self.generation_config
            #elif adapter_name == "mars":

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
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)

        generated_tokens = outputs[0][input_length:] 
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.name

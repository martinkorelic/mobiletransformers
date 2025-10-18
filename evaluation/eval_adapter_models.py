"""
DEEPEVAL Custom PEFT Model Integration for inference and evaluation.
"""

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
    def __init__(self, adapter_path, model_name="PEFT model", adapter_name="lora", base_model=None, empty_init = False, device="cuda"):
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
            if adapter_name == "base":
                print("No corresponding adapter weights found, using only base model")
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
                if empty_init:
                    self.model = get_peft_model(base_model, config)
                else:
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
                if not empty_init:
                    self.model = load_mars_adapters(model, adapter_tensors)
                else:
                    self.model = model

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
        elif adapter_name == "base":
            self.model = AutoModelForCausalLM.from_pretrained(base_model)
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
    
    def setup_attention_viz(self, max_new_tokens: int = 1, 
                           do_sample: bool = False, 
                           early_stopping: bool = True):
        """
        Configure generation settings optimized for attention visualization.
        
        Args:
            max_new_tokens: Number of tokens to generate (keep small for viz)
            do_sample: Use sampling (False for deterministic results)
            early_stopping: Enable early stopping
        """
        self.generation_config.max_new_tokens = max_new_tokens
        self.generation_config.do_sample = do_sample
        self.generation_config.early_stopping = early_stopping
        self.generation_config.output_attentions = True  # Enable attention output
        self.generation_config.return_dict_in_generate = True  # Get structured output
        
        # Set pad token if not already set
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Attention visualization setup complete:")
        print(f"  - max_new_tokens: {max_new_tokens}")
        print(f"  - do_sample: {do_sample}")
        print(f"  - output_attentions: True")
        print(f"  - return_dict_in_generate: True")
    
    def simple_attention_viz(self, prompt: str, layer: int = -1, head: int = 0, 
                            save_path: str = "attention_plot.png", show_plot: bool = False,
                            average_heads: bool = False) -> tuple:
        """
        Simple, reliable attention visualization that works in any Python environment.
        
        Args:
            prompt: Input text prompt
            layer: Which layer to visualize (-1 for last layer)
            head: Which attention head to visualize (ignored if average_heads=True)
            save_path: Where to save the plot image
            show_plot: Whether to try displaying plot (only works in GUI environments)
            average_heads: If True, average across all heads and normalize
        
        Returns:
            tuple: (generated_text, attention_matrix)
        """
        # Check if attention viz is properly configured
        if not (hasattr(self.generation_config, 'output_attentions') and 
                self.generation_config.output_attentions):
            print("⚠️  Warning: Run setup_attention_viz() first!")
            return self.generate(prompt), None
        
        # Generate with attention
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            if hasattr(self, 'adapter_name') and self.adapter_name == "qmars":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        
        # Extract results
        generated_tokens = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        attentions = outputs.attentions[-1] if outputs.attentions else None
        
        if attentions is None:
            print("⚠️  No attention data available")
            return generated_text, None
        
        # Get tokens and handle dimension mismatch
        all_tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
        attention_seq_len = attentions[0].shape[-1]
        
        if len(all_tokens) > attention_seq_len:
            all_tokens = all_tokens[:attention_seq_len]
            print(f"✓ Adjusted tokens to match attention dimensions ({attention_seq_len})")
        
        # Extract specific layer
        if layer < 0:
            layer = len(attentions) + layer  # Convert negative indexing
        
        # Get attention matrix - either single head or averaged across all heads
        if average_heads:
            # Average across all heads: shape [batch, heads, seq_len, seq_len] -> [seq_len, seq_len]
            attention_matrix = attentions[layer][0].mean(dim=0).cpu().numpy()  # Average over head dimension
            print(f"✓ Averaged across {attentions[layer][0].shape[0]} attention heads")
            
            # Normalize the averaged attention matrix
            attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
            print(f"✓ Normalized attention weights (each row sums to 1)")
        else:
            # Single head
            attention_matrix = attentions[layer][0][head].cpu().numpy()  # [seq_len, seq_len]
        
        # Create visualization
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # For long sequences, focus on the end where the decision happens

        display_tokens = all_tokens
        
        # Clean tokens for display
        clean_tokens = []
        for token in display_tokens:
            # Clean up token representation
            clean_token = token.replace('▁', '').replace('<0x0A>', '\\n').replace('<s>', '[START]')
            if len(clean_token) == 0:
                clean_token = '_'
            elif len(clean_token) > 6:
                clean_token = clean_token[:6] + '...'
            clean_tokens.append(clean_token)
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Main attention heatmap with viridis colormap
        im = plt.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Attention Weight')
        
        # Labels and ticks
        plt.xticks(range(len(clean_tokens)), clean_tokens, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(len(clean_tokens)), clean_tokens, fontsize=8)
        
        # Title with results
        head_info = f"Averaged across all heads" if average_heads else f"Head {head}"
        normalization_info = " (Normalized)" if average_heads else ""
        plt.title(f'Attention Pattern (Layer {layer}, {head_info}){normalization_info}\n'
                 f'Generated: "{generated_text}"\n'
                 f'Last {len(clean_tokens)} tokens', fontsize=12, pad=20)
        
        # Labels
        plt.xlabel('Attended To (Keys)', fontsize=10)
        plt.ylabel('Attending From (Queries)', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Attention plot saved to: {save_path}")
        
        # Try to show if requested (only works in GUI environments)
        if show_plot:
            try:
                plt.show()
            except:
                print("⚠️  Cannot display plot in this environment, but image saved successfully")
        
        plt.close()  # Close figure to free memory
        
        print(f"✓ Attention visualization complete!")
        print(f"📊 Generated answer: '{generated_text}'")
        if average_heads:
            print(f"📈 Averaged and normalized across all attention heads")
        else:
            print(f"🎯 Showing attention head {head}")
        print(f"📁 Open {save_path} to view the attention heatmap")
        
        return generated_text, attention_matrix
 
    def analyze_final_decision(self, prompt: str, layer: int = -1, 
                              save_path: str = "decision_analysis.png", 
                              average_heads: bool = True) -> tuple:
        """
        Focus ONLY on what influenced the final token generation decision.
        Shows which tokens the model paid attention to when making its final choice.
        
        Args:
            prompt: Input text prompt
            layer: Which layer to analyze (-1 for last layer)
            save_path: Where to save the plot
            average_heads: If True, average across all heads
        
        Returns:
            tuple: (generated_text, final_attention_weights, top_influences)
        """
        # Check setup
        if not (hasattr(self.generation_config, 'output_attentions') and 
                self.generation_config.output_attentions):
            print("⚠️  Warning: Run setup_attention_viz() first!")
            return self.generate(prompt), None, None
        
        # Generate with attention
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            if hasattr(self, 'adapter_name') and self.adapter_name == "qmars":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        
        # Extract results
        generated_tokens = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        attentions = outputs.attentions[-1] if outputs.attentions else None
        
        if attentions is None:
            print("⚠️  No attention data available")
            return generated_text, None, None
        
        # Get tokens and handle dimension mismatch
        all_tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
        attention_seq_len = attentions[0].shape[-1]
        
        if len(all_tokens) > attention_seq_len:
            all_tokens = all_tokens[:attention_seq_len]
            print(f"✓ Adjusted tokens to match attention dimensions ({attention_seq_len})")
        
        # Extract specific layer
        if layer < 0:
            layer = len(attentions) + layer
        
        # Get attention matrix and focus ONLY on the last row (final decision)
        if average_heads:
            # Average across all heads: [heads, seq_len, seq_len] -> [seq_len, seq_len]
            attention_matrix = attentions[layer][0].mean(dim=0).cpu().numpy()
            print(f"✓ Averaged across {attentions[layer][0].shape[0]} attention heads")
        else:
            attention_matrix = attentions[layer][0][0].cpu().numpy()  # Just first head
        
        # Extract ONLY the last row - this shows what influenced the final decision
        final_attention = attention_matrix[-1, :]  # Shape: [seq_len]
        
        # Clean tokens for display
        clean_tokens = []
        for token in all_tokens:
            clean_token = token.replace('▁', '').replace('<0x0A>', '\\n').replace('<s>', '[START]')
            if len(clean_token) == 0:
                clean_token = '_'
            clean_tokens.append(clean_token)
        
        # Find top influences
        top_indices = final_attention.argsort()[-10:][::-1]  # Top 10 most attended tokens
        top_influences = [(clean_tokens[i], final_attention[i]) for i in top_indices if i < len(clean_tokens)]
        
        # Create visualization
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create bar chart showing what influenced the final decision
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Top plot: Bar chart of attention weights
        tokens_display = [t[:12] for t, w in top_influences]  # Truncate long tokens
        weights = [w for t, w in top_influences]
        
        # Create color gradient using viridis colormap
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(weights)))
        bars = ax1.bar(range(len(tokens_display)), weights, color=colors)
        ax1.set_xlabel('Tokens', fontsize=12)
        ax1.set_ylabel('Attention Weight', fontsize=12)
        ax1.set_title(f'What Influenced the Final Decision: "{generated_text}"\n'
                     f'Top {len(top_influences)} Most Attended Tokens (Layer {layer})', fontsize=14, pad=20)
        ax1.set_xticks(range(len(tokens_display)))
        ax1.set_xticklabels(tokens_display, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Bottom plot: Full attention sequence as line plot
        ax2.plot(range(len(final_attention)), final_attention, color='darkgreen', linewidth=2)
        ax2.fill_between(range(len(final_attention)), final_attention, alpha=0.3, color='lightgreen')
        ax2.set_xlabel('Token Position', fontsize=12)
        ax2.set_ylabel('Attention Weight', fontsize=12)
        ax2.set_title('Full Attention Pattern for Final Token Generation', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Highlight top attention positions
        for i in top_indices[:5]:  # Highlight top 5
            if i < len(final_attention):
                ax2.scatter(i, final_attention[i], color='red', s=50, zorder=5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print analysis
        print(f"\n🎯 FINAL DECISION ANALYSIS")
        print(f"Generated Token: '{generated_text}'")
        print(f"{'='*50}")
        print(f"📊 TOP INFLUENCES (what the model focused on):")
        
        for i, (token, weight) in enumerate(top_influences[:5], 1):
            percentage = weight * 100
            print(f"  {i}. '{token}' → {weight:.4f} ({percentage:.1f}%)")
        
        print(f"\n💡 INTERPRETATION:")
        question_tokens = [t for t, w in top_influences[:5] if any(keyword in t.lower() 
                          for keyword in ['france', 'capital', 'paris', 'london', 'berlin', 'rome'])]
        if question_tokens:
            print(f"  ✓ Model focused on relevant tokens: {question_tokens}")
        else:
            print(f"  ⚠️  Model attention might be on structural tokens")
        
        print(f"📁 Detailed visualization saved to: {save_path}")
        
        return generated_text, final_attention, top_influences
    
    def analyze_prediction_probabilities(self, prompt: str, top_k: int = 20, 
                                       save_path: str = "prediction_probs.png") -> dict:
        """
        Analyze the language model head predictions - what tokens were most likely to be generated.
        Shows the final softmax probabilities for top-k tokens.
        
        Args:
            prompt: Input text prompt
            top_k: Number of top predictions to show
            save_path: Where to save the probability plot
        
        Returns:
            dict: Prediction analysis with probabilities and tokens
        """
        print(f"🔍 Analyzing LM head predictions for top-{top_k} tokens...")
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            # Get model outputs with logits
            if hasattr(self, 'adapter_name') and self.adapter_name == "qmars":
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        # Get logits for the last token (next token prediction)
        logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to tokens
        top_tokens = [self.tokenizer.decode(idx.item()) for idx in top_indices]
        top_probs_list = top_probs.cpu().numpy()
        
        # Also generate the actual prediction for comparison
        with torch.no_grad():
            if hasattr(self, 'adapter_name') and self.adapter_name == "qmars":
                with torch.cuda.amp.autocast():
                    generated = self.model.generate(**inputs, generation_config=self.generation_config)
            else:
                generated = self.model.generate(**inputs, generation_config=self.generation_config)
        
        # Extract the newly generated token (not the whole sequence)
        actual_token_id = generated[0][input_length].item()  # First new token after input
        actual_text = self.tokenizer.decode(actual_token_id)
        actual_prob = probabilities[actual_token_id].item()
        
        # Create visualization
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Clean tokens for display
        clean_tokens = []
        for token in top_tokens:
            clean_token = token.replace('▁', '').replace('<0x0A>', '\\n').replace(' ', '_')
            if len(clean_token) == 0:
                clean_token = '[SPACE]'
            elif len(clean_token) > 10:
                clean_token = clean_token[:10] + '...'
            clean_tokens.append(clean_token)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top plot: Bar chart of top predictions
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(clean_tokens)))
        bars = ax1.barh(range(len(clean_tokens)), top_probs_list, color=colors)
        
        # Highlight the actual prediction if it's in top-k
        actual_in_topk = actual_text in top_tokens
        if actual_in_topk:
            actual_idx = top_tokens.index(actual_text)
            bars[actual_idx].set_color('red')
            bars[actual_idx].set_alpha(0.8)
        
        ax1.set_yticks(range(len(clean_tokens)))
        ax1.set_yticklabels(clean_tokens)
        ax1.set_xlabel('Probability', fontsize=12)
        ax1.set_title(f'Top {top_k} Token Predictions from Language Model Head\n'
                     f'Actual Generated: "{actual_text}" (prob: {actual_prob:.4f})', 
                     fontsize=14, pad=20)
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, top_probs_list)):
            ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{prob:.4f}', ha='left', va='center', fontsize=9)
        
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, max(top_probs_list) * 1.15)
        
        # Bottom plot: Probability distribution (log scale for better visualization)
        log_probs = np.log10(top_probs_list + 1e-10)  # Add small value to avoid log(0)
        ax2.bar(range(len(clean_tokens)), log_probs, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(clean_tokens)))
        ax2.set_xticklabels(clean_tokens, rotation=45, ha='right')
        ax2.set_ylabel('Log₁₀(Probability)', fontsize=12)
        ax2.set_title('Log Scale View (better for low probabilities)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed analysis
        print(f"\n🎯 LANGUAGE MODEL HEAD ANALYSIS")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}")
        print(f"📊 TOP {min(10, top_k)} PREDICTIONS:")
        
        for i, (token, prob) in enumerate(zip(top_tokens[:10], top_probs_list[:10]), 1):
            percentage = prob * 100
            indicator = " ← GENERATED" if token == actual_text else ""
            print(f"  {i:2d}. '{token}' → {prob:.6f} ({percentage:.2f}%){indicator}")
        
        print(f"\n💡 ANALYSIS:")
        print(f"  ✓ Model confidence: {top_probs_list[0]*100:.2f}% for top choice")
        print(f"  ✓ Entropy: {-np.sum(top_probs_list * np.log2(top_probs_list + 1e-10)):.3f} bits")
        
        if actual_in_topk:
            actual_rank = top_tokens.index(actual_text) + 1
            print(f"  ✓ Generated token rank: #{actual_rank} of {top_k}")
        else:
            print(f"  ⚠️  Generated token not in top-{top_k} predictions!")
        
        print(f"📁 Detailed visualization saved to: {save_path}")
        
        # Return comprehensive results
        results = {
            "prompt": prompt,
            "actual_generated": actual_text,
            "actual_probability": actual_prob,
            "actual_rank": top_tokens.index(actual_text) + 1 if actual_in_topk else None,
            "top_predictions": [
                {"token": token, "probability": float(prob), "rank": i+1}
                for i, (token, prob) in enumerate(zip(top_tokens, top_probs_list))
            ],
            "model_confidence": float(top_probs_list[0]),
            "entropy": float(-np.sum(top_probs_list * np.log2(top_probs_list + 1e-10))),
            "top_k": top_k
        }
        
        return results
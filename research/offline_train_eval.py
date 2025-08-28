import gc
import os
import json
import torch
import random
import numpy as np
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

from research.ablation_analysis import analyze_training_metrics
from research.utils import save_peft_metrics_to_npz

# Disable DeepEval telemetry logging
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig, 
    TrainingArguments, Trainer, PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.vblora import VBLoRAConfig
from peft.tuners.loha import LoHaConfig
from evaluation.eval_adapter_models import CustomPeftModel
from peft_models.ablation.config import AblationConfig
from peft_models.ablation.model import AblationModel
from peft_models.mars.config import MarsConfig
from peft_models.mars.model import MarsModel
from datasets import Dataset
from research.visualization_trainer import PEFTUsageCallback
from tools.utils import preload_dataset

from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import PeftType
from config import TASK_EPOCHS, BATCH_SIZE, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION, EXPERIMENT_RANKS
from peft_models.lora_xs.initialization_utils import find_and_initialize

def add_peft_type(name, value):
    """Dynamically add a new value to the PeftType enum."""
    setattr(PeftType, name, value)
    PeftType._value2member_map_[value] = name

# Add custom PEFT type dynamically
add_peft_type("MARS", "MARS")
add_peft_type("ABLATION", "ABLATION")

PEFT_TYPE_TO_MODEL_MAPPING[PeftType("MARS")] = MarsModel
PEFT_TYPE_TO_MODEL_MAPPING[PeftType("ABLATION")] = AblationModel

from trainer.utils import (
    process_sample_hellaswag_deepeval,
    process_sample_boolq_deepeval,
    process_sample_arc_deepeval,
    process_sample_logiqa_deepeval,
    process_sample_winogrande_deepeval
)

from deepeval.benchmarks import BoolQ, ARC, LogiQA, Winogrande, HellaSwag
from deepeval.benchmarks.modes import ARCMode

class SLModel(Enum):
    TINYLLAMA = "TinyLlama/TinyLlama_v1.1",
    LLAMA = "meta-llama/Llama-3.2-1B"
    QWEN = "Qwen/Qwen2.5-1.5B",
    GEMMA = "google/gemma-2-2b"
 
class PEFTBenchmarkDataset(Enum):
    """Enum for supported datasets with their configurations."""

    ### Easy tasks
    BOOLQ = "boolq"
    ARC_E = "arc_e"
    LOGIQA = "logiqa"
    WINOGRANDE = "winogrande"

    ### Complex tasks
    HELLASWAG = "hellaswag"
    ARC_C = "arc_c"

    ### Mobile tasks
    MINI_PERSONALQA = "mini_personalqa"
    MINI_RECOMMENDATION = "mini_recommendation"

DATASET_MAPPING = {

    ### Easy tasks
    PEFTBenchmarkDataset.BOOLQ.value : ("google/boolq", "boolq_train_deepeval"),
    PEFTBenchmarkDataset.WINOGRANDE.value: ("allenai/winogrande", "winogrande_train_deepeval", "winogrande_l"),
    PEFTBenchmarkDataset.ARC_E.value: ("allenai/ai2_arc", "arc_train_deepeval", "ARC-Easy"),
    PEFTBenchmarkDataset.LOGIQA.value: ("data/logiqa_train", "logiqa_train_deepeval"),

    ### Complex tasks
    PEFTBenchmarkDataset.HELLASWAG.value : ("Rowan/hellaswag", "hellaswag_train_deepeval"),
    PEFTBenchmarkDataset.ARC_C.value : ("allenai/ai2_arc", "arc_train_deepeval", "ARC-Challenge"),

    ### Mobile tasks
    PEFTBenchmarkDataset.MINI_PERSONALQA.value : ("data/MiniPersonalQA_train", "TODO"),
    PEFTBenchmarkDataset.MINI_RECOMMENDATION.value : ("data/MiniRecommendation_train", "TODO"),

}

class PEFTMethod(Enum):
    """Enum for supported PEFT methods."""
    LORA = "lora"
    MARS = "mars"
    LORA_XS = "lora_xs"
    LOHA = "loha"
    VB_LORA = "vb_lora"
    QLORA = "qlora"

    # Ablation variants
    ABLATION_0 = "abl_0"
    ABLATION_A = "abl_A"
    ABLATION_B = "abl_B"
    ABLATION_C = "abl_C"
    ABLATION_D = "abl_D"
    ABLATION_E = "abl_E"
    ABLATION_F = "abl_F"
    ABLATION_G = "abl_G"
    ABLATION_H = "abl_H"
    
    @classmethod
    def get_available_methods(cls):
        """Get list of available PEFT methods."""
        return [method.value for method in cls]

@dataclass
class DataCollatorForSupervisedDataset:
    """Dynamically pads input sequences for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # Convert to tensors
        input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        labels = [torch.tensor(x, dtype=torch.long) for x in labels]

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Pad sequences dynamically
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Construct attention mask dynamically
        attention_mask = input_ids.ne(pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


class PEFTEval:

    def __init__(self, model_path, benchmark : PEFTBenchmarkDataset, adapter_name="lora", n_shots=0):
        self.peft_llm = CustomPeftModel(adapter_path=model_path, adapter_name=adapter_name)
        self.benchmark_name = benchmark
        self.n_shots = n_shots
        self.output_dir = model_path

        self._setup_benchmark()

    def _setup_benchmark(self):
        if self.benchmark_name == PEFTBenchmarkDataset.LOGIQA:
            self.benchmark = LogiQA(
                n_shots=self.n_shots,
                confinement_instructions=" "
            )

            self.peft_llm.set_generation_config(
                max_new_tokens=1
            )
        elif self.benchmark_name == PEFTBenchmarkDataset.BOOLQ:
            self.benchmark = BoolQ(
                n_shots=self.n_shots,
                confinement_instructions=" "
            )

            self.peft_llm.set_generation_config(
                max_new_tokens=1
            )
        elif self.benchmark_name == PEFTBenchmarkDataset.WINOGRANDE:
            self.benchmark = Winogrande(
                n_shots=self.n_shots,
                confinement_instructions=" "
            )

            self.peft_llm.set_generation_config(
                max_new_tokens=1
            )
        elif self.benchmark_name == PEFTBenchmarkDataset.ARC_E:
            self.benchmark = ARC(
                n_shots=self.n_shots,
                mode=ARCMode.EASY,
                confinement_instructions=" "
            )
            self.peft_llm.set_generation_config(
                max_new_tokens=1
            )
        elif self.benchmark_name == PEFTBenchmarkDataset.ARC_C:
            self.benchmark = ARC(
                n_shots=self.n_shots,
                mode=ARCMode.CHALLENGE,
                confinement_instructions=" "
            )
            self.peft_llm.set_generation_config(
                max_new_tokens=1
            )
        elif self.benchmark_name == PEFTBenchmarkDataset.HELLASWAG:
            self.benchmark = HellaSwag(
                n_shots=self.n_shots,
                confinement_instructions=" "
            )

            self.peft_llm.set_generation_config(
                max_new_tokens=1
            )

    def eval(self):
        results = self.benchmark.evaluate(model=self.peft_llm)

        with open(f"{self.output_dir}/eval_results.json", "w") as f:
            json.dump({
                "task": self.benchmark_name.value,
                "results": results
            }, f, indent=4, ensure_ascii=False)

class PEFTTrainer:
    """A class for PEFT training with better configuration management."""
    
    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama_v1.1",
        dataset: str = "boolq",
        peft_method: str = "lora",
        lora_rank: int = 4,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
        bias: str = "none",
        lora_targets: Optional[List[str]] = None,
        max_dataset_length: int = None,
        dataset_split : bool = False,
        dataset_shuffle : bool = False,
        dataset_test_ratio : bool = 0.1,
        batch_size: int = 32,
        per_device_batch_size: int = 6,
        gradient_accumulation_steps : int = None,
        learning_rate: float = 3e-4,
        num_epochs: int = 1,
        max_steps: int = -1,
        warmup_ratio: float = 0.1,
        extra_peft_config : dict = {},
        output_dir: str = "./experiment_results",
        hf_token: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize PEFT Trainer.
        
        Args:
            model_id: HuggingFace model identifier
            dataset: Dataset name ('boolq', 'hellaswag', 'arc', 'logiqa') or SupportedDataset enum
            peft_method: PEFT method ('lora', 'mars', 'lora_xs', etc.) or PEFTMethod enum
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (defaults to 2*rank)
            lora_dropout: LoRA dropout rate
            bias: Bias configuration
            lora_targets: List of target modules (defaults to common transformer modules)
            max_dataset_length: Maximum samples to use
            batch_size: Total batch size
            per_device_batch_size: Batch size per device
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_ratio: Warmup ratio
            output_dir: Base output directory
            hf_token: HuggingFace token
            seed: Random seed
        """
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha or (lora_rank * 2)
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.max_dataset_length = max_dataset_length
        self.batch_size = batch_size
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.base_output_dir = output_dir
        self.hf_token = hf_token
        self.seed = seed
        self.extra_peft_config = extra_peft_config
        self.dataset_split = dataset_split
        self.dataset_shuffle = dataset_shuffle
        self.dataset_test_ratio = dataset_test_ratio
        self.max_steps = max_steps

        if gradient_accumulation_steps:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = self.batch_size // self.per_device_batch_size
        
        # Validate and set dataset configuration
        self._setup_dataset(dataset)
        
        # Validate and set PEFT method
        self._setup_peft_method(peft_method)
        
        # Setup target modules - configurable with sensible defaults
        self.lora_targets = lora_targets or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        
        # Create model-specific output directory
        self.model_name = self._create_model_name()
        self.output_dir = os.path.join(self.base_output_dir, self.model_name)
        
        # Initialize components
        self.base_model = None
        self.tokenizer = None
        self.model = None
        self.config = None
        self.dataset = None
        
        # Set seed for reproducibility
        self._set_seed()
    
    def _setup_dataset(self, dataset_input):
        """Setup dataset configuration from simple string identifier or enum."""
        # Handle both string and enum inputs

        if isinstance(dataset_input, PEFTBenchmarkDataset):
            self.dataset_config = dataset_input    
        else:
            if dataset_input.lower() not in list(DATASET_MAPPING.keys()):
                raise ValueError(f"Unsupported dataset: {dataset_input}. Choose from: {list(DATASET_MAPPING.keys())}")
            
            self.dataset_config = PEFTBenchmarkDataset[dataset_input.upper()]
        
        self.dataset_id = DATASET_MAPPING[self.dataset_config.value][0]
        self.preprocess_id = DATASET_MAPPING[self.dataset_config.value][1]

        if len(DATASET_MAPPING[self.dataset_config.value]) > 2:
            self.dataset_name = DATASET_MAPPING[self.dataset_config.value][2]
        else:
            self.dataset_name = None
    
    def _setup_peft_method(self, peft_method_input):
        """Setup PEFT method from string identifier or enum."""
        # Handle both string and enum inputs
        if isinstance(peft_method_input, PEFTMethod):
            self.peft_method = peft_method_input.value
        else:
            # Validate string input
            available_methods = PEFTMethod.get_available_methods()
            if peft_method_input not in available_methods:
                raise ValueError(f"Unsupported PEFT method: {peft_method_input}. Choose from: {available_methods}")
            
            self.peft_method = peft_method_input
    
    def _create_model_name(self) -> str:
        """Create a unique model name based on configuration."""
        # Extract model name from model_id (e.g., "TinyLlama/TinyLlama_v1.1" -> "TinyLlama_v1.1")
        model_short_name = self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id
        
        # Create name pattern: {model_name}-{peft_method}-{dataset}-r{rank}-a{alpha_ratio}
        alpha_ratio = self.lora_alpha // self.lora_rank if self.lora_rank > 0 else 1
        
        return f"{model_short_name}-{self.peft_method}-{self.dataset_config.value.lower()}-r{self.lora_rank}-a{alpha_ratio}"
    
    def _set_seed(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print(f"Loading model: {self.model_id}")

        quantization_config = None
        if self.peft_method == "qlora":
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
        
        if quantization_config is not None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                token=self.hf_token,
                quantization_config=quantization_config
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                token=self.hf_token
            )

        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            token=self.hf_token
        )
        
        self.config = AutoConfig.from_pretrained(
            self.model_id, 
            token=self.hf_token
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully. Max position embeddings: {self.config.max_position_embeddings}")
    
    def _prepare_dataset(self):
        """Prepare the dataset for training."""
        print(f"Preparing dataset: {self.dataset_id}")
        
        # Load dataset (this function needs to be implemented)
        ds = preload_dataset(self.dataset_id, self.dataset_name)
        
        # Prepare dataset with preprocessing
        self.dataset = prepare_dataset(
            ds, 
            self.preprocess_id, 
            self.tokenizer,
            max_context_length=self.config.max_position_embeddings,
            max_dataset_length=self.max_dataset_length,
            batch_size=self.batch_size,
            test_ratio=self.dataset_test_ratio,
            split=self.dataset_split,
            shuffle=self.dataset_shuffle
        )
        
        print(f"Dataset prepared: {self.dataset}")
    
    def setup_peft_model(self):
        """Setup the PEFT model."""
        print(f"Setting up PEFT model with method: {self.peft_method}")
        
        self.base_model.enable_input_require_grads()
        
        if self.peft_method == "lora":
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_targets,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type="CAUSAL_LM",
                **self.extra_peft_config
            )
            self.model = get_peft_model(self.base_model, lora_config)
        elif self.peft_method == "qlora":
            qlora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_targets,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type="CAUSAL_LM",
                **self.extra_peft_config
            )
            self.model = prepare_model_for_kbit_training(self.base_model)
            self.model = get_peft_model(self.model, qlora_config)
        elif self.peft_method == "mars":
            # Mars implementation
            self.model = create_mars_model(
                self.base_model, 
                self.lora_targets,
                r=self.lora_rank,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                alpha=self.lora_alpha,
                **self.extra_peft_config
            )
        elif self.peft_method.startswith("abl"):
            variant = self.peft_method.split("_")[1]
            
            self.model = create_ablation_model(
                self.base_model,
                variant,
                self.lora_targets,
                r=self.lora_rank,
                alpha=self.lora_alpha,
                **self.extra_peft_config
            )
        elif self.peft_method == "lora_xs":
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_targets,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type="CAUSAL_LM",
                **self.extra_peft_config
            )
            peft_config_dict = {}
            reconstruct_dict = {
                'reconstruction_type': "svd",
                'reconstr_mode': "separated",
                'half_init_dec': False,
                'replacement_module_random_init': False,
                'r_squared': True,
                'svd': {
                    'rank': self.lora_rank,
                    'n_iter': 10,
                    'random_state': self.seed
                }
            }
            peft_config_dict[self.peft_method] = lora_config
            self.model = get_peft_model(self.base_model, lora_config)
            find_and_initialize(self.model, peft_config_dict, self.peft_method, "svd", reconstruct_dict, None)

            for param in self.model.parameters():
                param.data = param.data.contiguous()
        elif self.peft_method == "loha":
            loha_config = LoHaConfig(
                r=self.lora_rank,
                alpha=self.lora_alpha,
                target_modules=self.lora_targets,
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.base_model, loha_config)
        elif self.peft_method == "vb_lora":
            vb_lora_config = VBLoRAConfig(
                r=self.lora_rank,
                target_modules=self.lora_targets,
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.base_model, vb_lora_config)

        else:
            raise ValueError(f"Unsupported PEFT method: {self.peft_method}")
        
        print(f"PEFT model setup complete. Trainable parameters: {self._count_trainable_parameters()}")
    
    def _count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save_training_config(self):
        """Save training configuration to JSON."""
        config = {
            "model_id": self.model_id,
            "dataset": {
                "name": self.dataset_config.name,
                "dataset_id": self.dataset_id,
                "preprocess_id": self.preprocess_id
            },
            "peft_config": {
                "method": self.peft_method,
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "bias": self.bias,
                "target_modules": self.lora_targets,
                "trainable_parameter_count": self._count_trainable_parameters()
            },
            "training_config": {
                "max_dataset_length": self.max_dataset_length,
                "batch_size": self.batch_size,
                "per_device_batch_size": self.per_device_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "warmup_ratio": self.warmup_ratio
            },
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "training_configuration.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training configuration saved to: {config_path}")
    
    def train(self):
        """Execute the complete training pipeline."""
        print(f"Starting training pipeline for model: {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        self._prepare_dataset()
        
        # Setup PEFT model
        self.setup_peft_model()

        # Save training configuration
        self.save_training_config()
        
        # Setup data collator
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            save_strategy="no",
            warmup_ratio=self.warmup_ratio,
            num_train_epochs=self.num_epochs,
            weight_decay=0.0,
            logging_steps=1,
            max_steps=self.max_steps,
            logging_strategy="steps",
            report_to=[],
            gradient_checkpointing=False,
            do_eval=False,
            remove_unused_columns=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            per_device_train_batch_size=self.per_device_batch_size,
            per_device_eval_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate
        )
        
        # Setup logging callback
        log_file = os.path.join(self.output_dir, "training_logs.json")
        logging_callback = PEFTUsageCallback(output_file=log_file)
        
        # Enable gradient checkpointing
        self.model.model.gradient_checkpointing_enable()
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=data_collator,
            callbacks=[logging_callback]
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        self._save_model()
        
        print(f"Training completed! Results saved to: {self.output_dir}")
    
    def _save_model(self):
        """Save the trained model directly to output directory."""
        if "lora" in self.peft_method:
            self.model.save_pretrained(self.output_dir)
            print(f"LoRA adapters saved to: {self.output_dir}")
        elif "mars" in self.peft_method:
            self.model.base_model.save_pretrained(self.output_dir)
            print(f"MARS adapters saved to: {self.output_dir}")
        elif self.peft_method.startswith("abl"):
            self.model.base_model.save_pretrained(self.output_dir)
            print(f"Ablation adapters saved to: {self.output_dir}")
        #elif self.peft_method == "vb_lora":
        else:
            # Default PEFT saving
            self.model.save_pretrained(self.output_dir)
            print(f"{self.peft_method.upper()} adapters saved to: {self.output_dir}")
    
    def save_analysis(self):
        save_peft_metrics_to_npz(self.model, f'{self.output_dir}/analysis_metrics.npz')

    def analysis(self):
        analyze_training_metrics(self.model, f'{self.output_dir}/analysis_metrics.npz')

def create_ablation_model(base_model, variant, lora_target, r, alpha, **peft_config):

    ablation_config = AblationConfig(
            variant=variant,
            peft_type="ABLATION",
            r=r,
            alpha=alpha,
            target_modules=lora_target,
            task_type=None,
            **peft_config
        )
    return get_peft_model(base_model, ablation_config, adapter_name="ablation")

def create_mars_model(base_model, lora_target, r, alpha, **peft_config):
    """Create MARS model with specified configuration."""

    # Drop unnecessary info
    if "lora_dropout" in peft_config:
        peft_config.pop("lora_dropout")

    mars_config = MarsConfig(
            peft_type="MARS",
            r=r,
            target_modules=lora_target,
            alpha=alpha,
            task_type=None,
            **peft_config
        )
    return get_peft_model(base_model, mars_config, adapter_name="mars")


def prepare_dataset(dataset: Dataset, preprocess_id, tokenizer, max_dataset_length=None, remove_long_samples=True, max_context_length=512, test_ratio=0.1, batch_size=128, split=True, shuffle=True):
    """Prepare dataset for training with specified preprocessing strategy."""

    raw_columns = dataset["train"].column_names

    if split:
        test_size = test_ratio if max_dataset_length is None else int(max_dataset_length * test_ratio)
        train_size = (1 - test_ratio) if max_dataset_length is None else int(max_dataset_length * (1 - test_ratio))
        dataset = dataset["train"].train_test_split(test_size=test_size, train_size=train_size, shuffle=shuffle)

    def process_sample(sample):
        """Process sample based on preprocessing strategy."""
        if preprocess_id == DATASET_MAPPING[PEFTBenchmarkDataset.HELLASWAG.value][1]:
            return process_sample_hellaswag_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DATASET_MAPPING[PEFTBenchmarkDataset.BOOLQ.value][1]:
            return process_sample_boolq_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DATASET_MAPPING[PEFTBenchmarkDataset.ARC_E.value][1] or preprocess_id == DATASET_MAPPING[PEFTBenchmarkDataset.ARC_C.value][1]:
            return process_sample_arc_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DATASET_MAPPING[PEFTBenchmarkDataset.LOGIQA.value][1]:
            return process_sample_logiqa_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DATASET_MAPPING[PEFTBenchmarkDataset.WINOGRANDE.value][1]:
            return process_sample_winogrande_deepeval(sample, tokenizer, (batch_size > 1))

        return tokenizer(sample, return_dict=True, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=False)
    
    def filter_sample(sample):
        """Filter samples that exceed max context length."""
        return len(sample["input_ids"]) < max_context_length

    # Convert the list of tokenized samples into a Dataset
    dataset = dataset.map(process_sample, batched=(batch_size > 1), batch_size=batch_size)

    if remove_long_samples is not None:
        dataset = dataset.filter(filter_sample, batched=False)

    dataset = dataset.remove_columns(raw_columns)

    return dataset

################################################################
###################### EXPERIMENT SCRIPTS ######################
################################################################

def custom_task_experiments(tasks, peft_method = "lora", adapter_name = "lora"):

    for task, experiment_config in tasks.items():
        for rank in experiment_config["ranks"]:
            task_epochs = TASK_EPOCHS[task]
            extra_peft_config = experiment_config.get("extra_peft_config", {})
            output_dir_res = experiment_config.get("output_dir", "./experiment_results")

            try:
                trainer = PEFTTrainer(
                    model_id="TinyLlama/TinyLlama_v1.1",
                    dataset=task,
                    peft_method=peft_method,
                    lora_rank=rank,
                    batch_size=BATCH_SIZE,
                    per_device_batch_size=PER_DEVICE_BATCH_SIZE,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                    num_epochs=task_epochs,
                    output_dir=output_dir_res,
                    extra_peft_config=extra_peft_config
                )

                trainer.train()
                output_dir = trainer.output_dir
                dataset_config = trainer.dataset_config

                del trainer
                gc.collect()

                evaluator = PEFTEval(output_dir, dataset_config, adapter_name=adapter_name)
                evaluator.eval()

                del evaluator
                gc.collect()
            except Exception as e:
                print("Error occured:", e)


def task_all_experiments(peft_method = "lora", adapter_name = "lora", output_dir_res="./experiment_results", extra_peft_config = {}):
    """
    Baseline task experiment script function.
    
    Trains and evaluates adapter methods on BoolQ, LogiQA, ARC_E, Winogrande, ARC_C and HellaSwag benchmark.
    """

    tasks = ["boolq", "logiqa", "arc_e", "winogrande", "arc_c", "hellaswag"]
    ranks = [2, 8, 32]

    # If lora_xs is used we multiply the rank by 32, because it has small amount of trainable parameters
    if peft_method == "lora_xs":
        ranks = [r * 32 for r in ranks]

    for rank in ranks:
        for task in tasks:

            try:
                trainer = PEFTTrainer(
                    model_id="TinyLlama/TinyLlama_v1.1",
                    dataset=task,
                    peft_method=peft_method,
                    lora_rank=rank,
                    batch_size=BATCH_SIZE,
                    per_device_batch_size=PER_DEVICE_BATCH_SIZE,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                    num_epochs=TASK_EPOCHS[task],
                    output_dir=output_dir_res,
                    extra_peft_config=extra_peft_config
                )

                trainer.train()
                output_dir = trainer.output_dir
                dataset_config = trainer.dataset_config

                del trainer
                gc.collect()

                evaluator = PEFTEval(output_dir, dataset_config, adapter_name=adapter_name)
                evaluator.eval()

                del evaluator
                gc.collect()
            except Exception as e:
                print("Error occured:", e)

def ablation_training_experiment():
    """
    Ablation training metric experiment.
    """

    tasks = ["boolq"]
    ranks = [32]
    epochs = [1]

    for rank in ranks:
        for task, epoch in zip(tasks, epochs):

            try:
                trainer = PEFTTrainer(
                    model_id="TinyLlama/TinyLlama_v1.1",
                    dataset=task,
                    peft_method="abl_A",
                    lora_rank=rank,
                    batch_size=32,
                    num_epochs=epoch,
                    max_steps=100,
                    extra_peft_config={
                        "metric_tracking": True
                    }
                )

                trainer.train()
                trainer.save_analysis()

                del trainer
                gc.collect()
            except Exception as e:
                print("Error occured:", e)

if __name__ == "__main__":

    def run_task_1():
        n_bits = [4, 8]

        for n_bit in n_bits:
            task_all_experiments("mars", "mars",
                                    output_dir_res=f"./experiment_results/TinyLlama_v1.1-mars-opt3-q{n_bit}",
                                    extra_peft_config={"optimization_level": 3, "quant_n_bits": n_bit})
    
    #################################

    def run_task_2():
        n_bits = [4, 8]

        for n_bit in n_bits:
            task_all_experiments("mars", "mars",
                                    output_dir_res=f"./experiment_results/TinyLlama_v1.1-mars-opt4-q{n_bit}",
                                    extra_peft_config={"optimization_level": 4, "quant_n_bits": n_bit})

    
    #################################

    def run_task_3():
    
        methods = ["lora_xs", "vb_lora", "loha"]

        for m in methods:
            task_all_experiments(m, "lora", output_dir_res=f"./experiment_results/TinyLlama_v1.1-{m}")
    
    run_task_3()
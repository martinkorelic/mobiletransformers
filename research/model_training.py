
import gc
import math
import json, random
from typing import Dict, List
from attr import dataclass
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig, TrainerState, PreTrainedTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import os
from peft import PeftModel, LoraConfig, get_peft_model
from optimization.mars.config import MarsConfig
from optimization.mars.modelv2 import MarsModel

from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import PeftType
from tools.utils import preload_dataset

def add_peft_type(name, value):
    """Dynamically add a new value to the PeftType enum."""
    setattr(PeftType, name, value)
    PeftType._value2member_map_[value] = name

# Add custom PEFT type dynamically
add_peft_type("MARS", "MARS")

PEFT_TYPE_TO_MODEL_MAPPING[PeftType("MARS")] = MarsModel

from trainer.utils import (
    process_sample_dolly,
    process_sample_alpaca,
    process_sample_hellaswag,
    process_sample_hellaswag_deepeval,
    process_sample_boolq_deepeval,
    process_sample_arc_deepeval,
    process_sample_logiqa_deepeval
)
from optimization.lora_xs.initialization_utils import find_and_initialize
from .visualization_trainer import LogStepTimerCallback, MemoryUsageCallback, ProfCallback

from optimization.mars.test import get_mars_linear_layers, visualize_layer_metrics_with_changes

from safetensors.torch import load_file, save_file

MODEL_ID = "TinyLlama/TinyLlama_v1.1"

LORA_RANK = 4
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
PEFT_CONFIG = {
    "lora_dropout": 0,
    "bias": "none",
    "lora_alpha": LORA_RANK*2
}

# Boolq: google/boolq
# HellaSWAG: Rowan/hellaswag
# ARC: allenai/ai2_arc
# LogiQA: data/logiqa_train

# TODO: Convert into enums with configs
DATASET_ID = "data/logiqa_train"#"databricks/databricks-dolly-15k"
DATASET_NAME = None
PREPROCESS_ID = "logiqa_train_deepeval"

PEFT_METHOD = "mars"

MAX_DATASET_LENGTH = 1000

BATCH_SIZE = 32

@dataclass
class DataCollatorForSupervisedDataset:
    """Dynamically pads input sequences for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # Convert to tensors
        input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        labels = [torch.tensor(x, dtype=torch.long) for x in labels]

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id  # Default to EOS if PAD is missing

        # Pad sequences dynamically
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Construct attention mask dynamically: 1 for non-pad tokens, 0 for pad tokens
        attention_mask = input_ids.ne(pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


# Define preprocessing function for tokenization
def prepare_dataset(dataset : Dataset, preprocess_id, tokenizer, max_dataset_length = None, remove_long_samples = True, max_context_length=512, test_ratio=0.1, batch_size = 128, split=True, shuffle=True):

    raw_columns = dataset["train"].column_names

    if split:
        test_size = test_ratio if max_dataset_length == None else int(max_dataset_length * test_ratio)
        train_size = (1 - test_ratio) if max_dataset_length == None else int(max_dataset_length * (1 - test_ratio))
        dataset = dataset["train"].train_test_split(test_size=test_size, train_size=train_size, shuffle=shuffle)

    def process_sample(sample):

        if preprocess_id == "alpaca":
            return process_sample_alpaca(sample, tokenizer)
        elif preprocess_id == "databricks-dolly-15k":
            return process_sample_dolly(sample, tokenizer)
        elif preprocess_id == "hellaswag_train":
            return process_sample_hellaswag(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == "hellaswag_train_deepeval":
            return process_sample_hellaswag_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == "boolq_train_deepeval":
            return process_sample_boolq_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == "arc_train_deepeval":
            return process_sample_arc_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == "logiqa_train_deepeval":
            return process_sample_logiqa_deepeval(sample, tokenizer, (batch_size > 1))

        return tokenizer(sample, return_dict=True, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=False)
    
    def filter_sample(sample):
        return len(sample["input_ids"]) < max_context_length

    # Convert the list of tokenized samples into a Dataset
    dataset = dataset.map(process_sample, batched=(batch_size > 1), batch_size=batch_size)

    if remove_long_samples != None:
        dataset = dataset.filter(filter_sample, batched=False)

    dataset = dataset.remove_columns(raw_columns)

    return dataset

def train_pipeline(model_id, dataset_id, preprocess_id, peft_method, lora_rank, lora_target, peft_config, max_dataset_length, batch_size):

    base_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=os.environ["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    config = AutoConfig.from_pretrained(model_id, token=os.environ["HF_TOKEN"])

    # Prepare dataset
    ds = preload_dataset(dataset_id, DATASET_NAME)

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared_dataset = prepare_dataset(ds, preprocess_id, tokenizer, max_context_length=config.max_position_embeddings, max_dataset_length=max_dataset_length)
    
    print(prepared_dataset)

    datacol = DataCollatorForSupervisedDataset(tokenizer=tokenizer)#DataCollatorForLanguageModeling(tokenizer, return_tensors="pt", mlm=False, padding=True)

    # Prepare the model
    lora_config = LoraConfig(
            r=lora_rank,
            target_modules=lora_target,
            task_type="CAUSAL_LM",
            **peft_config
        )
    
    # Depends on the GPU memory
    per_device_batch = 6
    
    train_args = {
        "learning_rate": 3e-4,
        "per_device_train_batch_size" : per_device_batch,
        "per_device_eval_batch_size" : per_device_batch,
        "gradient_accumulation_steps": batch_size // per_device_batch,
    }

    other_args = {
        "method": peft_method
    }

    print(train_args)

    base_model.enable_input_require_grads()

    if peft_method == "lora-xs":
        adapter_name = "default"
        peft_config_dict = {}
        reconstruct_dict = {
            'reconstruction_type': "svd",
            'reconstr_mode': "separated",
            'half_init_dec': False,
            'replacement_module_random_init': False,
            'r_squared': True,
            'svd': {
                'rank': lora_rank,
                'n_iter': 10,
                'random_state': 42
            }
        }
        peft_config_dict[adapter_name] = lora_config
        model = get_peft_model(base_model, lora_config)
        find_and_initialize(model, peft_config_dict, adapter_name, "svd", reconstruct_dict, None)

        for param in model.parameters():
            param.data = param.data.contiguous()
    elif peft_method == "mars":

        model = create_peft_model(base_model, lora_target, lora_method="mars", r=lora_rank, **peft_config)

        model.train()
    elif peft_method == "lora":
        model = get_peft_model(base_model, lora_config)
    elif peft_method.startswith("joint"):
        return train_joint_models(
            base_model, prepared_dataset, datacol, lora_target, peft_method, **peft_config
        )

    print(model)

    # Train the model
    train_model(
        model,
        prepared_dataset,
        datacol,
        train_args,
        other_args
    )

def train_joint_models(base_model, dataset_dict, datacol, lora_target, peft_method, num_cycles = 1, data_ratios = None, **peft_config):

    # TODO: Manual config
    # Create PEFT models with different ranks
    rank_a = 8
    rank_b = 4
    rank_c = rank_a + rank_b  # Target model has summed ranks

    peft_config["mixture"] = False
    peft_config["only_mixtures"] = False

    output_dir_a = "peft_model_a"
    output_dir_b = "peft_model_b"
    output_dir_c = "peft_model_c"

    total_data = len(dataset_dict["train"])
    cycle_data_size = total_data // num_cycles
    mixture_layers = {}

    # TODO: Define new learning rates for each of the cycles, or create a global learning rate
    # Learning rates for each of the models, the last learning rate should be remembered and then reused as the initial learning rate
    # In case of cosine decaying, the last learning rate always drops to 0.
    # Optimizer states are loaded from the last cycle checkpoint
    lrs = [
        3e-4,
        3e-4,
        3e-4
    ]

    for cycle in range(num_cycles):
        print(f"Starting cycle {cycle + 1}/{num_cycles}...")
        
        # Split dataset for this cycle
        start_idx = cycle * cycle_data_size
        end_idx = start_idx + cycle_data_size
        train_dataset = dataset_dict["train"].select(range(start_idx, end_idx))
        
        # Apply data ratios if provided
        if data_ratios:
            train_a_size = int(len(train_dataset) * data_ratios[0])
            train_b_size = int(len(train_dataset) * data_ratios[1])
        else:
            train_a_size = len(train_dataset) // 3
            train_b_size = len(train_dataset) // 3

        train_dataset_a = train_dataset.select(range(train_a_size))
        train_dataset_b = train_dataset.select(range(train_a_size, train_a_size + train_b_size))
        train_dataset_c = train_dataset.select(range(train_a_size + train_b_size, len(train_dataset)))
        test_dataset = dataset_dict["test"]
        
        # Create / load model A
        if peft_method == "joint_mars":
            peft_config["subspace"] = (rank_a, 1)
        
        # Update scheduler
        scheduler_args = get_cosine_scheduler(cycle, last_min_lr=lrs[0])
        lrs[0] = scheduler_args["eta_min"]

        print(scheduler_args)

        if cycle > 0:
            # Load from previous checkpoint
            if peft_method.endswith("lora"):
                checkpoint_path = get_latest_checkpoint(output_dir_a)
            else:
                checkpoint_path = f"{output_dir_a}/mars_checkpoint"
            model_a = load_peft_model(base_model, checkpoint_path, peft_method, **peft_config)
            training_args = get_training_args(output_dir_a, peft_method, scheduler_args, resume_from_checkpoint=checkpoint_path)

            # Update mixture layers if needed
            if peft_config["mixture"] and mixture_layers:
                print("Updating mixture layers...")
                model_a = update_layers(model_a, mixture_layers, [rank_a, rank_b], extract_index=0, only_mixture=peft_config["only_mixtures"])
            resume = checkpoint_path
        else:
            # Create a new PEFT model
            model_a = create_peft_model(base_model, lora_target, peft_method, rank_a, **peft_config)
            training_args = get_training_args(output_dir_a, peft_method, scheduler_args)
            resume = None

        print("Starting Model A training...")

        list_trainable_layers(model_a)
        print(count_trainable_parameters(model_a))
        # Train model A
        train_peft_model(model_a, train_dataset_a, test_dataset, datacol, output_dir_a, training_args, peft_method, resume)

        # Clear model from memory
        model_a.to("cpu")
        del model_a
        model_a = None
        torch.cuda.empty_cache()
        gc.collect()

        # Create / load model B
        if peft_method == "joint_mars":
            peft_config["subspace"] = (rank_b, 2)
        
        # Update scheduler
        scheduler_args = get_cosine_scheduler(cycle, last_min_lr=lrs[1])
        lrs[1] = scheduler_args["eta_min"]

        print(scheduler_args)

        if cycle > 0:
            # Load from previous checkpoint
            if peft_method.endswith("lora"):
                checkpoint_path = get_latest_checkpoint(output_dir_b)
            else:
                checkpoint_path = f"{output_dir_b}/mars_checkpoint"
            model_b = load_peft_model(base_model, checkpoint_path, peft_method, **peft_config)
            training_args = get_training_args(output_dir_b, peft_method, scheduler_args, resume_from_checkpoint=checkpoint_path)

            # Update mixture layers if needed
            if peft_config["mixture"] and mixture_layers:
                print("Updating mixture layers...")
                model_b = update_layers(model_b, mixture_layers, [rank_a, rank_b], extract_index=1, only_mixture=peft_config["only_mixtures"])
            resume = checkpoint_path
        else:
            # Create a new PEFT model
            model_b = create_peft_model(base_model, lora_target, peft_method, rank_b, **peft_config)
            training_args = get_training_args(output_dir_b, peft_method, scheduler_args)
            resume = None

        print("Starting Model B training...")

        # Train model B
        train_peft_model(model_b, train_dataset_b, test_dataset, datacol, output_dir_b, training_args, peft_method, resume)

        # Clear model from memory
        model_b.to("cpu")
        del model_b
        model_b = None
        del mixture_layers
        mixture_layers = None
        torch.cuda.empty_cache()
        gc.collect()

        # Create / load joint model C
        if peft_method == "joint_mars":
            peft_config["subspace"] = (rank_c, 3)

        # Update scheduler
        scheduler_args = get_cosine_scheduler(cycle, last_min_lr=lrs[2])
        lrs[2] = scheduler_args["eta_min"]

        print(scheduler_args)

        if cycle > 0:
            # Load from previous checkpoint
            if peft_method.endswith("lora"):
                checkpoint_path = get_latest_checkpoint(output_dir_c)
            else:
                checkpoint_path = f"{output_dir_c}/mars_checkpoint"
            model_c = create_joint_model(base_model, checkpoint_path_a, checkpoint_path_b, output_dir_c, lora_target, peft_method, ranks=[rank_a, rank_b], resume=True, **peft_config)
            training_args = get_training_args(output_dir_c, peft_method, scheduler_args, resume_from_checkpoint=checkpoint_path)
            resume = checkpoint_path
        else:
            # Create a new PEFT model
            # Load from previous checkpoints of model A and B
            if peft_method.endswith("lora"):
                checkpoint_path_a = get_latest_checkpoint(output_dir_a)
                checkpoint_path_b = get_latest_checkpoint(output_dir_b)
            else:
                checkpoint_path_a = f"{output_dir_a}/mars_checkpoint"
                checkpoint_path_b = f"{output_dir_b}/mars_checkpoint"
            model_c = create_joint_model(base_model, checkpoint_path_a, checkpoint_path_b, output_dir_c, lora_target, peft_method, ranks=[rank_a, rank_b], **peft_config)
            training_args = get_training_args(output_dir_c, peft_method, scheduler_args)
            resume = None
        
        print("Starting joint Model C training...")

        train_peft_model(model_c, train_dataset_c, test_dataset, datacol, output_dir_c, training_args, peft_method, resume=resume)

        # Offload the mixture layers after training the joint model
        if peft_config["mixture"]:
            # Make a new dict and delete old one
            mixture_layers = offload_layers(model_c, only_mixture=peft_config["only_mixtures"])
        
        # Clear model from memory
        model_c.to("cpu")
        del model_c
        model_c = None
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Finished cycle {cycle + 1}/{num_cycles}.")

def offload_layers(model, only_mixture=True):
    
    mixture_layers = {}
    
    for name, param in model.named_parameters():
        if only_mixture and "mixture" in name:
            #print(f"Offloading {name}")
            mixture_layers[name] = param.clone().detach().to("cpu")
        elif not only_mixture and param.requires_grad:
            #print(f"Offloading {name}")
            mixture_layers[name] = param.clone().detach().to("cpu")

    print(f"Extracted {len(mixture_layers)} mixture matrices from the model.")
    return mixture_layers

def update_layers(model, mixture_layers, ranks, extract_index=0, only_mixture=True):
    new_state_dict = {}

    for name, mixture_matrix in mixture_layers.items():
        #r_total = sum(ranks)  # Total rank should match the matrix size
        #assert mixture_matrix.shape == (r_total, r_total), f"Invalid shape for {name}: {mixture_matrix.shape}"

        start = 0
        for i, r in enumerate(ranks):
            if extract_index == i:

                if "mixture" in name:
                    new_state_dict[name] = mixture_matrix[start:start + r, start:start + r]

                # Replace only mixture layers
                if only_mixture:
                    continue
                
                # Replace all the B layers as well
                if "mixture" not in name:
                    new_state_dict[name] = mixture_matrix[:, start:start + r]

            start += r

    # Load extracted blocks into the model's state dictionary
    model_state = model.state_dict()
    model_state.update(new_state_dict)
    model.load_state_dict(model_state, strict=False)
    return model

def create_joint_model(base_model, model_a_dir, model_b_dir, model_c_dir, lora_target, peft_method, ranks, resume=False, **peft_config):
    """
    Creates Model C with LoRA rank (r3 = r1 + r2) and merges adapters from Model A and Model B.

    Args:
        model_a_dir (str): Path to Model A's saved adapter.
        model_b_dir (str): Path to Model B's saved adapter.
        model_c_dir (str): Output path for merged Model C.
        base_model (AutoModelForCausalLM): Base model (loaded once to avoid redundancy).
        r1 (int): Rank of Model A's LoRA adapter.
        r2 (int): Rank of Model B's LoRA adapter.

    Returns:
        None
    """
    
    # Define Model C with summed LoRA rank
    r = sum(ranks)

    if not resume:
        if peft_method.endswith("lora"):
            joint_peft_config = LoraConfig(
                r=r,
                lora_alpha=r,
                target_modules=lora_target,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(base_model, joint_peft_config)
            model.save_pretrained(model_c_dir)

        elif peft_method.endswith("mars"):
            joint_peft_config = MarsConfig(
                peft_type="MARS",
                r=r,
                mixture=peft_config.get("mixture", False),
                subspace=peft_config.get("subspace", (r, 3)),
                target_modules=lora_target,  # Target specific model layers
                task_type=None
            )

            model = get_peft_model(base_model, joint_peft_config, adapter_name="mars")
            model.base_model.save_pretrained(model_c_dir)
    else:
        model = load_peft_model(base_model, model_c_dir, peft_method, **peft_config)

    # Load LoRA adapters for Model A & Model B
    adapter_a = load_file(os.path.join(model_a_dir, "adapter_model.safetensors"))
    adapter_b = load_file(os.path.join(model_b_dir, "adapter_model.safetensors"))

    # Merge adapters by **concatenating along rank dimension**
    merged_adapters = {}
    for key in adapter_a.keys():
        # Ignore mixtures
        if key in adapter_b and "mixture" not in key:
            tensor_a = adapter_a[key]
            tensor_b = adapter_b[key]

            # Check which dimension matches
            if tensor_a.shape[1] == tensor_b.shape[1]:  # Match in dim=1
                concat_dim = 0
            elif tensor_a.shape[0] == tensor_b.shape[0]:  # Match in dim=0
                concat_dim = 1
            else:
                raise ValueError(f"Cannot combine {key}, incompatible shapes {tensor_a.shape} vs {tensor_b.shape}")
            
            # Both models have this key -> Concatenate tensors along rank dimension
            merged_adapters[key] = torch.cat([adapter_a[key], adapter_b[key]], dim=concat_dim)

    # Save merged adapter
    save_file(merged_adapters, os.path.join(model_c_dir, "adapter_model.safetensors"))

    # Load merged adapters
    if peft_method.endswith("lora"):
        model.model.load_state_dict(merged_adapters, strict=False)
    elif peft_method.endswith("mars"):
        model.base_model.model.load_state_dict(merged_adapters, strict=False)

    print(f"Successfully combined adapters into {model_c_dir}.")

    del adapter_a
    adapter_a = None
    del adapter_b
    adapter_b = None
    del merged_adapters
    merged_adapters = None
    gc.collect()

    return model

def get_training_args(output_dir, peft_method, scheduler_args = {}, resume_from_checkpoint=None):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=(resume_from_checkpoint is not None),
        #max_steps=1,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=2,
        logging_steps=10,
        #warmup_steps=20,
        optim="adamw_torch",
        report_to="none",  # Disable logging to TensorBoard
        logging_dir=None,  # Ensure no logging directory is used
        #logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        resume_from_checkpoint=resume_from_checkpoint,
        save_strategy=("epoch" if peft_method.endswith("lora") else "no"),
        #eval_strategy="epoch", # Evaluate at end of epoch
        #do_eval=True,
        do_train=True,
        #no_cuda=True,
        gradient_accumulation_steps=3,
        warmup_ratio=scheduler_args.get("warmup_ratio", 0),
        learning_rate=scheduler_args.get("initial_lr", 1e-5),
        lr_scheduler_type="cosine",
        # TODO
        #lr_scheduler_type="cosine_with_min_lr",
        #lr_scheduler_kwargs={
        #    "min_lr": scheduler_args.get("eta_min", 0.0)
        #}
    )

def get_cosine_scheduler(cycle, last_min_lr, initial_warmup_ratio=0.1, warmup_decay=0.8, eta_min_decay=0.2):
    """Dynamically adjust warmup steps and eta_min per cycle."""

    # Adjust warmup proportionally
    warmup_ratio = initial_warmup_ratio * (warmup_decay ** cycle)

    # Reduce min LR progressively
    eta_min = last_min_lr * eta_min_decay  # Decrease eta_min by 20% per cycle
    eta_min = max(eta_min, 1e-7)  # Ensure eta_min doesn’t go too low
    
    return  {
        "initial_lr": last_min_lr,
        "warmup_ratio": warmup_ratio,
        "eta_min": eta_min,  # Lower minimum LR in each cycle
    }


def get_latest_checkpoint(output_dir):
    """Get the latest checkpoint path from the training output directory."""
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]

    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    return os.path.join(output_dir, latest_checkpoint)

def load_mars_adapters(model, adapter_path):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
    
    # Load adapter weights
    adapter_state_dict = load_file(adapter_path)

    # Load adapters into model (allow missing keys to avoid errors)
    model.base_model.model.load_state_dict(adapter_state_dict, strict=False)

    return model

def load_peft_model(base_model, adapter_dir, peft_method, **peft_config):

    if peft_method.endswith("lora"):

        # Load PEFT adapter config
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LoRA adapter config not found in {config_path}")
        
        peft_config = LoraConfig.from_pretrained(adapter_dir)

        # Load the PEFT model with adapter
        model = PeftModel.from_pretrained(base_model, adapter_dir, config=peft_config)
        print(f"Loaded LoRA adapters from {adapter_dir}")

    elif peft_method.endswith("mars"):
        
        mars_config = {}
        with open(f"{adapter_dir}/adapter_config.json", "r", encoding="utf-8") as f:
            mars_config = json.load(f)

        # We cannot load from PeftConfig pretrained, so we manually load custom PEFT config
        peft_config = MarsConfig(**mars_config)
        
        # Create PeftModel
        model = get_peft_model(base_model, peft_config, adapter_name="mars")

        # Load adapters
        model = load_mars_adapters(model, f"{adapter_dir}/adapter_model.safetensors")

        print(f"Loaded MARS adapters from {adapter_dir}")

    return model

def create_peft_model(base_model, lora_target, lora_method, r, **peft_config):

    if lora_method.endswith("lora"):
        # Prepare the model
        lora_config = LoraConfig(
                r=r,
                target_modules=lora_target,
                task_type="CAUSAL_LM",
                lora_dropout=peft_config.get("lora_dropout", 0),
                bias=peft_config.get("bias", "none"),
                lora_alpha=peft_config.get("lora_alpha", r)
            )
        return get_peft_model(base_model, lora_config)
    elif lora_method.endswith("mars"):
        mars_config = MarsConfig(
            peft_type="MARS",
            r=r,
            alpha=peft_config.get("alpha", r),
            target_modules=lora_target,
            **peft_config,
            #optimization_level=3,
            #modules_to_quantize=['q', 'up'],
            #modules_to_preserve_errors=['up'],
            task_type=None
        )

        return get_peft_model(base_model, mars_config, adapter_name="mars")

    return None

def compute_max_steps(train_dataloader, grad_accum_steps=1, epochs=1):
    len_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = len_dataloader // grad_accum_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    return math.ceil(epochs * num_update_steps_per_epoch)

def train_peft_model(model, train_dataset, eval_dataset, data_collator, output_dir, training_args, peft_method, resume=None):
    
    # Manual seed
    set_manual_seed(42)

    # TODO: Trainer does not log out validation loss
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LogStepTimerCallback()]
    )
    
    if resume is not None:
        # This loads in optimizer and scheduler, however if the training ends and LR reaches 0
        # (if using cosine or other decaying LR scheduler), we need to somehow save the learning rate and restart from there

        print("Loading in optimizer / scheduler / state...")
        trainer.create_optimizer_and_scheduler(num_training_steps=compute_max_steps(train_dataset))

        # TODO: If we modify weights, we cannot load the previous optimizer and scheduler, difference in parameter groups
        trainer._load_optimizer_and_scheduler(resume)
        trainer._load_rng_state(resume)
        trainer.state = TrainerState.load_from_json(os.path.join(output_dir, f"trainer_state.json"))

    trainer.train()

    if peft_method.endswith("mars"):

        output_dir = f"{output_dir}/mars_checkpoint"

        model.base_model.save_pretrained(output_dir)
        trainer.save_state()

        # Save optimizer and scheduler
        trainer._save_optimizer_and_scheduler(output_dir)

        # Save RNG state
        trainer._save_rng_state(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_dir

def train_model(peft_model, encoded_dataset, data_collator, train_args_plus, other_args):

    #peft_model = (peft_model if other_args["method"] in ["mars"] else peft_model)

    # Reactivate some layers for fine-tuning (if needed)
    list_trainable_layers(peft_model)

    # Compute number of trainable parameters
    print("Number of trainable parameters:")
    print(count_trainable_parameters(peft_model))

    # Step 4: Set Up Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        #eval_strategy="steps",
        #eval_steps=1000,
        save_strategy="no",
        warmup_ratio=0.1,
        #max_steps=2,
        num_train_epochs=2,
        weight_decay=0.0,
        logging_steps=10,
        report_to="none",  # Disable logging to TensorBoard
        logging_dir=None,  # Ensure no logging directory is used
        #logging_dir="./results/logs",
        #logging_steps=10,
        # Major problem using this on AMD GPU, avoid
        #fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        do_eval=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        save_steps=2000,
        lr_scheduler_type="cosine",
        **train_args_plus
    )

    peft_model.model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        callbacks=[MemoryUsageCallback()]
    )

    trainer.train()

    save_model = True
    visualize_training = False

    if save_model:
        if "lora" in other_args["method"]:
            peft_model.save_pretrained("./results/adapters")
        elif "mars" in other_args["method"]:
            peft_model.base_model.save_pretrained("./results/adapters")

    if visualize_training:
        trainable_layers = get_mars_linear_layers(peft_model)

        visualize_layer_metrics_with_changes(trainable_layers)

def list_trainable_layers(peft_model):
    """
    Activates layers in a PEFT model for training based on specified keywords.

    Args:
        peft_model: The PEFT model to modify.
        keywords (list): A list of keyword strings. Layers with names containing these
                         keywords will be made trainable.

    Returns:
        None
    """

        #print(param.requires_grad)
        # Check if any keyword is in the layer name
        #if any(keyword in name for keyword in keywords):
        #    param.requires_grad = True  # Make the parameter trainable
        #    print(f"Layer '{name}' activated for training.")
        #else:
        #    param.requires_grad = False  # Keep other layers frozen

    # Summarize the changes
    trainable_layers = [name for name, param in peft_model.named_parameters() if param.requires_grad]
    print(f"Total trainable layers: {len(trainable_layers)}")
    print(f"Trainable layers: {trainable_layers}")

def set_manual_seed(seed):
    """Set the manual seed for reproducibility across LoRA weight initialization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Computes the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    train_pipeline(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        preprocess_id=PREPROCESS_ID,
        peft_method=PEFT_METHOD,
        lora_rank=LORA_RANK,
        lora_target=LORA_TARGET,
        peft_config=PEFT_CONFIG,
        max_dataset_length=MAX_DATASET_LENGTH,
        batch_size=BATCH_SIZE
    )
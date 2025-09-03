import json, random
from typing import Dict, List
from attr import dataclass
from enum import Enum
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig, PreTrainedTokenizer
from datasets import Dataset
import os
from peft import PeftModel, LoraConfig, get_peft_model
from peft_models.mars.config import MarsConfig
from peft_models.mars.model import MarsModel

from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import PeftType
from research.utils import load_mars_adapters
from tools.utils import preload_dataset

from config import HF_TOKEN

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
from peft_models.lora_xs.initialization_utils import find_and_initialize
from ..visualization_trainer import PEFTUsageCallback
from peft_models.mars.test import get_mars_linear_layers, visualize_layer_metrics_with_changes


class DatasetID(Enum):
    """Enum for different preprocessing strategies."""
    ALPACA = "alpaca"
    DATABRICKS_DOLLY_15K = "databricks-dolly-15k"
    HELLASWAG_TRAIN = "hellaswag_train"
    HELLASWAG_TRAIN_DEEPEVAL = "hellaswag_train_deepeval"
    BOOLQ_TRAIN_DEEPEVAL = "boolq_train_deepeval"
    ARC_TRAIN_DEEPEVAL = "arc_train_deepeval"
    LOGIQA_TRAIN_DEEPEVAL = "logiqa_train_deepeval"

###########################################
############# CONFIGURATIONS ##############
###########################################

MODEL_ID = "TinyLlama/TinyLlama_v1.1"

LORA_RANK = 4
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
PEFT_CONFIG = {
    "lora_dropout": 0,
    "bias": "none",
    "lora_alpha": LORA_RANK*2
}

# Available datasets:
# Boolq: google/boolq
# HellaSWAG: Rowan/hellaswag
# ARC: allenai/ai2_arc
# LogiQA: data/logiqa_train
DATASET_ID = "data/logiqa_train"
DATASET_NAME = None
PREPROCESS_ID = "logiqa_train_deepeval"

PEFT_METHOD = "lora"
MAX_DATASET_LENGTH = None
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

        if preprocess_id == DatasetID.ALPACA.value:
            return process_sample_alpaca(sample, tokenizer)
        elif preprocess_id == DatasetID.DATABRICKS_DOLLY_15K.value:
            return process_sample_dolly(sample, tokenizer)
        elif preprocess_id == DatasetID.HELLASWAG_TRAIN.value:
            return process_sample_hellaswag(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DatasetID.HELLASWAG_TRAIN_DEEPEVAL.value:
            return process_sample_hellaswag_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DatasetID.BOOLQ_TRAIN_DEEPEVAL.value:
            return process_sample_boolq_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DatasetID.ARC_TRAIN_DEEPEVAL.value:
            return process_sample_arc_deepeval(sample, tokenizer, (batch_size > 1))
        elif preprocess_id == DatasetID.LOGIQA_TRAIN_DEEPEVAL.value:
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

    base_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    config = AutoConfig.from_pretrained(model_id, token=HF_TOKEN)

    # Prepare dataset
    ds = preload_dataset(dataset_id, DATASET_NAME)

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared_dataset = prepare_dataset(ds, preprocess_id, tokenizer, max_context_length=config.max_position_embeddings, max_dataset_length=max_dataset_length)
    
    print(prepared_dataset)

    datacol = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

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
    #elif peft_method.startswith("joint"):
    #    return train_joint_models(
    #        base_model, prepared_dataset, datacol, lora_target, peft_method, **peft_config
    #    )

    print(model)

    # Train the model
    train_model(
        model,
        prepared_dataset,
        datacol,
        train_args,
        other_args
    )


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

def train_model(peft_model, encoded_dataset, data_collator, train_args_plus, other_args):

    #peft_model = (peft_model if other_args["method"] in ["mars"] else peft_model)

    # Reactivate some layers for fine-tuning (if needed)
    list_trainable_layers(peft_model)

    # Compute number of trainable parameters
    print("Number of trainable parameters:")
    print(count_trainable_parameters(peft_model))

    training_args = TrainingArguments(
        output_dir="./results",
        #eval_strategy="steps",
        #eval_steps=1000,
        save_strategy="no",
        warmup_ratio=0.1,
        #max_steps=2,
        num_train_epochs=2,
        weight_decay=0.0,
        logging_steps=1,
        logging_strategy="steps",
        report_to=[],  # Disable logging to TensorBoard
        #logging_dir=None,  # Ensure no logging directory is used
        logging_dir="./results/logs",
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
        callbacks=[PEFTUsageCallback()]
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
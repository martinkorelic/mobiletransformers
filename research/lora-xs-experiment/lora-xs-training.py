
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig, DataCollatorForSeq2Seq, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
import os
from peft import LoraConfig, get_peft_model

from trainer.utils import process_sample_dolly, process_sample_alpaca, process_sample_commonsense, process_sample_hellaswag
from optimization.lora_xs.initialization_utils import find_and_initialize

MODEL_ID = "TinyLlama/TinyLlama_v1.1"

LORA_RANK = 32
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "o_proj"]
PEFT_CONFIG = {
    "lora_dropout": 0,
    "bias": "none",
    "lora_alpha": LORA_RANK * 2
}

DATASET_ID = "data/hellaswag_train"#"databricks/databricks-dolly-15k"

PEFT_METHOD = "lora"

MAX_DATASET_LENGTH = None

BATCH_SIZE = 32

def preload_dataset(dataset_id):

    dataset_ids = dataset_id.split("/")
    
    # Take local data
    if len(dataset_ids) >= 2 and dataset_ids[-2] == "data":

        filepath = dataset_id
        data = None

        if os.path.exists(f'./{dataset_id}.json'):
            filepath = f'./{dataset_id}.json'

            with open(filepath, 'r', encoding="utf-8") as f:
                data = json.load(f)

        elif os.path.exists(f'./{dataset_id}.jsonl'):
            filepath = f'./{dataset_id}.jsonl'

            with open(filepath, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(data)

         # Create a DatasetDict with the "train" split
        dataset_dict = DatasetDict({"train": dataset})

        return dataset_dict

    return load_dataset(dataset_id)


# Define preprocessing function for tokenization
def prepare_dataset(dataset : Dataset, dataset_id, tokenizer, max_dataset_length = None, remove_long_samples = True, max_context_length=512, test_ratio=0.1, batch_size = 128, split=True, shuffle=False):

    raw_columns = dataset["train"].column_names

    dataset_ids = dataset_id.split("/")
    if len(dataset_ids) == 2:
        dataset_id = dataset_ids[1]

    if split:
        test_size = test_ratio if max_dataset_length == None else int(max_dataset_length * test_ratio)
        train_size = (1 - test_ratio) if max_dataset_length == None else int(max_dataset_length * (1 - test_ratio))
        dataset = dataset["train"].train_test_split(test_size=test_size, train_size=train_size, shuffle=shuffle)    

    def process_sample(sample):

        if dataset_id == "alpaca":
            return process_sample_alpaca(sample, tokenizer)
        elif dataset_id == "databricks-dolly-15k":
            return process_sample_dolly(sample, tokenizer)
        elif dataset_id == "commonsense":
            return process_sample_commonsense(sample, tokenizer, (batch_size > 1))
        elif dataset_id == "hellaswag_train":
            return process_sample_hellaswag(sample, tokenizer, (batch_size > 1))

        return tokenizer(sample, return_dict=True, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=False)
    
    def filter_sample(sample):
        return len(sample["input_ids"]) < max_context_length


    # Convert the list of tokenized samples into a Dataset
    dataset = dataset.map(process_sample, batched=(batch_size > 1), batch_size=batch_size)

    if remove_long_samples != None:
        dataset = dataset.filter(filter_sample, batched=False)

    dataset = dataset.remove_columns(raw_columns)

    return dataset


def train_loraxs(model_id, dataset_id, peft_method, lora_rank, lora_target, peft_config, max_dataset_length, batch_size):

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=os.environ["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    config = AutoConfig.from_pretrained(model_id, token=os.environ["HF_TOKEN"])

    # Prepare dataset
    ds = preload_dataset(dataset_id)

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared_dataset = prepare_dataset(ds, dataset_id, tokenizer, max_context_length=config.max_position_embeddings, max_dataset_length=max_dataset_length)
    
    print(prepared_dataset)

    datacol = DataCollatorForLanguageModeling(tokenizer, return_tensors="pt", mlm=False)

    # Prepare the model
    lora_config = LoraConfig(
            r=lora_rank,
            target_modules=lora_target,
            task_type="CAUSAL_LM",
            **peft_config
        )
    
    per_device_batch = 6
    
    train_args = {
        "learning_rate": 4e-4,
        "per_device_train_batch_size" : per_device_batch,
        "per_device_eval_batch_size" : per_device_batch,
        "gradient_accumulation_steps": batch_size // per_device_batch
    }
    
    model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)

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
        find_and_initialize(model, peft_config_dict, adapter_name, "svd", reconstruct_dict, None)

        for param in model.parameters():
            param.data = param.data.contiguous()

    # TODO: Reactivate some layers for fine-tuning (if needed)
    activate_trainable_layers(model, "")

    # Train the model
    train_model(
        model,
        prepared_dataset,
        datacol,
        train_args
    )


def train_model(model, encoded_dataset, data_collator, train_args_plus):
    # Step 4: Set Up Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=600,
        save_strategy="steps",
        warmup_steps=100,
        num_train_epochs=1,
        weight_decay=0.0,
        logging_dir="./logs",
        logging_steps=10,
        # Major problem using this on AMD GPU, avoid
        #fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=True,
        optim="adamw_torch",
        save_steps=2000,
        lr_scheduler_type="cosine",
        **train_args_plus
    )

    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained("./results/adapters")

def activate_trainable_layers(peft_model, keywords):
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

if __name__ == "__main__":
    train_loraxs(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        peft_method=PEFT_METHOD,
        lora_rank=LORA_RANK,
        lora_target=LORA_TARGET,
        peft_config=PEFT_CONFIG,
        max_dataset_length=MAX_DATASET_LENGTH,
        batch_size=BATCH_SIZE
    )
"""
Utility functions for the framework.
"""

import json
import os
import shutil
from jinja2 import Template
import psutil
import torch
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict

def load_and_save_dataset(dataset_name, save_path=None, train_file="train_dataset", split=None, save_format="jsonl", max_dataset_length=None):
    """
    Load a dataset from Hugging Face Hub and save it locally.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub
        save_path (str, optional): Local path to save the dataset. 
                                 If None, saves to './datasets/{dataset_name}'
        config_name (str, optional): Configuration name for datasets with multiple configs
        split (str, optional): Specific split to load ('train', 'test', 'validation', etc.)
        **kwargs: Additional arguments to pass to load_dataset()
    
    Returns:
        datasets.Dataset or datasets.DatasetDict: The loaded dataset
    """
    try:
        # Load the dataset
        dataset = preload_dataset(dataset_name, split)

        if type(dataset) == DatasetDict:
            dataset = dataset[split]
        
        # Set default save path if not provided
        if save_path is None:
            save_path = f"./datasets/{dataset_name.replace('/', '_')}"
        
        # Trim dataset if max_dataset_length is specified
        if max_dataset_length is not None:
            dataset = trim_dataset(dataset, max_dataset_length)

        # Save the dataset based on format
        if save_format.lower() == "jsonl":
            save_as_jsonl(dataset, save_path, train_file)
        else:
            # Default HuggingFace format
            print(f"Saving dataset to: {save_path}")
            dataset.save_to_disk(save_path)
            print(f"Dataset successfully saved to {save_path}")
        return dataset
        
    except Exception as e:
        print(f"Error loading or saving dataset: {str(e)}")
        return None

def trim_dataset(dataset, max_length):
    """
    Trim dataset to maximum number of examples.
    
    Args:
        dataset: Dataset or DatasetDict to trim
        max_length (int): Maximum number of examples to keep
    
    Returns:
        Trimmed dataset
    """
    from datasets import DatasetDict, Dataset
    
    if isinstance(dataset, DatasetDict):
        # Handle DatasetDict (multiple splits)
        trimmed_dict = {}
        for split_name, split_dataset in dataset.items():
            original_length = len(split_dataset)
            if original_length > max_length:
                trimmed_dict[split_name] = split_dataset.select(range(max_length))
                print(f"Trimmed {split_name} split from {original_length} to {max_length} examples")
            else:
                trimmed_dict[split_name] = split_dataset
                print(f"Kept {split_name} split unchanged ({original_length} examples)")
        return DatasetDict(trimmed_dict)
    
    elif isinstance(dataset, Dataset):
        # Handle single Dataset
        original_length = len(dataset)
        if original_length > max_length:
            trimmed_dataset = dataset.select(range(max_length))
            print(f"Trimmed dataset from {original_length} to {max_length} examples")
            return trimmed_dataset
        else:
            print(f"Dataset unchanged ({original_length} examples)")
            return dataset
    
    return dataset

def save_as_jsonl(dataset, save_path, dataset_name):
    """
    Save dataset as JSONL (JSON Lines) format.
    
    Args:
        dataset: The dataset to save
        save_path (str): Directory path to save the files
        dataset_name (str): Name of the dataset for file naming
    """
    
    if isinstance(dataset, DatasetDict):
        # Handle DatasetDict (multiple splits)
        for split_name, split_dataset in dataset.items():

            print(f"Saving {split_name} split to: {save_path}_{split_name}.jsonl")
            
            with open(f"{save_path}_{split_name}.jsonl", 'w', encoding='utf-8') as f:
                for example in split_dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    elif isinstance(dataset, Dataset):
        # Handle single Dataset
        file_path = os.path.join(save_path, f"{dataset_name}.jsonl")
        
        print(f"Saving dataset to: {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Dataset successfully saved as JSONL format to {save_path}")

def preload_dataset(dataset_id, dataset_name=None, split=None):

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
        dataset_dict = DatasetDict({'train': dataset})

        return dataset_dict

    return load_dataset(dataset_id, dataset_name, split=split)

def create_chat_input(query_prompt, config, add_generation_prompt=True):
    
    # Extract key parts of the config
    chat_template = config["chat_template"]
    eos_token = config["eos_token"]
    
    # Construct messages for template rendering
    # Simulating a simple conversation setup here with roles: system, user, assistant
    messages = [
        {"role": "user", "content": query_prompt}
    ]

    # Define a rendering context for the chat template
    rendering_context = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "eos_token": eos_token
    }

    # Render the chat template using the rendering context
    chat_input = render_template(chat_template, rendering_context)
    return chat_input

def render_template(template_str, context):
    """Render the chat template string with Jinja-style template logic"""
    
    template = Template(template_str)
    return template.render(context)

def move_onnx_model(model_path, destination_dir, delete=False):
    """
    Move or copy ONNX model and its data file to a new destination directory.
    
    Args:
        model_path (str): Path to the .onnx model file
        destination_dir (str): Destination directory path
        delete (bool): If True, move files (delete from source). If False, copy files.
    
    Returns:
        str: Path to the .onnx file in the new location
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Get the model filename
    model_filename = os.path.basename(model_path)
    destination_model_path = os.path.join(destination_dir, model_filename)
    
    # Move or copy the .onnx file
    if os.path.exists(model_path):
        if delete:
            shutil.move(model_path, destination_model_path)
            print(f"✓ Moved {model_filename} to {destination_dir}")
        else:
            shutil.copy2(model_path, destination_model_path)
            print(f"✓ Copied {model_filename} to {destination_dir}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check for and move/copy .onnx.data file
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        data_filename = os.path.basename(data_path)
        destination_data_path = os.path.join(destination_dir, data_filename)
        if delete:
            shutil.move(data_path, destination_data_path)
            print(f"✓ Moved {data_filename} to {destination_dir}")
        else:
            shutil.copy2(data_path, destination_data_path)
            print(f"✓ Copied {data_filename} to {destination_dir}")
    
    return destination_model_path

def move_files_excluding(source_dir, target_dir, exclude_files):
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)

        if os.path.isfile(source_file) and not any(ef in filename for ef in exclude_files):
            shutil.move(source_file, target_file)

def delete_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{directory_path}' does not exist.")

class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.pre_backward_memory = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            # Log GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
            logs["gpu_memory_allocated_MB"] = allocated
            logs["gpu_memory_reserved_MB"] = reserved

            # Log memory usage before backward pass
            if self.pre_backward_memory:
                logs["gpu_memory_allocated_MB_pre_bp"] = self.pre_backward_memory["gpu_memory_allocated_MB_pre_bp"]
        else:
            # Log CPU memory usage using psutil
            mem = psutil.virtual_memory()
            logs["cpu_memory_used_MB"] = mem.used / 1024**2  # Convert to MB
            # Log memory usage before backward pass
            if self.pre_backward_memory:
                logs["cpu_memory_used_MB_pre_bp"] = self.pre_backward_memory["cpu_memory_used_MB_pre_bp"]
    
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            # Log GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            #reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
            self.pre_backward_memory["gpu_memory_allocated_MB_pre_bp"] = allocated
            #self.pre_backward_memory["gpu_memory_reserved_MB_pre_bs"] = reserved
        else:
            # Log CPU memory usage using psutil
            mem = psutil.virtual_memory()
            self.pre_backward_memory["cpu_memory_used_MB_pre_bp"] = mem.used / 1024**2  # Convert to MB
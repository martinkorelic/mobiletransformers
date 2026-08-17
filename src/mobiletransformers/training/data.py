"""Dataset loading/trimming/serialization for training.

Migrated from ``tools/utils.py`` (Migration Map S1). ``datasets`` is imported lazily so importing the
package (and the CLI) stays cheap in the core environment — see ``tests/unit/test_import_weight.py``.
"""

from __future__ import annotations

import json
import os


def load_and_save_dataset(
    dataset_name,
    save_path=None,
    train_file="train_dataset",
    split=None,
    save_format="jsonl",
    max_dataset_length=None,
):
    from datasets import DatasetDict  # noqa: PLC0415

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
    from datasets import Dataset, DatasetDict

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
    from datasets import Dataset, DatasetDict  # noqa: PLC0415

    if isinstance(dataset, DatasetDict):
        # Handle DatasetDict (multiple splits)
        for split_name, split_dataset in dataset.items():
            print(f"Saving {split_name} split to: {save_path}_{split_name}.jsonl")

            with open(f"{save_path}_{split_name}.jsonl", "w", encoding="utf-8") as f:
                for example in split_dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

    elif isinstance(dataset, Dataset):
        # Handle single Dataset
        file_path = os.path.join(save_path, f"{dataset_name}.jsonl")

        print(f"Saving dataset to: {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Dataset successfully saved as JSONL format to {save_path}")


def preload_dataset(dataset_id, dataset_name=None, split=None):
    from datasets import Dataset, DatasetDict, load_dataset  # noqa: PLC0415

    dataset_ids = dataset_id.split("/")

    # Take local data
    if len(dataset_ids) >= 2 and dataset_ids[-2] == "data":
        filepath = dataset_id
        data = None

        if os.path.exists(f"./{dataset_id}.json"):
            filepath = f"./{dataset_id}.json"

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

        elif os.path.exists(f"./{dataset_id}.jsonl"):
            filepath = f"./{dataset_id}.jsonl"

            with open(filepath, encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(data)
        empty_test = dataset.select([])

        # Create a DatasetDict with the "train" split
        dataset_dict = DatasetDict({"train": dataset, "test": empty_test})

        return dataset_dict
    ds = load_dataset(dataset_id, dataset_name, split=split)

    empty_test = ds["train"].select([])

    ds["test"] = empty_test
    return ds

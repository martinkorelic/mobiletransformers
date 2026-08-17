"""Filesystem helpers for moving/removing exported artifacts.

Migrated from ``tools/utils.py`` (Migration Map S1). Deliberately dependency-free — these are used by
the export path, which must stay importable in the core environment.
"""

from __future__ import annotations

import os
import shutil


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

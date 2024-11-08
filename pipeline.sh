#!/bin/bash

# Define paths to the onnxruntime and onnxruntime-training virtual environments
# Two separate virtual env because they cannot be configured in the same one
# The reason why we define two virtual environments and run separate scripts is because sometimes we
# can get package conflict errors with onnxruntime and onnxruntime-training

onnxruntime_train_venv=.ort-venv/
onnxruntime_venv=.venv/

config_path=config.yml

script1=trainer.builder
script2=inference.builder
script3=artifact.onnx_builder
script4=inference.validator

script_args="--config_file $config_path"

# Activate the first virtual environment (if specified)
if [ -n "$onnxruntime_venv" ]; then
    deactivate || true
    echo "[PIPELINE] Activating venv: $onnxruntime_venv"
    source "$onnxruntime_venv/bin/activate"
fi

# Run building scripts

echo "[PIPELINE] Running training model builder..."
python -m "$script1" $script_args

echo "[PIPELINE] Running inference model builder..."
python -m "$script2" $script_args

# Activate the second virtual environment (if specified)
if [ -n "$onnxruntime_train_venv" ]; then
    deactivate || true
    echo "[PIPELINE] Activating venv: $onnxruntime_train_venv"
    source "$onnxruntime_train_venv/bin/activate"
fi

# Run artifact generation scripts for on-device learning and inference
echo "[PIPELINE] Running artifact generation script for on-device learning and inference..."
python -m "$script3" $script_args

# Activate the first virtual environment (if specified)
if [ -n "$onnxruntime_venv" ]; then
    deactivate || true
    echo "[PIPELINE] Activating venv: $onnxruntime_venv"
    source "$onnxruntime_venv/bin/activate"
fi

# Run validator tests
echo "[PIPELINE] Running inference validator script for on-device inference testing..."
python -m "$script4" $script_args
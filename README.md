# On-device LLM framework for Fine-Tuning and Inference

This is a framework for generating ONNX training and inference model from Huggingface / local LLM models and deploying them on-device for training and inference. This framework currently supports:
- PEFT training methods and improved inference methods for text generation task
- Deployment of the models into ready application for on-device LLM **Fine-Tuning** and **Inference / Generation**
- Includes **ONNX GenAI** and **native generation** for on-device inference
- **Ready-to-go Android application** for on-device LLM Fine-Tuning and Generation
- ~~Merging of finetuned adapters into inference model for generation~~

## TODOs

### On-device
- [ ] Create PEFT specific adapter merging methods into inference model
- [ ] Custom tokenizer that is not dependent on ONNX GenAI framework
- [ ] Create implementation that will load and save adapters on-device (using [ONNX Runtime LoraAdapters](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_lora_adapter.html)?)

### Script
- [ ] Add metadata to the training model
- [ ] Add profiling to the inference validator
- [ ] Add other PEFT training methods
- [ ] Validator for on-device training
- [ ] Simulator for on-device training, merging and inference
- [ ] (Android) Script for deployment of the models on-device

### Roadmap
- [ ] (On-device) Add application support for iOS (including CoreML)
- [ ] (On-device) Add optimization methods for RLHF (?)

## Usage

### Pre-requisites
To start using with your own models or other custom models please configure the two files before running the pipeline in offline phase:
- `config.yml` - All configurations related to the model customization and pipeline configurations
- `pipeline.sh` - Configuration and pipeline executing of the scripts
- `.env` - Create your own environment file in the root folder with the following variables:

```ini
# Huggingface token
HF_TOKEN=...

# Path to Huggingface model cache directory (optional)
HF_CACHE=/path/to/.cache/huggingface/hub
```

Refer to configuration file `config.yml` for all descriptions about the configurations. Read about all the separate script usage in the following sections.

### Training model builder

Script that fetches the Huggingface LLM model and converts it into a ONNX graph compatible for artifact training generation.

Usage example (use `--help` to get details about the arguments):
```sh
python -m trainer.builder --config_file config.yml
```

### Inference model builder

Script that fetches the Huggingface LLM model and converts it into a ONNX graph compatible for artifact inference generation.
This uses optimizations prepared by ONNX GenAI framework and exposes the needed adapters for on-device loading of newly updated parameters.

Usage example (use `--help` to get details about the arguments):
```sh
python -m inference.builder --config_file config.yml
```

### Artifact builder

Script that creates the training and inference model artifacts which can be deployed to the device. The models are utilized by the on-device application.

Usage example (use `--help` to get details about the arguments):
```sh
python -m artifact.onnx_builder --config_file config.yml
```

### Inference validator

Script that validates the generation / inference of the inference artifact model.

Usage example (use `--help` to get details about the arguments):
```sh
python -m inference.validator --config_file config.yml
```

### Training validator

Script that validates the training of the training artifcat model. Training with dataset and observing the performance results. 

TODO: to be written...

### On-device simulator

Script that simulates the training of the model, the merging of the model adapters and inference of the newly trained inference model.

TODO: to be written...

## On-device deployment

TODO: to be written...

---

> Any other research made by myself can be found in `docs/ResearchNotes.md`.
>
> Created by Martin Korelič - August 2024
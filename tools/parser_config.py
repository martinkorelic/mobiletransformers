"""
File that stores all the section names of the config.yml file.
"""

ARTIFACT_CONFIG = "ARTIFACT_BUILDER"
ARTIFACT_VALIDATOR_CONFIG = "ARTIFACT_VALIDATOR"
TRAIN_CONFIG = "TRAIN_BUILDER"
INFERENCE_CONFIG = "INFERENCE_BUILDER"
INFERENCE_ARTIFACT_CONFIG = "inference_config"
TEST_GENERATION_CONFIG = "test_generation_config"

# Boolq: google/boolq
# HellaSWAG: Rowan/hellaswag
# ARC: allenai/ai2_arc
# LogiQA: data/logiqa_train

# If "data" is prefix, the data is loaded from local "data/"" directory
TASK_NAME_TO_DATASET = {
    "logiqa": "data/logiqa_train",
    "hellaswag": "Rowan/hellaswag",
    "arc": "allenai/ai2_arc",
    "boolq": "google/boolq"
}
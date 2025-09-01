import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get('HF_TOKEN')

#### EXPERIMENT CONFIG ####

TASK_EPOCHS = {
    "boolq": 2,
    "logiqa": 3,
    "arc_e": 4,
    "winogrande": 4,
    "arc_c": 4,
    "hellaswag": 1,
    "mini_personalqa": 6
}

BATCH_SIZE = 32
PER_DEVICE_BATCH_SIZE = 6
GRADIENT_ACCUMULATION = 2

EXPERIMENT_RANKS = [2, 8, 32]
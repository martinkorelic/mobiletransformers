import os

from evaluation.eval_adapter_onnx_model import CustomPeftONNXModel

from deepeval.benchmarks import ARC
from deepeval.benchmarks.modes import ARCMode

# Disable DeepEval telemetry logging
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

SLM_MODEL_ID = "TinyLlama/TinyLlama_v1.1"
SLM_MODEL_NAME = "quant_model.onnx"
SLM_MODEL_DIR = "build/inference-arc-e"
MERGED_WEIGHTS_DIR = "build/train-arc-e/merged"

# Initialize models
slm_generator = CustomPeftONNXModel(SLM_MODEL_ID, SLM_MODEL_NAME, SLM_MODEL_DIR, load_merged_weights=True, merged_weights_dir=MERGED_WEIGHTS_DIR)

# Define benchmark with specific tasks and shots
benchmark = ARC(
    n_shots=0,
    mode=ARCMode.CHALLENGE,
    verbose_mode=True,
    confinement_instructions=" "
)

slm_generator.set_generation_config(
    max_new_tokens=1
)

results = benchmark.evaluate(model=slm_generator)
print(results)
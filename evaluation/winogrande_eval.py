import json
import os

# Disable DeepEval telemetry logging
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.benchmarks import Winogrande
from evaluation.eval_adapter_models import CustomPeftModel

MODEL_PATH = "experiment_results/TinyLlama_v1.1-lora-winogrande-r2-a2"

custom_llm = CustomPeftModel(adapter_path=MODEL_PATH, adapter_name="lora")

# Define benchmark with specific tasks and shots
benchmark = Winogrande(
    n_shots=2,
    verbose_mode=True,
    confinement_instructions=""
)

custom_llm.set_generation_config(
    max_new_tokens=3
)

results = benchmark.evaluate(model=custom_llm)
print(results)

# Save results to a JSON file
with open(f"{MODEL_PATH}/eval_results.json", "w") as f:
    json.dump({
        "task": "winogrande",
        "accuracy": results
    }, f, indent=4, ensure_ascii=False)
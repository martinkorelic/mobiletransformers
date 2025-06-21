import json
import os

# Disable DeepEval telemetry logging
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask
from evaluation.eval_adapter_models import CustomPeftModel

# results/joint-mars-8-4-12-a-2-shuffled
MODEL_PATH = "results/mars-new-8-hellaswag"

custom_llm = CustomPeftModel(adapter_path=MODEL_PATH, adapter_name="mars", base_model='TinyLlama/TinyLlama_v1.1')

# Define benchmark with specific tasks and shots
benchmark = HellaSwag(
    n_shots=0,
    verbose_mode=True,
    confinement_instructions=" "
)

results = benchmark.evaluate(model=custom_llm)
print(results)

# Save results to a JSON file
#with open("deepeval_results.json", "w") as f:
#    json.dump(results, f, indent=4, ensure_ascii=False)
#print("Evaluation results saved to deepeval_results.json")
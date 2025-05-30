import os

# Disable DeepEval telemetry logging
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.benchmarks import BoolQ
from evaluation.eval_adapter_models import CustomPeftModel

MODEL_PATH = "results/mars-2-boolq-a-1"

custom_llm = CustomPeftModel(adapter_path=MODEL_PATH, adapter_name="mars")

# Define benchmark with specific tasks and shots
benchmark = BoolQ(
    n_shots=0,
    verbose_mode=True,
    confinement_instructions=" "
)

custom_llm.set_generation_config(
    max_new_tokens=1
)

results = benchmark.evaluate(model=custom_llm)
print(results)

# Save results to a JSON file
#with open("deepeval_results.json", "w") as f:
#    json.dump(results, f, indent=4, ensure_ascii=False)
#print("Evaluation results saved to deepeval_results.json")
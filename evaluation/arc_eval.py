import os

# Disable DeepEval telemetry logging
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.benchmarks import ARC
from deepeval.benchmarks.modes import ARCMode
from evaluation.eval_adapter_models import CustomPeftModel

MODEL_PATH = "experiment_results/TinyLlama_v1.1-lora_xs/TinyLlama_v1.1-lora_xs-arc_c-r64-a2"

custom_llm = CustomPeftModel(adapter_path=MODEL_PATH, adapter_name="lora_xs")

# Define benchmark with specific tasks and shots
benchmark = ARC(
    n_shots=0,
    mode=ARCMode.CHALLENGE,
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
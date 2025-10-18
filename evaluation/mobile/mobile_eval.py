from evaluation.mobile_evaluator import MobileEvaluator

from evaluation.eval_adapter_models import CustomPeftModel
from evaluation.eval_adapter_onnx_model import CustomPeftONNXModel

def evaluate_finetuned():

    ADAPTER_DIR = "experiment_results/TinyLlama_v1.1-mars-minipersonalqa/Qwen2-0.5B-mars-mini_personalqa-r8-a2"
    ADAPTER_NAME = "mars"
    EVAL_DATASET = "data/MiniPersonalQA_eval.jsonl"

    model = CustomPeftModel(ADAPTER_DIR, adapter_name=ADAPTER_NAME)

    model.set_generation_config(max_new_tokens=1)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True)

    # Print results
    evaluator.print_results(results)

def evaluate_onnx_mini_personalqa():

    SLM_MODEL_ID = "Qwen/Qwen2-0.5B"
    SLM_MODEL_NAME = "quant_model.onnx"
    SLM_MODEL_DIR = "build/inference-Qwen2-0.5B-f32"

    BASE_MODEL_DIR = "experiment_results/train-qwen2-personalqa"
    MERGED_WEIGHTS_DIR = "experiment_results/train-qwen2-personalqa/merged"

    EVAL_DATASET = "data/MiniPersonalQA_eval.jsonl"

    model = CustomPeftONNXModel(SLM_MODEL_ID, SLM_MODEL_NAME, SLM_MODEL_DIR, load_merged_weights=True, merged_weights_dir=MERGED_WEIGHTS_DIR)

    model.set_generation_config(max_new_tokens=1)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True, save_results_dir=BASE_MODEL_DIR, save_outputs=True)

    # Print results
    evaluator.print_results(results)

def evaluate_onnx_mini_recommendation():

    SLM_MODEL_ID = "Qwen/Qwen2-0.5B"
    SLM_MODEL_NAME = "quant_model.onnx"
    SLM_MODEL_DIR = "build/inference-Qwen2-0.5B-f32"

    MERGED_WEIGHTS_DIR = "experiment_results/train-qwen2-recommendation-mobile/merged"
    EVAL_DATASET = "data/MiniRecommendation_eval.jsonl"
    TASK = "mini_recommendation"

    BASE_MODEL_DIR = "experiment_results/train-qwen2-recommendation-mobile"

    model = CustomPeftONNXModel(SLM_MODEL_ID, SLM_MODEL_NAME, SLM_MODEL_DIR, load_merged_weights=True, merged_weights_dir=MERGED_WEIGHTS_DIR)

    model.set_generation_config(max_new_tokens=128)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model, task=TASK)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True, save_results_dir=BASE_MODEL_DIR, save_outputs=True)

    # Print results
    evaluator.print_results(results)



#evaluate_onnx_mini_personalqa()
#evaluate_onnx_mini_recommendation()
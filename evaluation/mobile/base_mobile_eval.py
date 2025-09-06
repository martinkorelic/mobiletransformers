from evaluation.eval_adapter_models import CustomPeftModel
from evaluation.mobile_evaluator import MobileEvaluator

MINI_PERSONAL_QA_EXAMPLES = [
    {"type": "train", "category": "App Usage", "question": "What specific method is used by news apps to send me updates?", "choices": {"A": "Push notifications", "B": "Sending a letter", "C": "A town crier", "D": "Email newsletters"}, "correct_answer": "A"},
    {"type": "train", "category": "Communication & Social", "question": "The text thread with my old college buddies is always buzzing with new messages. Which of my social circles has a particularly lively group chat?", "choices": {"A": "My coworkers", "B": "My college friends", "C": "My high school acquaintances", "D": "My family"}, "correct_answer": "B"},
    {"type": "train", "category": "Location & Travel", "question": "What type of route do I use for my daily commute to my job?", "choices": {"A": "A scenic bike path", "B": "A local side street", "C": "The highway", "D": "A pedestrian walkway"}, "correct_answer": "C"}
]

MINI_RECOMMENDATION_EXAMPLES = [
    {"type": "train", "category": "Energy Management", "prompt": "I feel really tired this evening", "recommendation": "Recommend early sleep, lower lighting"},
]

def evaluate_base_mini_personalqa():
    EVAL_DATASET = "data/MiniPersonalQA_eval.jsonl"
    BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

    model = CustomPeftModel("", adapter_name="base", base_model=BASE_MODEL)

    model.set_generation_config(max_new_tokens=10)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True, few_shot_examples=MINI_PERSONAL_QA_EXAMPLES, save_results_dir=".", save_outputs=True)

    # Print results
    evaluator.print_results(results)

def evaluate_base_mini_recommendation():
    SLM_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    EVAL_DATASET = "data/MiniRecommendation_eval.jsonl"
    TASK = "mini_recommendation"

    model = CustomPeftModel("", adapter_name="base", base_model=SLM_MODEL_ID)

    model.set_generation_config(max_new_tokens=128)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model, task=TASK)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True, save_outputs=True, few_shot_examples=[])

    # Print results
    evaluator.print_results(results)

evaluate_base_mini_recommendation()
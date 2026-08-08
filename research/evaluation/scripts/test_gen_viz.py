from mobiletransformers.evaluation.eval_adapter_models import CustomPeftModel
from deepeval.benchmarks.arc.template import ARCTemplate

ADAPTER_PATH = "experiment_results/TinyLlama_v1.1-abl_G-loraq8/TinyLlama_v1.1-abl_G-arc_c-r32-a2"
ADAPTER_NAME = "ablation"
BASE_MODEL = "TinyLlama/TinyLlama_v1.1"

test_arc = {
            "id": "Mercury_7220990",
            "question": "Which factor will most likely cause a person to develop a fever?",
            "choices": {
                "text": [
                    "a leg muscle relaxing after exercise",
                    "a bacterial population in the bloodstream",
                    "several viral particles on the skin",
                    "carbohydrates being digested in the stomach",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        }

q = ARCTemplate.format_question(test_arc, include_answer=False) + "\n\n "

# Initialize models
generator = CustomPeftModel(ADAPTER_PATH, adapter_name=ADAPTER_NAME, base_model=BASE_MODEL, empty_init=True)

#generator.setup_attention_viz(max_new_tokens=1)
results = generator.analyze_prediction_probabilities(q, top_k=20)

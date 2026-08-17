SLM_MODEL_ID = "TinyLlama/TinyLlama_v1.1"
SLM_MODEL_NAME = "quant_model.onnx"
SLM_MODEL_DIR = "build/inference-arc-e"
MERGED_WEIGHTS_DIR = "build/train-arc-e/merged"

from mobiletransformers.artifacts.validation import MobileTransformerGenerator
from deepeval.benchmarks.arc.template import ARCTemplate

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
slm_generator = MobileTransformerGenerator(
    model_id=SLM_MODEL_ID,
    model_name=SLM_MODEL_NAME, 
    model_dir=SLM_MODEL_DIR,
    load_merged_weights=True,
    merged_weights_dir=MERGED_WEIGHTS_DIR
)

g = slm_generator.generate(q, decode_between=True)

print(g)
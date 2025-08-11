SLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SLM_MODEL_NAME = "quant_model.onnx"
SLM_MODEL_DIR = "build/inference-Phi-3-mini-4k-instruct-int8-DQ"

from inference.validator import ORTransformerGenerator

# Initialize models
slm_generator = ORTransformerGenerator(
    model_id=SLM_MODEL_ID,
    model_name=SLM_MODEL_NAME, 
    model_dir=SLM_MODEL_DIR
)

g = slm_generator.generate("Hello how are you?")

print(g)
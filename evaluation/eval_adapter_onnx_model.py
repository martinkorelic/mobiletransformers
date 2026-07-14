"""
DEEPEVAL Adapter for loading PEFT ONNX models and performing inference for evaluation.
"""

from transformers import AutoTokenizer
from deepeval.models import DeepEvalBaseLLM
from inference.validator import MobileTransformerGenerator

class CustomPeftONNXModel(DeepEvalBaseLLM):
    def __init__(self, model_id, model_name, model_dir, load_merged_weights=False, merged_weights_dir=None):

        self.model = MobileTransformerGenerator(
            model_id=model_id,
            model_name=model_name, 
            model_dir=model_dir,
            load_merged_weights=load_merged_weights,
            merged_weights_dir=merged_weights_dir
        )

        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Custom generation config
        self.early_stopping = True
        self.max_new_tokens = 3

    def set_generation_config(self, early_stopping=True, max_new_tokens=20):
        """Set generation configuration parameters."""
        self.max_new_tokens = max_new_tokens
    
    def load_model(self):
        """Return the loaded model."""
        return self.model
    
    def generate(self, prompt: str) -> str:
        """
        Generates a response from the model using text-generation pipeline.
        """

        output = self.model.generate(prompt, self.max_new_tokens)

        return output.strip()
    
    async def a_generate(self, prompt: str) -> str:
        """Async version of generate."""
        return self.generate(prompt)
    
    def get_model_name(self):
        """Return the model name."""
        return self.name
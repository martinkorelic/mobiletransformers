from onnxruntime.training.api import CheckpointState
from onnx import numpy_helper

# Load the checkpoint
cp = CheckpointState.load_checkpoint("build/train/checkpoint")

# Access parameter by name
param_name = "backbone.model.layers.9.self_attn.k_proj.base_layer.weight_quantized"
param = cp.parameters[param_name]  # This returns a `Parameter` object

# Now access the NumPy array
array = param.data # This uses the Parameter wrapper around the OrtValue

# Print details
print("Shape:", array.shape)
print("Dtype:", array.dtype)

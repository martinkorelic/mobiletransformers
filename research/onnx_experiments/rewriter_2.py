import onnx
import gc
import os
import onnx
from onnx import numpy_helper
import numpy as np

def convert_dq_q4_initializers_to_inputs(input_model_path, output_model_path):
    """
    Load an ONNX model, convert all initializers with '.weight_DQ_Q4' in their names
    to model inputs, and save the modified model.
    
    Args:
        input_model_path (str): Path to the input ONNX model
        output_model_path (str): Path where the modified model will be saved
        
    Returns:
        int: Number of initializers converted to inputs
    """
    print(f"Loading ONNX model from: {input_model_path}")
    
    # Load the ONNX model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Find initializers to convert
    initializers_to_convert = []
    converted_names = []
    
    for param in graph.initializer:
        if ".weight_DQ_Q4" in param.name:
            initializers_to_convert.append(param)
            converted_names.append(param.name)
            print(f"Marking initializer for conversion: {param.name}")
    
    print(f"Found {len(initializers_to_convert)} initializers to convert")
    
    # Convert initializers to inputs
    for param in initializers_to_convert:
        print(f"Converting initializer: {param.name}")
        
        try:
            # Create a ValueInfoProto for the input
            input_info = onnx.helper.make_tensor_value_info(
                param.name,
                param.data_type,
                param.dims
            )
            
            # Add to graph inputs
            graph.input.append(input_info)
            print(f"  Added input: {param.name} with shape {param.dims}")
            
        except Exception as e:
            print(f"  ✗ Error creating input for {param.name}: {e}")
            continue
    
    # Remove converted initializers
    for param in initializers_to_convert:
        try:
            graph.initializer.remove(param)
            print(f"  Removed initializer: {param.name}")
        except Exception as e:
            print(f"  ✗ Error removing initializer {param.name}: {e}")
    
    # Save the modified model
    print(f"Saving modified model to: {output_model_path}")
    try:
        onnx.save(model, output_model_path, save_as_external_data=True)
        print("✓ Model saved successfully")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return 0
    
    # Verify the model is still valid
    try:
        # Load and check the saved model
        onnx.checker.check_model(output_model_path)
        print("✓ Model validation passed after converting initializers to inputs")
        
    except Exception as e:
        print(f"✗ Warning: Model validation failed: {e}")
        print("This might indicate the model was corrupted during conversion")
    
    # Clean up
    del model
    
    print(f"\n🎉 Conversion complete!")
    print(f"   Successfully converted {len(initializers_to_convert)} initializers to inputs")
    print("   Converted initializers:")
    for name in converted_names:
        print(f"     - {name}")
    
    return len(initializers_to_convert)



def inspect_initializer_weights(model_path, weight_name):
    """
    Load an ONNX model and inspect a specific initializer weight.
    
    Args:
        model_path (str): Path to the ONNX model
        weight_name (str): Name of the initializer to inspect
        
    Returns:
        numpy.ndarray: The weight as a numpy array
    """
    print(f"Loading ONNX model from: {model_path}")
    
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Find the specific initializer
    target_initializer = None
    
    print(f"Searching for initializer: {weight_name}")
    
    for init in graph.initializer:
        if init.name == weight_name:
            target_initializer = init
            print(f"✓ Found initializer: {init.name}")
            break
    
    if target_initializer is None:
        print(f"✗ Initializer '{weight_name}' not found in model")
        print("Available initializers with similar names:")
        for init in graph.initializer:
            if "weight_DQ_Q4" in init.name:
                print(f"  - {init.name}")
        return None
    
    # Convert to numpy array
    print(f"Converting initializer to numpy array...")
    try:
        weight_array = numpy_helper.to_array(target_initializer)
        
        
        # Inspect the numpy array
        print(f"\n📊 Numpy Array Inspection:")
        print(f"   Shape: {weight_array.shape}")
        print(f"   Data type: {weight_array.dtype}")
        print(f"   Size: {weight_array.size} elements")
        print(f"   Memory size: {weight_array.nbytes} bytes")
        
        # Check value ranges
        print(f"\n📈 Value Statistics:")
        print(f"   Min value: {weight_array.min()}")
        print(f"   Max value: {weight_array.max()}")
        print(f"   Mean value: {weight_array.mean():.4f}")
        print(f"   Unique values: {len(np.unique(weight_array))}")
        
        # Show first few values
        print(f"\n🔍 First 10 values:")
        print(f"   {weight_array.flatten()[:10]}")
        
        # If it's int4 packed, show how to unpack
        if weight_array.dtype == np.uint8:
            print(f"\n🔧 Int4 Unpacking Analysis:")
            print(f"   If this is packed int4 (Int4x2), each byte contains 2 values")
            print(f"   Logical shape would be: {weight_array.shape[:-1] + (weight_array.shape[-1] * 2,)}")
            
            # Show how to unpack first few bytes
            first_bytes = weight_array.flatten()[:5]
            print(f"   First 5 bytes: {first_bytes}")
            print(f"   Unpacked int4 values:")
            for i, byte_val in enumerate(first_bytes):
                low_4bit = byte_val & 0xF  # Lower 4 bits
                high_4bit = (byte_val >> 4) & 0xF  # Upper 4 bits
                print(f"     Byte {i}: {byte_val} -> [{low_4bit}, {high_4bit}]")
        
        return weight_array
        
    except Exception as e:
        print(f"✗ Error converting to numpy array: {e}")
        return None

# Example usage:
# Replace 'your_model.onnx' with your actual model path
model_path = "build/train_models/quant_model.onnx"  # Change this to your model path
weight_name = "backbone.model.layers.0.self_attn.v_proj.base_layer.weight_DQ_Q4"

# Inspect the weight
weight_array = inspect_initializer_weights(model_path, weight_name)

if weight_array is not None:
    print(f"\n✅ Successfully loaded weight array!")
    print(f"Use this array for your inputs with shape: {weight_array.shape}")
    print(f"Data type: {weight_array.dtype}")
else:
    print(f"\n❌ Failed to load weight array")




# Example usage
#if __name__ == "__main__":
#    # Example usage
#    input_path = "build/train_models/quant_model.onnx"
#    output_path = "model_cleaned.onnx"
#    
#    if os.path.exists(input_path):
#        converted_count = convert_dq_q4_initializers_to_inputs(input_path, output_path)
#        print(f"\nDone! Converted {converted_count} initializers to inputs.")
#    else:
#        print(f"Input model file not found: {input_path}")
#        print("Please update the input_path variable with the correct path to your ONNX model.")
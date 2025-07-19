import onnx
import os

from onnx.external_data_helper import write_external_data_tensors

def check_quantization_parameters(model_path, external_data_path=None):
    """
    Check and print weight_scale and weight_zero_point values from an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        external_data_path: Path to external data file (if applicable)
    
    Returns:
        dict: Dictionary containing found quantization parameters
    """
    import numpy as np
    
    # Load the model
    if external_data_path:
        # If external data exists, load with external data
        model = onnx.load(model_path, load_external_data=True)
    else:
        model = onnx.load(model_path)
    
    quant_params = {}
    
    print("=== Quantization Parameters Check ===")
    
    for initializer in model.graph.initializer:
        # Check if this is a quantization parameter tensor
        is_weight_scale = initializer.name.endswith('weight_scale')
        is_weight_zero_point = initializer.name.endswith('weight_zero_point')
        
        if is_weight_scale or is_weight_zero_point:
            print(f"\nTensor: {initializer.name}")
            print(f"  Data Type: {onnx.TensorProto.DataType.Name(initializer.data_type)}")
            print(f"  Shape: {[dim for dim in initializer.dims]}")
            print(f"  Data Location: {'EXTERNAL' if initializer.data_location == onnx.TensorProto.EXTERNAL else 'INTERNAL'}")
            
            # Extract the actual values
            try:
                if initializer.data_location == onnx.TensorProto.EXTERNAL:
                    # For external data, we need to read from external file
                    if len(initializer.external_data) > 0:
                        external_file = None
                        offset = 0
                        length = None
                        
                        for entry in initializer.external_data:
                            if entry.key == "location":
                                external_file = entry.value
                            elif entry.key == "offset":
                                offset = int(entry.value)
                            elif entry.key == "length":
                                length = int(entry.value)
                        
                        if external_file:
                            # Try to read from external file
                            external_full_path = os.path.join(os.path.dirname(model_path), external_file)
                            if os.path.exists(external_full_path):
                                with open(external_full_path, 'rb') as f:
                                    f.seek(offset)
                                    if length:
                                        raw_data = f.read(length)
                                    else:
                                        raw_data = f.read()
                                
                                # Convert raw data to numpy array
                                tensor_data = onnx.numpy_helper.to_array(
                                    onnx.TensorProto(
                                        dims=initializer.dims,
                                        data_type=initializer.data_type,
                                        raw_data=raw_data
                                    )
                                )
                            else:
                                print(f"  ⚠️  External file not found: {external_full_path}")
                                tensor_data = None
                        else:
                            print(f"  ⚠️  No external file location found")
                            tensor_data = None
                    else:
                        print(f"  ⚠️  External data location specified but no external_data entries")
                        tensor_data = None
                else:
                    # For internal data, convert directly
                    tensor_data = onnx.numpy_helper.to_array(initializer)
                
                if tensor_data is not None:
                    print(f"  Values:")
                    if tensor_data.size <= 10:  # Print all values if small
                        print(f"    {tensor_data.flatten()}")
                    else:  # Print first/last few values if large
                        flat_data = tensor_data.flatten()
                        print(f"    First 5: {flat_data[:5]}")
                        print(f"    Last 5:  {flat_data[-5:]}")
                        print(f"    Shape: {tensor_data.shape}")
                    
                    print(f"  Statistics:")
                    print(f"    Min: {np.min(tensor_data):.6f}")
                    print(f"    Max: {np.max(tensor_data):.6f}")
                    print(f"    Mean: {np.mean(tensor_data):.6f}")
                    print(f"    Std: {np.std(tensor_data):.6f}")
                    
                    # Store in results
                    quant_params[initializer.name] = {
                        'data': tensor_data,
                        'shape': tensor_data.shape,
                        'dtype': tensor_data.dtype,
                        'min': np.min(tensor_data),
                        'max': np.max(tensor_data),
                        'mean': np.mean(tensor_data),
                        'std': np.std(tensor_data)
                    }
                else:
                    print(f"  ❌ Could not read tensor data")
                    
            except Exception as e:
                print(f"  ❌ Error reading tensor data: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Found {len(quant_params)} quantization parameter tensors")
    
    return quant_params

def check_quantization_parameters_simple(model_path):
    """
    Simplified version that only works with internal data (no external data support).
    """
    import numpy as np
    
    model = onnx.load(model_path, load_external_data=False)
    
    print("=== Quantization Parameters Check (Internal Data Only) ===")
    
    found_count = 0
    for initializer in model.graph.initializer:
        is_weight_scale = initializer.name.endswith('weight_scale')
        is_weight_zero_point = initializer.name.endswith('weight_zero_point')
        
        if is_weight_scale or is_weight_zero_point:
            found_count += 1
            print(f"\nTensor: {initializer.name}")
            
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                print(f"  ⚠️  This tensor uses external data - use check_quantization_parameters() instead")
            else:
                try:
                    tensor_data = onnx.numpy_helper.to_array(initializer)
                    print(f"  Shape: {tensor_data.shape}")
                    print(f"  Data Type: {tensor_data.dtype}")
                    
                    if tensor_data.size <= 10:
                        print(f"  Values: {tensor_data.flatten()}")
                    else:
                        flat_data = tensor_data.flatten()
                        print(f"  First 5: {flat_data[:5]}")
                        print(f"  Last 5:  {flat_data[-5:]}")
                    
                    print(f"  Min: {np.min(tensor_data):.6f}, Max: {np.max(tensor_data):.6f}")
                    print(f"  Mean: {np.mean(tensor_data):.6f}, Std: {np.std(tensor_data):.6f}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
    
    print(f"\nFound {found_count} quantization parameter tensors")

# Example usage:
if __name__ == "__main__":
    
    # Now check the quantization parameters in the new model
    print("\n" + "="*50)
    print("CHECKING QUANTIZATION PARAMETERS IN NEW MODEL")
    print("="*50)
    
    quant_params = check_quantization_parameters(
        "build/inference_models/quant_model.onnx"
    )
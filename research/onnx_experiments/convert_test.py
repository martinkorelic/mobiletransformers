
import onnx
import os

def force_dequantize_external_and_save(model, output_path, external_data_filename=None):
    """
    Force DequantizeLinear x_scale and x_zero_point tensors to be external and save the model.
    
    Args:
        model: Loaded ONNX model (onnx.ModelProto)
        output_path: Path where to save the modified model
        external_data_filename: Name of external data file (optional, defaults to model_name.onnx.data)
    
    Returns:
        int: Number of tensors that were forced to external
    """
    if external_data_filename is None:
        model_name = os.path.splitext(os.path.basename(output_path))[0]
        external_data_filename = f'{model_name}.onnx.data'
    
    # First, convert all tensors to external data
    onnx.external_data_helper.convert_model_to_external_data(
        model, 
        location=external_data_filename, 
        size_threshold=0
    )
    
    forced_count = 0
    
    # Force DequantizeLinear tensors to external manually
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            # Check if initializer name ends with x_scale or x_zero_point
            is_dequant_tensor = (initializer.name.endswith('weight_scale') or 
                               initializer.name.endswith('weight_zero_point'))
            
            if is_dequant_tensor:
                print(f"Forcing DequantizeLinear tensor to external: {initializer.name}")
                initializer.data_location = onnx.TensorProto.EXTERNAL
                
                # Clear and set external data
                del initializer.external_data[:]
                
                location_entry = onnx.StringStringEntryProto()
                location_entry.key = "location"
                location_entry.value = external_data_filename
                initializer.external_data.append(location_entry)
                
                # Clear internal data - check if field exists before clearing
                if initializer.HasField("raw_data"):
                    initializer.ClearField("raw_data")
                
                # Clear repeated fields (they don't use HasField)
                # Only clear fields that exist in this ONNX version
                data_fields = [
                    "int32_data", "int64_data", "float_data", "double_data", 
                    "uint64_data", "string_data"
                ]
                
                # Add complex data fields if they exist in this ONNX version
                if hasattr(initializer, 'complex64_data'):
                    data_fields.append("complex64_data")
                if hasattr(initializer, 'complex128_data'):
                    data_fields.append("complex128_data")
                
                # Clear all existing data fields
                for field_name in data_fields:
                    if hasattr(initializer, field_name):
                        field_value = getattr(initializer, field_name)
                        if hasattr(field_value, '__len__') and len(field_value) > 0:
                            initializer.ClearField(field_name)
                
                forced_count += 1
    
    return model

# Example usage:
if __name__ == "__main__":
    # Load your model
    model = onnx.load("build/inference/quant_model.onnx")
    
    # Force DequantizeLinear tensors to external and save
    forced_count = force_dequantize_external_and_save(
        model, 
        "build/inference/ext_quant_model.onnx.onnx"
    )
    
    print(f"Done! {forced_count} tensors were forced to external.")
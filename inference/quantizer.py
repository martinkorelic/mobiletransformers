import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import numpy as np
import math

# Mapping of data type names to ONNX TensorProto types
DTYPE_MAP = {
    'int8': TensorProto.INT8,
    'uint8': TensorProto.UINT8,
    'int4': TensorProto.INT4,
    'uint4': TensorProto.UINT4,
    'int16': TensorProto.INT16,
    'float8e4m3fn': TensorProto.FLOAT8E4M3FN,
    'float8e4m3fnuz': TensorProto.FLOAT8E4M3FNUZ,
    'float8e5m2': TensorProto.FLOAT8E5M2,
    'float8e5m2fnuz': TensorProto.FLOAT8E5M2FNUZ,
}

# Quantization ranges for different data types
QUANT_RANGES = {
    'int8': (-128, 127),
    'uint8': (0, 255),
    #'int4': (-8, 7),
    #'uint4': (0, 15),
    'int16': (-32768, 32767),
    # Float8 types use their natural floating-point ranges
    #'float8e4m3fn': (-448.0, 448.0),      # Approximate range for E4M3FN
    #'float8e4m3fnuz': (0.0, 448.0),       # Approximate range for E4M3FNUZ (no negative zero)
    #'float8e5m2': (-57344.0, 57344.0),    # Approximate range for E5M2
    #'float8e5m2fnuz': (0.0, 57344.0),     # Approximate range for E5M2FNUZ (no negative zero)
}


def get_supported_dtypes():
    """Return list of supported quantization data types."""
    return list(DTYPE_MAP.keys())


def is_integer_dtype(dtype):
    """Check if the data type is an integer type."""
    return dtype in ['int8', 'uint8', 'int4', 'uint4', 'int16']


def is_float8_dtype(dtype):
    """Check if the data type is a float8 type."""
    return dtype.startswith('float8')


def calculate_blocked_scale_shape(weight_shape, axis, block_size):
    """Calculate the shape of the scale tensor for blocked quantization."""
    scale_shape = list(weight_shape)
    # For blocked quantization, the scale shape is identical to input shape
    # except for the blocking dimension where it's ceil(Di/B)
    scale_shape[axis] = math.ceil(weight_shape[axis] / block_size)
    return scale_shape


def create_quantization_utility_graph(weight_shape, axis=0, block_size=128, dtype='int8'):
    """Create a utility graph that quantizes float weights to specified dtype."""
    # Calculate scale shape for blocked quantization
    scale_shape = calculate_blocked_scale_shape(weight_shape, axis, block_size)
    
    # Define tensor info
    weight_input = helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape)
    scale_input = helper.make_tensor_value_info("scale", TensorProto.FLOAT, scale_shape)
    
    # Get ONNX tensor type for the specified dtype
    onnx_dtype = DTYPE_MAP[dtype]
    quantized_output = helper.make_tensor_value_info("quantized_weight", onnx_dtype, weight_shape)
    
    # Create QuantizeLinear node with explicit output_dtype
    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["weight", "scale"],
        outputs=["quantized_weight"],
        axis=axis,
        block_size=block_size,
        output_dtype=onnx_dtype,
        name="quantize_weight"
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[quantize_node],
        name="quantization_utility",
        inputs=[weight_input, scale_input],
        outputs=[quantized_output]
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return model


def compute_quantization_params(weight_data, axis=0, block_size=128, dtype='int8'):
    """Compute scale for blocked quantization."""
    weight_shape = weight_data.shape
    scale_shape = calculate_blocked_scale_shape(weight_shape, axis, block_size)
    
    # Initialize scale array
    scale = np.zeros(scale_shape, dtype=np.float32)
    
    # Get quantization range for the specified dtype
    if dtype not in QUANT_RANGES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    qmin, qmax = QUANT_RANGES[dtype]
    
    # Get the dimension size along the quantization axis
    axis_size = weight_shape[axis]
    
    # Compute scale for each block
    for block_idx in range(scale_shape[axis]):
        # Calculate the range for this block
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, axis_size)
        
        # Create slice objects for all dimensions
        slices = [slice(None)] * len(weight_shape)
        slices[axis] = slice(start_idx, end_idx)
        
        # Extract the block data
        block_data = weight_data[tuple(slices)]
        
        # Compute min and max for this block
        block_min = np.min(block_data)
        block_max = np.max(block_data)
        
        # Compute scale based on dtype
        range_val = block_max - block_min
        if range_val == 0:
            block_scale = 1.0
        else:
            if is_integer_dtype(dtype):
                # For integer types, scale based on the quantization range
                quant_range = qmax - qmin
                block_scale = range_val / quant_range
            elif is_float8_dtype(dtype):
                # For float8 types, we need to map to the float8 range
                # This is a simplified approach - in practice, you might want more sophisticated scaling
                float8_range = qmax - qmin
                block_scale = range_val / float8_range
            else:
                # Default case
                block_scale = range_val / 255.0
        
        # Store the computed scale
        scale_slices = [slice(None)] * len(scale_shape)
        scale_slices[axis] = slice(block_idx, block_idx + 1)
        
        scale[tuple(scale_slices)] = block_scale
    
    return scale


def quantize_weight_with_utility_graph(weight_data, axis=0, block_size=128, dtype='int8'):
    """Use utility graph to quantize weight data."""
    # Compute quantization parameters
    scale = compute_quantization_params(weight_data, axis, block_size, dtype)
    
    # Create utility graph
    util_model = create_quantization_utility_graph(weight_data.shape, axis, block_size, dtype)
    
    # Create inference session
    session = ort.InferenceSession(util_model.SerializeToString())
    
    # Run quantization
    quantized_weight = session.run(
        ["quantized_weight"],
        {
            "weight": weight_data.astype(np.float32),
            "scale": scale
        }
    )[0]
    
    return quantized_weight, scale


def find_weight_transpose_nodes(model):
    """Find all nodes that have 'weight_transpose' in their name and are Transpose operations."""
    weight_transpose_nodes = []
    
    for node in model.graph.node:
        if node.op_type == "Transpose" and "weight_transpose" in node.name:
            weight_transpose_nodes.append(node)
    
    return weight_transpose_nodes


def find_weight_initializers_for_transpose_nodes(model, transpose_nodes):
    """Find weight initializers that are inputs to the transpose nodes."""
    weight_initializers = []
    
    for node in transpose_nodes:
        weight_name = node.input[0]  # First input should be the weight
        
        # Find the corresponding initializer
        for initializer in model.graph.initializer:
            if initializer.name == weight_name:
                weight_initializers.append((initializer, node))
                break
    
    return weight_initializers


def insert_dequantize_nodes_and_quantize_weights(model, axis=0, block_size=128, dtype='int8'):
    """Insert DequantizeLinear nodes before Transpose nodes and quantize weights."""
    # Validate dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported types: {get_supported_dtypes()}")
    
    # Find transpose nodes with weight_transpose in name
    transpose_nodes = find_weight_transpose_nodes(model)
    
    if not transpose_nodes:
        print("No transpose nodes with 'weight_transpose' in name found.")
        return model
    
    print(f"Found {len(transpose_nodes)} transpose nodes to process.")
    
    # Find corresponding weight initializers
    weight_initializers = find_weight_initializers_for_transpose_nodes(model, transpose_nodes)
    
    if not weight_initializers:
        print("No weight initializers found for transpose nodes.")
        return model
    
    print(f"Found {len(weight_initializers)} weight initializers to quantize to {dtype}.")
    
    # Prepare new initializers and nodes
    new_initializers = list(model.graph.initializer)
    new_nodes = []
    
    # Process each weight initializer
    for initializer, transpose_node in weight_initializers:
        weight_name = initializer.name
        weight_data = numpy_helper.to_array(initializer)
        
        print(f"Processing weight: {weight_name} with shape {weight_data.shape}")
        
        # Quantize the weight
        quantized_weight, scale = quantize_weight_with_utility_graph(
            weight_data, axis, block_size, dtype
        )
        
        # Create new initializer names
        quantized_weight_name = f"{weight_name}_quantized"
        scale_name = f"{weight_name}_scale"
        dequantized_weight_name = f"{weight_name}_dequantized"
        
        # Create new initializers
        quantized_initializer = numpy_helper.from_array(quantized_weight, quantized_weight_name)
        scale_initializer = numpy_helper.from_array(scale, scale_name)
        
        new_initializers.extend([quantized_initializer, scale_initializer])
        
        # Create DequantizeLinear node
        dequantize_node = helper.make_node(
            "DequantizeLinear",
            inputs=[quantized_weight_name, scale_name],
            outputs=[dequantized_weight_name],
            axis=axis,
            block_size=block_size,
            name=f"dequantize_{weight_name}"
        )
        
        new_nodes.append(dequantize_node)
        
        # Update the transpose node to use dequantized weight
        updated_transpose_inputs = [dequantized_weight_name] + list(transpose_node.input[1:])
        updated_transpose_node = helper.make_node(
            transpose_node.op_type,
            inputs=updated_transpose_inputs,
            outputs=list(transpose_node.output),
            name=transpose_node.name
        )
        
        # Copy attributes from original transpose node
        for attr in transpose_node.attribute:
            updated_transpose_node.attribute.append(attr)
        
        new_nodes.append(updated_transpose_node)
        
        print(f"Added DequantizeLinear node for {weight_name}")
    
    # Add all other nodes (excluding the original transpose nodes we're replacing)
    transpose_node_names = {node.name for _, node in weight_initializers}
    for node in model.graph.node:
        if node.name not in transpose_node_names:
            new_nodes.append(node)
    
    # Remove original weight initializers that we've quantized
    quantized_weight_names = {init.name for init, _ in weight_initializers}
    new_initializers = [init for init in new_initializers if init.name not in quantized_weight_names]
    
    # Create new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=list(model.graph.input),
        outputs=list(model.graph.output),
        initializer=new_initializers
    )
    
    # Create new model
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    
    return new_model


def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add DequantizeLinear nodes and quantize weights")
    parser.add_argument("input_model", help="Path to input ONNX model")
    parser.add_argument("output_model", help="Path to output ONNX model")
    parser.add_argument("--axis", type=int, default=0, help="Quantization axis (default: 0)")
    parser.add_argument("--block_size", type=int, default=128, help="Block size (default: 128)")
    parser.add_argument("--dtype", type=str, default="int8", 
                        choices=get_supported_dtypes(),
                        help=f"Quantization data type (default: int8). Supported types: {', '.join(get_supported_dtypes())}")
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.input_model}")
    model = onnx.load(args.input_model)
    
    # Process the model
    print(f"Processing model with dtype: {args.dtype}")
    processed_model = insert_dequantize_nodes_and_quantize_weights(
        model, axis=args.axis, block_size=args.block_size, dtype=args.dtype
    )
    
    # Check the processed model
    #print("Checking processed model...")
    #onnx.checker.check_model(processed_model)
    
    # Save the processed model
    print(f"Saving processed model to {args.output_model}")
    onnx.save(processed_model, args.output_model)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()


# Example usage:
# python quantize_weights.py input_model.onnx output_model.onnx --dtype int8 --axis 0 --block_size 128
# python quantize_weights.py input_model.onnx output_model.onnx --dtype float8e4m3fn --axis 0 --block_size 128
# python quantize_weights.py input_model.onnx output_model.onnx --dtype uint4 --axis 0 --block_size 128
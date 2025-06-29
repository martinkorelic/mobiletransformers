import onnx
import numpy as np
from onnx import helper, TensorProto
import onnxruntime as ort

def create_onnx_chunking_model(
    output_path="onnx_chunking_model.onnx"
):
    """
    Creates an ONNX model that performs only the chunking of the intermediate matrix.

    Inputs:
    - intermediate: float32 [N*rank_adapter, shared_rank]
    - adapter_index: int64 scalar
    - rank_adapter: int64 scalar
    - shared_rank_val: int64 scalar (explicitly passed for model creation/testing)

    Outputs:
    - chunked_intermediate: float32 [rank_adapter, shared_rank]
    """

    # Inputs for the chunking model
    inputs = [
        helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, ["N_times_rank_adapter", "shared_rank_val_dim"]),
        helper.make_tensor_value_info("adapter_index", TensorProto.INT64, []),
        helper.make_tensor_value_info("rank_adapter", TensorProto.INT64, []),
        helper.make_tensor_value_info("shared_rank_val", TensorProto.INT64, []) # Added as explicit input
    ]

    # Output for the chunking model
    outputs = [
        helper.make_tensor_value_info("chunked_intermediate", TensorProto.FLOAT, ["rank_adapter_dim", "shared_rank_val_dim"])
    ]

    nodes = []

    # Constants
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["zero_scalar"],
            value=helper.make_tensor("zero_scalar_tensor", TensorProto.INT64, [], [0])
        )
    )
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one_scalar"],
            value=helper.make_tensor("one_scalar_tensor", TensorProto.INT64, [1], [1]) # For Reshape
        )
    )
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes_slice"],
            value=helper.make_tensor("axes_slice_tensor", TensorProto.INT64, [2], [0, 1])
        )
    )

    # Compute slice_start: adapter_index * rank_adapter
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["adapter_index", "rank_adapter"],
            outputs=["slice_start_val"],
            name="compute_slice_start"
        )
    )

    # Compute slice_end: slice_start + rank_adapter
    nodes.append(
        helper.make_node(
            "Add",
            inputs=["slice_start_val", "rank_adapter"],
            outputs=["slice_end_val"],
            name="compute_slice_end"
        )
    )

    # Reshape scalar slice_start_val and slice_end_val to [1] for Concat
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["slice_start_val", "one_scalar"],
            outputs=["slice_start_reshaped"],
            name="reshape_slice_start"
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["slice_end_val", "one_scalar"],
            outputs=["slice_end_reshaped"],
            name="reshape_slice_end"
        )
    )

    # Reshape zero_scalar to [1] for concat (to match slice_start_reshaped)
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["zero_scalar", "one_scalar"],
            outputs=["zero_scalar_reshaped"],
            name="reshape_zero_scalar_for_concat"
        )
    )

    # Reshape shared_rank_val (from input) to [1] for concat
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["shared_rank_val", "one_scalar"],
            outputs=["shared_rank_val_reshaped"],
            name="reshape_shared_rank_val"
        )
    )


    # slice_starts for the Slice op: [slice_start, 0]
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=["slice_start_reshaped", "zero_scalar_reshaped"],
            outputs=["slice_starts"],
            axis=0,
            name="concat_slice_starts"
        )
    )

    # slice_ends for the Slice op: [slice_end, shared_rank_val]
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=["slice_end_reshaped", "shared_rank_val_reshaped"],
            outputs=["slice_ends"],
            axis=0,
            name="concat_slice_ends"
        )
    )

    # Slice the intermediate matrix
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["intermediate", "slice_starts", "slice_ends", "axes_slice"],
            outputs=["chunked_intermediate"],
            name="slice_chunk"
        )
    )
    # chunked_intermediate shape: [rank_adapter, shared_rank]

    graph = helper.make_graph(
        nodes=nodes,
        name="Chunking_Model",
        inputs=inputs,
        outputs=outputs,
        doc_string="Model for testing intermediate matrix chunking"
    )

    model = helper.make_model(
        graph,
        producer_name="Chunking_Model",
        opset_imports=[helper.make_opsetid("", 11)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Chunking model saved to {output_path}")
    return model

def test_onnx_chunking_model(model_path="onnx_chunking_model.onnx"):
    """Tests the ONNX chunking model with sample data."""
    session = ort.InferenceSession(model_path)

    test_configs = [
        {"shared_rank": 32, "rank_adapter": 8, "N": 3, "adapter_index": 0}, # First chunk
        {"shared_rank": 32, "rank_adapter": 8, "N": 3, "adapter_index": 1}, # Second chunk
        {"shared_rank": 64, "rank_adapter": 16, "N": 3, "adapter_index": 2}, # Third chunk
    ]

    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config}")
        shared_rank = config["shared_rank"]
        rank_adapter = config["rank_adapter"]
        N = config["N"]
        adapter_index = config["adapter_index"]

        # intermediate: [N * rank_adapter, shared_rank]
        intermediate_np = np.random.rand(N * rank_adapter, shared_rank).astype(np.float32)
        adapter_index_np = np.array(adapter_index, dtype=np.int64)
        rank_adapter_np = np.array(rank_adapter, dtype=np.int64)
        shared_rank_val_np = np.array(shared_rank, dtype=np.int64) # Explicitly pass shared_rank

        inputs = {
            "intermediate": intermediate_np,
            "adapter_index": adapter_index_np,
            "rank_adapter": rank_adapter_np,
            "shared_rank_val": shared_rank_val_np
        }

        print(f"  Input 'intermediate' shape: {intermediate_np.shape}")
        print(f"  Input 'adapter_index': {adapter_index_np}")
        print(f"  Input 'rank_adapter': {rank_adapter_np}")
        print(f"  Input 'shared_rank_val': {shared_rank_val_np}")

        # Reference calculation in NumPy
        expected_chunk = intermediate_np[
            adapter_index * rank_adapter : (adapter_index + 1) * rank_adapter,
            :
        ]
        print(f"  NumPy Expected chunk shape: {expected_chunk.shape}")

        try:
            onnx_outputs = session.run(None, inputs)
            onnx_chunked_intermediate = onnx_outputs[0]
            print(f"  ONNX Actual chunk shape: {onnx_chunked_intermediate.shape}")

            # Verify shapes match
            if onnx_chunked_intermediate.shape == expected_chunk.shape:
                print(f"  ✓ Shape match!")
            else:
                print(f"  ✗ Shape MISMATCH! Expected {expected_chunk.shape}, got {onnx_chunked_intermediate.shape}")

            # Verify content (allow for small float differences)
            max_diff = np.max(np.abs(onnx_chunked_intermediate - expected_chunk))
            print(f"  Max content difference: {max_diff}")
            if max_diff < 1e-5: # Small tolerance for floating point
                print(f"  ✓ Content match (within tolerance)!")
            else:
                print(f"  ✗ Content MISMATCH!")

        except ort.OrtValue as e:
            print(f"ONNX Runtime error during session.run: {e}")
            print("This indicates an issue with the ONNX graph's slicing logic.")
            return


def create_lora_merger_model(
    output_path="lora_merger_dynamic.onnx"
):
    """
    Creates an ONNX model that merges LoRA weights with quantized base weights using dynamic shapes.
    
    Input:
    - weight_quantized: uint8 quantized base weights [out_features, in_features]
    - x_zero_point: uint8 scalar zero point for base weights
    - x_scale: float scalar scale for base weights  
    - lora_A: float32 LoRA A matrix [rank, in_features]
    - lora_B: float32 LoRA B matrix [out_features, rank]
    
    Output:
    - merged_weight_quantized: uint8 quantized merged weights [out_features, in_features]
    - merged_zero_point: uint8 scalar zero point
    - merged_scale: float scalar scale
    
    All dimensions are dynamic and inferred at runtime.
    """
    
    # Define input tensors with dynamic shapes
    inputs = [
        helper.make_tensor_value_info(
            "weight_quantized", 
            TensorProto.UINT8, 
            ["out_features", "in_features"]  # Dynamic dimensions
        ),
        helper.make_tensor_value_info(
            "x_zero_point", 
            TensorProto.UINT8, 
            []  # scalar
        ),
        helper.make_tensor_value_info(
            "x_scale", 
            TensorProto.FLOAT, 
            []  # scalar
        ),
        helper.make_tensor_value_info(
            "lora_A", 
            TensorProto.FLOAT, 
            ["rank", "in_features"]  # Dynamic dimensions
        ),
        helper.make_tensor_value_info(
            "lora_B", 
            TensorProto.FLOAT, 
            ["out_features", "rank"]  # Dynamic dimensions
        ),
        helper.make_tensor_value_info(
            "lora_alpha", 
            TensorProto.FLOAT, 
            []  # scalar
        )
    ]
    
    # Define output tensors with dynamic shapes
    outputs = [
        helper.make_tensor_value_info(
            "merged_weight_quantized", 
            TensorProto.UINT8, 
            ["out_features", "in_features"]  # Dynamic dimensions
        ),
        helper.make_tensor_value_info(
            "merged_zero_point", 
            TensorProto.UINT8, 
            []
        ),
        helper.make_tensor_value_info(
            "merged_scale", 
            TensorProto.FLOAT, 
            []
        )
    ]
    
    # Create the computation graph
    nodes = []
    
    # Step 1: Dequantize the base weights
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            inputs=["weight_quantized", "x_scale", "x_zero_point"],
            outputs=["base_weight_fp32"],
            name="dequantize_base_weights"
        )
    )
    
    # Step 2: Compute LoRA delta = lora_B @ lora_A
    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["lora_B", "lora_A"],
            outputs=["lora_delta"],
            name="compute_lora_delta"
        )
    )
    
    # Step 2b: Scale the LoRA delta by alpha
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["lora_delta", "lora_alpha"],
            outputs=["scaled_lora_delta"],
            name="scale_lora_delta"
        )
    )
    
    # Step 3: Add scaled LoRA delta to base weights
    nodes.append(
        helper.make_node(
            "Add",
            inputs=["base_weight_fp32", "scaled_lora_delta"],
            outputs=["merged_weight_fp32"],
            name="add_lora_delta"
        )
    )
    
    # Step 4: Dynamically quantize the merged weights
    nodes.append(
        helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["merged_weight_fp32"],
            outputs=["merged_weight_quantized", "merged_scale", "merged_zero_point"],
            name="quantize_merged_weights"
        )
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=nodes,
        name="LoRAMerger",
        inputs=inputs,
        outputs=outputs,
        doc_string="Merges LoRA weights with quantized base weights and outputs quantized result with alpha scaling"
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name="LoRAMerger",
        opset_imports=[helper.make_opsetid("", 11)]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name="LoRAMerger",
        opset_imports=[helper.make_opsetid("", 11)]
    )
    
    # Check the model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, output_path)
    print(f"LoRA merger model saved to: {output_path}")
    
    return model

def test_lora_merger_model(model_path="lora_merger_dynamic.onnx"):
    """Test the LoRA merger model with sample data using various dynamic shapes."""
    
    # Load the model
    session = ort.InferenceSession(model_path)
    
    # Test with different shapes to verify dynamic behavior
    test_configs = [
        {"out_features": 512, "in_features": 256, "lora_rank": 8},
        {"out_features": 1024, "in_features": 1024, "lora_rank": 16},
        {"out_features": 2048, "in_features": 512, "lora_rank": 32},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config}")
        
        out_features = config["out_features"]
        in_features = config["in_features"]
        lora_rank = config["lora_rank"]
        
        # Generate test quantized base weights
        base_weight_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        
        # Quantize base weights manually for testing
        scale = np.float32((base_weight_fp32.max() - base_weight_fp32.min()) / 255.0)
        zero_point = np.uint8(np.clip(np.round(-base_weight_fp32.min() / scale), 0, 255)).astype(np.uint8)
        weight_quantized = np.clip(np.round(base_weight_fp32 / scale) + zero_point, 0, 255).astype(np.uint8)
        
        # Generate LoRA weights
        lora_A = np.random.randn(lora_rank, in_features).astype(np.float32) * 0.01
        lora_B = np.random.randn(out_features, lora_rank).astype(np.float32) * 0.01
        
        # Generate LoRA alpha (for example, fixed value 16)
        lora_alpha = np.float32(16.0)
        
        # Prepare inputs
        inputs = {
            "weight_quantized": weight_quantized,
            "x_zero_point": np.array(zero_point, dtype=np.uint8),
            "x_scale": np.array(scale, dtype=np.float32),
            "lora_A": lora_A,
            "lora_B": lora_B,
            "lora_alpha": np.array(lora_alpha, dtype=np.float32)
        }
        
        # Run inference
        outputs = session.run(['merged_weight_quantized', 'merged_scale', 'merged_zero_point'], inputs)
        merged_weight_quantized, merged_scale, merged_zero_point = outputs
        
        print(f"  Input shape: {weight_quantized.shape}")
        print(f"  LoRA A shape: {lora_A.shape}")
        print(f"  LoRA B shape: {lora_B.shape}")
        print(f"  Output shape: {merged_weight_quantized.shape}")
        print(f"  Input scale: {scale:.6f}, zero_point: {zero_point}")
        print(f"  Output scale: {merged_scale:.6f}, zero_point: {merged_zero_point}")
        print(f"  LoRA alpha: {lora_alpha}")
        
        # Verify the computation manually
        base_dequantized = (weight_quantized.astype(np.float32) - zero_point) * scale
        lora_delta = np.matmul(lora_B, lora_A) * lora_alpha   # apply alpha!
        expected_merged = base_dequantized + lora_delta
        
        # Dequantize the output to compare
        output_dequantized = (merged_weight_quantized.astype(np.float32) - merged_zero_point) * merged_scale
        
        # Check if results are close (allowing for quantization error)
        max_diff = np.max(np.abs(expected_merged - output_dequantized))
        print(f"  Maximum difference: {max_diff:.6f}")
        
        if max_diff < 0.01:  # Allow small quantization errors
            print(f"  ✓ Test {i+1} passed!")
        else:
            print(f"  ✗ Test {i+1} failed!")
    
    return outputs

def test_chunking_debug(model_path="chunking_debug.onnx"):
    """Test the chunking debug model."""
    
    session = ort.InferenceSession(model_path)
    
    # Test parameters
    N = 3
    rank = 8
    shared_rank = 4
    adapter_index = 1
    
    # Create test data
    intermediate = np.random.randn(N * rank, shared_rank).astype(np.float32)
    
    print(f"Input intermediate shape: {intermediate.shape}")
    print(f"N={N}, rank={rank}, shared_rank={shared_rank}, adapter_index={adapter_index}")
    
    inputs = {
        "intermediate": intermediate,
        "adapter_index": np.array(adapter_index, dtype=np.int64),
        "rank": np.array(rank, dtype=np.int64)
    }
    
    outputs = session.run(None, inputs)
    chunked_intermediate, slice_start, slice_end = outputs
    
    print(f"Slice start: {slice_start}, Slice end: {slice_end}")
    print(f"Chunked intermediate shape: {chunked_intermediate.shape}")
    
    # Verify with numpy
    expected_start = adapter_index * rank
    expected_end = (adapter_index + 1) * rank
    expected_chunk = intermediate[expected_start:expected_end, :]
    
    print(f"Expected slice start: {expected_start}, Expected slice end: {expected_end}")
    print(f"Expected chunk shape: {expected_chunk.shape}")
    
    # Compare
    if np.allclose(chunked_intermediate, expected_chunk):
        print("✓ Chunking test passed!")
    else:
        print("✗ Chunking test failed!")
        print(f"Max difference: {np.max(np.abs(chunked_intermediate - expected_chunk))}")
    
    return chunked_intermediate

def create_chunking_debug_model(output_path="chunking_debug.onnx"):
    """
    Creates a debug model that only does the chunking part to verify shapes.
    
    Inputs:
    - intermediate: float32 [N*rank, shared_rank]
    - adapter_index: int64 scalar
    - rank: int64 scalar
    
    Outputs:
    - chunked_intermediate: float32 [rank, shared_rank]
    - slice_start: int64 scalar
    - slice_end: int64 scalar
    """

    inputs = [
        helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, ["n_times_rank", "shared_rank"]),
        helper.make_tensor_value_info("adapter_index", TensorProto.INT64, []),
        helper.make_tensor_value_info("rank", TensorProto.INT64, [])
    ]

    outputs = [
        helper.make_tensor_value_info("chunked_intermediate", TensorProto.FLOAT, ["rank", "shared_rank"]),
        helper.make_tensor_value_info("slice_start", TensorProto.INT64, []),
        helper.make_tensor_value_info("slice_end", TensorProto.INT64, [])
    ]

    nodes = []

    # Step 1: Calculate slice boundaries
    # slice_start = adapter_index * rank
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["adapter_index", "rank"],
            outputs=["slice_start"],
            name="compute_slice_start"
        )
    )

    # slice_end = slice_start + rank = (adapter_index + 1) * rank
    nodes.append(
        helper.make_node(
            "Add",
            inputs=["slice_start", "rank"],
            outputs=["slice_end"],
            name="compute_slice_end"
        )
    )

    # Step 2: Prepare slice parameters as 1-D arrays
    # We need to convert scalars to 1-D arrays for the Slice operation
    
    # Convert slice_start and slice_end to 1-D arrays
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["slice_start"],
            outputs=["slice_start_1d"],
            axes=[0],
            name="unsqueeze_slice_start"
        )
    )
    
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["slice_end"],
            outputs=["slice_end_1d"],
            axes=[0],
            name="unsqueeze_slice_end"
        )
    )
    
    # Create axes array [0] to specify we're slicing along dimension 0
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes_0"],
            value=helper.make_tensor("axes_0_tensor", TensorProto.INT64, [1], [0])
        )
    )

    # Step 3: Slice the intermediate matrix
    # intermediate[slice_start:slice_end, :] -> [rank, shared_rank]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["intermediate", "slice_start_1d", "slice_end_1d", "axes_0"],
            outputs=["chunked_intermediate"],
            name="slice_intermediate"
        )
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=nodes,
        name="Chunking_Debug",
        inputs=inputs,
        outputs=outputs,
        doc_string="Debug model for chunking intermediate matrix"
    )

    # Create the model
    model = helper.make_model(
        graph,
        producer_name="Chunking_Debug",
        opset_imports=[helper.make_opsetid("", 11)]
    )

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Chunking debug model saved to {output_path}")
    return model

def create_mars_merger_model(output_path="mars_merger_fixed.onnx"):
    """
    Creates a fixed ONNX model that merges PEFT (MARS) weights with quantized base weights.
    
    MARS (Multi-Adapter Routing with Shared components) is a parameter-efficient fine-tuning method
    that decomposes adapter weights into shared and adapter-specific components.
    
    The mathematical operation performed is:
    merged_weight = base_weight + alpha * (adapter_B @ intermediate_chunk @ shared_A)
    
    Where:
    - base_weight: Original model weights (quantized)
    - adapter_B: Adapter-specific "B" matrix [out_features, rank]
    - intermediate: Shared intermediate matrix [N*rank, shared_rank] containing chunks for N adapters
    - intermediate_chunk: Slice of intermediate for current adapter [rank, shared_rank]
    - shared_A: Shared "A" matrix [shared_rank, in_features]
    - alpha: Scaling factor for the adapter contribution
    
    The matrix multiplication chain:
    1. adapter_B [out_features, rank] @ intermediate_chunk [rank, shared_rank] 
       → [out_features, shared_rank]
    2. result @ shared_A [shared_rank, in_features] 
       → [out_features, in_features]
    3. Scale by alpha and add to base weights
    
    Args:
        output_path (str): Path where the ONNX model will be saved
        
    Returns:
        onnx.ModelProto: The created ONNX model
        
    Input tensors:
        - weight_quantized: Base model weights in UINT8 quantized format
        - x_zero_point: Zero point for base weight quantization (UINT8 scalar)
        - x_scale: Scale factor for base weight quantization (FLOAT scalar)
        - shared_A: Shared component matrix (FLOAT [shared_rank, in_features])
        - intermediate: Combined intermediate matrix for all adapters (FLOAT [N*rank, shared_rank])
        - adapter_B: Adapter-specific B matrix (FLOAT [out_features, rank])
        - adapter_index: Index of the adapter to use (INT64 scalar)
        - rank: Rank of the adapter (INT64 scalar)
        - alpha: Scaling factor for adapter contribution (FLOAT scalar)
        
    Output tensors:
        - merged_weight_quantized: Final merged weights in UINT8 quantized format
        - merged_zero_point: Zero point for merged weight quantization (UINT8 scalar)
        - merged_scale: Scale factor for merged weight quantization (FLOAT scalar)
    """
    

    inputs = [
        helper.make_tensor_value_info("weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]),
        helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []),
        helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("shared_A", TensorProto.FLOAT, ["shared_rank", "in_features"]),
        helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, ["n_times_rank", "shared_rank"]),
        helper.make_tensor_value_info("adapter_B", TensorProto.FLOAT, ["out_features", "rank"]),
        helper.make_tensor_value_info("adapter_index", TensorProto.INT64, []),
        helper.make_tensor_value_info("rank", TensorProto.INT64, []),
        helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [])
    ]

    outputs = [
        helper.make_tensor_value_info("merged_weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]),
        helper.make_tensor_value_info("merged_zero_point", TensorProto.UINT8, []),
        helper.make_tensor_value_info("merged_scale", TensorProto.FLOAT, [])
    ]

    nodes = []

    # Step 1: Dequantize base weights
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            inputs=["weight_quantized", "x_scale", "x_zero_point"],
            outputs=["base_weight_fp32"],
            name="dequantize_base_weights"
        )
    )

    # Step 2: Calculate slice boundaries for chunking intermediate matrix
    # slice_start = adapter_index * rank
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["adapter_index", "rank"],
            outputs=["slice_start"],
            name="compute_slice_start"
        )
    )

    # slice_end = slice_start + rank = (adapter_index + 1) * rank
    nodes.append(
        helper.make_node(
            "Add",
            inputs=["slice_start", "rank"],
            outputs=["slice_end"],
            name="compute_slice_end"
        )
    )

    # Step 2: Prepare slice parameters as 1-D arrays
    # We need to convert scalars to 1-D arrays for the Slice operation
    
    # Convert slice_start and slice_end to 1-D arrays
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["slice_start"],
            outputs=["slice_start_1d"],
            axes=[0],
            name="unsqueeze_slice_start"
        )
    )
    
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["slice_end"],
            outputs=["slice_end_1d"],
            axes=[0],
            name="unsqueeze_slice_end"
        )
    )
    
    # Create axes array [0] to specify we're slicing along dimension 0
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes_0"],
            value=helper.make_tensor("axes_0_tensor", TensorProto.INT64, [1], [0])
        )
    )

    # Step 3: Slice the intermediate matrix to get the chunk
    # intermediate[adapter_index*rank:(adapter_index+1)*rank, :] -> [rank, shared_rank]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["intermediate", "slice_start_1d", "slice_end_1d", "axes_0"],
            outputs=["chunked_intermediate"],
            name="slice_intermediate"
        )
    )

    # Step 4: Matrix multiplications following the correct order:
    # adapter_B[out_features, rank] @ chunked_intermediate[rank, shared_rank] @ shared_A[shared_rank, in_features]
    
    # First multiply: adapter_B @ chunked_intermediate
    # [out_features, rank] @ [rank, shared_rank] = [out_features, shared_rank]
    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["adapter_B", "chunked_intermediate"],
            outputs=["adapter_times_chunk"],
            name="adapter_chunk_matmul"
        )
    )

    # Then multiply result @ shared_A
    # [out_features, shared_rank] @ [shared_rank, in_features] = [out_features, in_features]
    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["adapter_times_chunk", "shared_A"],
            outputs=["lora_delta_prealpha"],
            name="final_matmul"
        )
    )

    # Step 5: Apply alpha scaling
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["lora_delta_prealpha", "alpha"],
            outputs=["lora_delta"],
            name="scale_alpha"
        )
    )

    # Step 6: Add LoRA delta to base weights
    nodes.append(
        helper.make_node(
            "Add",
            inputs=["base_weight_fp32", "lora_delta"],
            outputs=["merged_weight_fp32"],
            name="add_delta"
        )
    )

    # Step 7: Quantize the merged weights
    nodes.append(
        helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["merged_weight_fp32"],
            outputs=["merged_weight_quantized", "merged_scale", "merged_zero_point"],
            name="quantize_merged"
        )
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=nodes,
        name="MARS Merger",
        inputs=inputs,
        outputs=outputs,
        doc_string="MARS merger model for merging adapters."
    )

    # Create the model with detailed metadata
    model = helper.make_model(
        graph,
        producer_name="MARS_Merger_v1.0",
        producer_version="1.0.0",         
        doc_string="MARS (Multi-Adapter Rank Sharing) weight merger for PEFT quantized models. Merges adapter weights with quantized base weights using shared decomposition.",
        model_version=1,
        domain="com.martinkorelic.mars",
        opset_imports=[helper.make_opsetid("", 11)]
    )

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"MARS merger model saved to {output_path}")
    return model

def test_mars_merger_model(model_path="mars_merger_fixed.onnx"):
    """Test the fixed Mars merger model with sample data."""

    session = ort.InferenceSession(model_path)

    test_configs = [
        {"out_features": 512, "in_features": 256, "shared_rank": 4, "rank": 8, "N": 3},
        {"out_features": 1024, "in_features": 512, "shared_rank": 8, "rank": 16, "N": 3},
        {"out_features": 256, "in_features": 128, "shared_rank": 2, "rank": 4, "N": 5},
    ]

    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config}")
        of = config["out_features"]
        inf = config["in_features"]
        shared_rank = config["shared_rank"]
        rank = config["rank"]
        N = config["N"]

        # Generate base weight and quantize it with better precision
        base_weight_fp32 = np.random.randn(of, inf).astype(np.float32) * 0.1
        
        # Improved quantization with proper scale and zero point calculation
        weight_min = float(base_weight_fp32.min())
        weight_max = float(base_weight_fp32.max())
        
        # Ensure we have a reasonable range
        if weight_max - weight_min == 0:
            weight_max = weight_min + 1e-6
            
        scale = np.float32((weight_max - weight_min) / 255.0)
        zero_point = np.uint8(np.clip(np.round(-weight_min / scale), 0, 255))
        
        # Quantize
        weight_quantized = np.clip(np.round(base_weight_fp32 / scale) + zero_point, 0, 255).astype(np.uint8)
        
        # Verify quantization round-trip
        dequantized_check = (weight_quantized.astype(np.float32) - zero_point) * scale
        quant_error = np.max(np.abs(base_weight_fp32 - dequantized_check))
        
        print(f"  Quantization check - Max error: {quant_error:.6f}")
        print(f"  Scale: {scale:.6f}, Zero point: {zero_point}")

        # Generate MARS components
        shared_A = np.random.randn(shared_rank, inf).astype(np.float32) * 0.01
        intermediate = np.random.randn(N * rank, shared_rank).astype(np.float32) * 0.01
        adapter_B = np.random.randn(of, rank).astype(np.float32) * 0.01
        alpha = np.float32(16.0)
        adapter_index = np.int64(1)  # Test with the second chunk

        print(f"  Input shapes:")
        print(f"    shared_A: {shared_A.shape}")
        print(f"    intermediate: {intermediate.shape}")
        print(f"    adapter_B: {adapter_B.shape}")
        print(f"    adapter_index: {adapter_index}, rank: {rank}")

        # Prepare inputs
        inputs = {
            "weight_quantized": weight_quantized,
            "x_zero_point": np.array(zero_point, dtype=np.uint8),
            "x_scale": np.array(scale, dtype=np.float32),
            "shared_A": shared_A,
            "intermediate": intermediate,
            "adapter_B": adapter_B,
            "alpha": np.array(alpha, dtype=np.float32),
            "adapter_index": np.array(adapter_index, dtype=np.int64),
            "rank": np.array(rank, dtype=np.int64),
        }

        # Run the model
        try:
            outputs = session.run(None, inputs)
            
            # Extract outputs in the correct order as defined in the ONNX model
            # DynamicQuantizeLinear outputs: [quantized_tensor, scale, zero_point]
            merged_weight_quantized = outputs[0]
            merged_zero_point = outputs[1]
            merged_scale = outputs[2]
            
            # Convert to scalar values if they're single-element arrays
            if isinstance(merged_scale, np.ndarray) and merged_scale.size == 1:
                merged_scale_val = merged_scale.item()
            else:
                merged_scale_val = merged_scale
                
            if isinstance(merged_zero_point, np.ndarray) and merged_zero_point.size == 1:
                merged_zero_point_val = merged_zero_point.item()
            else:
                merged_zero_point_val = merged_zero_point

            # Compute reference result
            base_dequantized = (weight_quantized.astype(np.float32) - zero_point) * scale
            
            # Compute reference following the correct order:
            # adapter_B @ intermediate_chunk @ shared_A
            chunk_start = adapter_index * rank
            chunk_end = (adapter_index + 1) * rank
            chunked_intermediate = intermediate[chunk_start:chunk_end, :]  # [rank, shared_rank]
            
            print(f"    chunked_intermediate shape: {chunked_intermediate.shape}")
            
            # Matrix multiplication chain:
            # adapter_B @ chunked_intermediate = [out_features, rank] @ [rank, shared_rank] = [out_features, shared_rank]
            temp = adapter_B @ chunked_intermediate  # [out_features, shared_rank]
            print(f"    temp (adapter_B @ chunked_intermediate) shape: {temp.shape}")
            
            # temp @ shared_A = [out_features, shared_rank] @ [shared_rank, in_features] = [out_features, in_features]
            lora_delta = temp @ shared_A
            print(f"    lora_delta shape: {lora_delta.shape}")
            
            lora_delta *= alpha
            
            expected_merged = base_dequantized + lora_delta
            
            # Compare with model output (dequantize model output for comparison)
            model_dequantized = (merged_weight_quantized.astype(np.float32) - merged_zero_point_val) * merged_scale_val
            
            # Debug info
            print(f"  Expected merged range: [{expected_merged.min():.6f}, {expected_merged.max():.6f}]")
            print(f"  Model merged range: [{model_dequantized.min():.6f}, {model_dequantized.max():.6f}]")
            print(f"  Model quantization scale: {merged_scale_val:.6f}, zero_point: {merged_zero_point_val}")
            
            max_diff = np.max(np.abs(expected_merged - model_dequantized))
            relative_error = max_diff / (np.max(np.abs(expected_merged)) + 1e-8)
            
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Relative error: {relative_error:.6f}")
            
            # More lenient tolerance for quantization errors
            if relative_error < 0.1:  # 10% relative error tolerance due to quantization
                print(f"  ✓ Test {i+1} passed!")
            else:
                print(f"  ✗ Test {i+1} failed!")
                print(f"    Expected shape: {expected_merged.shape}")
                print(f"    Model output shape: {model_dequantized.shape}")
                
                # Let's also test without the final quantization step
                # by comparing the float32 merged weights before quantization
                print(f"  Debugging: Testing intermediate float32 values...")
                
        except Exception as e:
            print(f"  ✗ Test {i+1} failed with error: {e}")

    return outputs if 'outputs' in locals() else None

if __name__ == "__main__":

    # Create the basic LoRA merger model
    print("Creating basic LoRA merger model with dynamic shapes...")
    basic_model = create_mars_merger_model(
        output_path="mars_merger.onnx"
    )
    
    # Test the basic model
    print("\nTesting basic MARS merger model...")
    test_results = test_mars_merger_model("mars_merger.onnx")
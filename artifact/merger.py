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


def create_lora_merger_model(output_path="lora_merger.onnx", quantized=True):
    """
    Creates an ONNX LoRA merger model:
    - quantized == True:
      merges LoRA weights with quantized base weights
    - quantized == False:
      merges LoRA weights with float32 base weights
    Inputs:
        - weight (uint8 if quantized else float32)
        - scale (float32) [scalar]  (only if quantized)
        - zero_point (uint8) [scalar] (only if quantized)
        - lora_A (float32)
        - lora_B (float32)
        - alpha (float32) [scalar]
    Outputs:
        - merged_weight (uint8 if quantized else float32)
        - scale (float32) [scalar]  (only if quantized)
        - zero_point (uint8) [scalar] (only if quantized)
    """
    import onnx
    from onnx import helper, TensorProto

    inputs = []
    outputs = []
    nodes = []

    if quantized:
        # quantized inputs
        inputs.append(helper.make_tensor_value_info("weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]))
        inputs.append(helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []))
    else:
        # float base weights
        inputs.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, ["out_features", "in_features"]))

    # LoRA inputs
    inputs.append(helper.make_tensor_value_info("adapter_A", TensorProto.FLOAT, ["rank", "in_features"]))
    inputs.append(helper.make_tensor_value_info("adapter_B", TensorProto.FLOAT, ["out_features", "rank"]))
    inputs.append(helper.make_tensor_value_info("alpha", TensorProto.FLOAT, []))

    if quantized:
        outputs.append(helper.make_tensor_value_info("merged_weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]))
        outputs.append(helper.make_tensor_value_info("merged_scale", TensorProto.FLOAT, []))
        outputs.append(helper.make_tensor_value_info("merged_zero_point", TensorProto.UINT8, []))
    else:
        outputs.append(helper.make_tensor_value_info("merged_weight", TensorProto.FLOAT, ["out_features", "in_features"]))

    # 1. Dequantize if needed
    if quantized:
        nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=["weight_quantized", "x_scale", "x_zero_point"],
            outputs=["base_weight_fp32"],
            name="dequantize_base_weights"
        ))
        base_input = "base_weight_fp32"
    else:
        base_input = "weight"

    # 2. LoRA delta
    nodes.append(helper.make_node(
        "MatMul",
        inputs=["adapter_B", "adapter_A"],
        outputs=["lora_delta"],
        name="compute_lora_delta"
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=["lora_delta", "alpha"],
        outputs=["scaled_lora_delta"],
        name="scale_lora_delta"
    ))

    # 3. Add
    nodes.append(helper.make_node(
        "Add",
        inputs=[base_input, "scaled_lora_delta"],
        outputs=["merged_weight_fp32"],
        name="add_lora_delta"
    ))

    # 4. Requantize if needed
    if quantized:
        nodes.append(helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["merged_weight_fp32"],
            outputs=["merged_weight_quantized", "merged_scale", "merged_zero_point"],
            name="quantize_merged"
        ))
    else:
        # output is float, rename directly
        nodes.append(helper.make_node(
            "Identity",
            inputs=["merged_weight_fp32"],
            outputs=["merged_weight"],
            name="identity_output"
        ))

    graph = helper.make_graph(
        nodes=nodes,
        name="LoRAMergerModel",
        inputs=inputs,
        outputs=outputs
    )

    model = helper.make_model(graph, producer_name="LoRAMerger", opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✅ LoRA merger model saved to {output_path} with quantized={quantized}")
    return model

def test_all_lora_merger_models(
    quantized_model_path="lora_merger_quantized.onnx",
    float_model_path="lora_merger_float.onnx"
):
    """
    Tests both the quantized and float LoRA merger models on the same data.
    """

    # load both models
    quantized_session = ort.InferenceSession(quantized_model_path)
    float_session = ort.InferenceSession(float_model_path)

    test_configs = [
        {"out_features": 512, "in_features": 256, "lora_rank": 8},
        {"out_features": 1024, "in_features": 1024, "lora_rank": 16},
    ]

    for i, config in enumerate(test_configs):
        out_features = config["out_features"]
        in_features = config["in_features"]
        rank = config["lora_rank"]

        print(f"\n======================")
        print(f"[TEST CONFIG] {config}")
        print(f"======================")

        # Create a random float base weight
        base_weight_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.1

        # Quantize it for the quantized model
        scale = np.float32((base_weight_fp32.max() - base_weight_fp32.min()) / 255.0)
        zero_point = np.uint8(np.clip(np.round(-base_weight_fp32.min() / scale), 0, 255))
        weight_quantized = np.clip(np.round(base_weight_fp32 / scale) + zero_point, 0, 255).astype(np.uint8)

        # LoRA weights
        lora_A = np.random.randn(rank, in_features).astype(np.float32) * 0.01
        lora_B = np.random.randn(out_features, rank).astype(np.float32) * 0.01
        alpha = np.float32(16.0)

        # shared
        expected = base_weight_fp32 + alpha * np.matmul(lora_B, lora_A)

        # ---------------
        # QUANTIZED MODEL
        # ---------------
        quant_inputs = {
            "weight_quantized": weight_quantized,
            "x_scale": np.array(scale, dtype=np.float32),
            "x_zero_point": np.array(zero_point, dtype=np.uint8),
            "adapter_A": lora_A,
            "adapter_B": lora_B,
            "alpha": np.array(alpha, dtype=np.float32),
        }
        q_outputs = quantized_session.run(
            ["merged_weight_quantized", "merged_scale", "merged_zero_point"],
            quant_inputs
        )
        merged_q, merged_q_scale, merged_q_zero = q_outputs

        merged_q_dequant = (merged_q.astype(np.float32) - merged_q_zero) * merged_q_scale

        diff_q = np.max(np.abs(merged_q_dequant - expected))
        print(f"[QUANTIZED]")
        print(f"  merged_q shape: {merged_q.shape}")
        print(f"  max diff vs expected: {diff_q:.6f}")
        if diff_q < 0.01:
            print("  ✅ PASS")
        else:
            print("  ✗ FAIL")

        # ---------------
        # FLOAT MODEL
        # ---------------
        float_inputs = {
            "weight": base_weight_fp32,
            "adapter_A": lora_A,
            "adapter_B": lora_B,
            "alpha": np.array(alpha, dtype=np.float32),
        }
        f_outputs = float_session.run(["merged_weight"], float_inputs)
        merged_f = f_outputs[0]
        diff_f = np.max(np.abs(merged_f - expected))
        print(f"[FLOAT]")
        print(f"  merged_f shape: {merged_f.shape}")
        print(f"  max diff vs expected: {diff_f:.6f}")
        if diff_f < 1e-6:
            print("  ✅ PASS")
        else:
            print("  ✗ FAIL")

def create_mars_merger_model(output_path="mars_merger_model.onnx", quantized=True):
    """
    Creates a fixed ONNX model that merges PEFT (MARS) weights with quantized base weights.
    
    MARS (Multi Adapter Rank Sharing) is a parameter-efficient fine-tuning method
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
    
# Dynamic inputs depending on quantized or not
    if quantized:
        inputs = [
            helper.make_tensor_value_info("weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]),
            helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []),
            helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []),
        ]
        outputs = [
            helper.make_tensor_value_info("merged_weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]),
            helper.make_tensor_value_info("merged_zero_point", TensorProto.UINT8, []),
            helper.make_tensor_value_info("merged_scale", TensorProto.FLOAT, [])
        ]
    else:
        inputs = [
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, ["out_features", "in_features"]),
        ]
        outputs = [
            helper.make_tensor_value_info("merged_weight", TensorProto.FLOAT, ["out_features", "in_features"]),
        ]

    # common inputs regardless of quantized or not
    common_inputs = [
        helper.make_tensor_value_info("shared_A", TensorProto.FLOAT, ["shared_rank", "in_features"]),
        helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, ["n_times_rank", "shared_rank"]),
        helper.make_tensor_value_info("adapter_B", TensorProto.FLOAT, ["out_features", "rank"]),
        helper.make_tensor_value_info("adapter_index", TensorProto.INT64, []),
        helper.make_tensor_value_info("rank", TensorProto.INT64, []),
        helper.make_tensor_value_info("alpha", TensorProto.FLOAT, []),
    ]
    inputs.extend(common_inputs)

    nodes = []

    # Step 1: get base_weight_fp32
    if quantized:
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                inputs=["weight_quantized", "x_scale", "x_zero_point"],
                outputs=["base_weight_fp32"],
                name="dequantize_base_weights"
            )
        )
    else:
        # just rename directly
        nodes.append(
            helper.make_node(
                "Identity",
                inputs=["weight"],
                outputs=["base_weight_fp32"],
                name="pass_through_base_weight"
            )
        )

    # Step 2: calculate slice boundaries
    nodes.append(
        helper.make_node("Mul", ["adapter_index", "rank"], ["slice_start"], name="compute_slice_start")
    )
    nodes.append(
        helper.make_node("Add", ["slice_start", "rank"], ["slice_end"], name="compute_slice_end")
    )
    nodes.append(
        helper.make_node("Unsqueeze", ["slice_start"], ["slice_start_1d"], axes=[0], name="unsqueeze_slice_start")
    )
    nodes.append(
        helper.make_node("Unsqueeze", ["slice_end"], ["slice_end_1d"], axes=[0], name="unsqueeze_slice_end")
    )
    nodes.append(
        helper.make_node(
            "Constant", [], ["axes_0"],
            value=helper.make_tensor("axes_0_tensor", TensorProto.INT64, [1], [0])
        )
    )
    nodes.append(
        helper.make_node(
            "Slice", ["intermediate", "slice_start_1d", "slice_end_1d", "axes_0"],
            ["chunked_intermediate"], name="slice_intermediate"
        )
    )

    # Step 3: adapter_B @ chunked_intermediate
    nodes.append(
        helper.make_node("MatMul", ["adapter_B", "chunked_intermediate"], ["adapter_times_chunk"], name="adapter_chunk_matmul")
    )

    # Step 4: (adapter_B @ chunked) @ shared_A
    nodes.append(
        helper.make_node("MatMul", ["adapter_times_chunk", "shared_A"], ["lora_delta_prealpha"], name="final_matmul")
    )

    # Step 5: alpha scaling
    nodes.append(
        helper.make_node("Mul", ["lora_delta_prealpha", "alpha"], ["lora_delta"], name="scale_alpha")
    )

    # Step 6: add LoRA delta
    nodes.append(
        helper.make_node("Add", ["base_weight_fp32", "lora_delta"], ["merged_weight_fp32"], name="add_delta")
    )

    # Step 7: quantization if requested
    if quantized:
        nodes.append(
            helper.make_node(
                "DynamicQuantizeLinear",
                inputs=["merged_weight_fp32"],
                outputs=["merged_weight_quantized", "merged_scale", "merged_zero_point"],
                name="quantize_merged"
            )
        )
    else:
        # just output the floating-point
        nodes.append(
            helper.make_node(
                "Identity",
                inputs=["merged_weight_fp32"],
                outputs=["merged_weight"],
                name="identity_output"
            )
        )

    # put together the graph
    graph = helper.make_graph(
        nodes=nodes,
        name="MARS Merger",
        inputs=inputs,
        outputs=outputs,
        doc_string="MARS merger model for merging adapters (quantized or float)."
    )

    model = helper.make_model(
        graph,
        producer_name="MARS_Merger_v1.0",
        producer_version="1.0.0",
        doc_string="MARS (Multi-Adapter Rank Sharing) weight merger for PEFT models.",
        model_version=1,
        domain="com.martinkorelic.mars",
        opset_imports=[helper.make_opsetid("", 11)]
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    if quantized:
        print(f"Quantized MARS merger model saved to {output_path}")
    else:
        print(f"MARS merger model saved to {output_path}")
    return model

def test_mars_merger_model(
    quantized_model_path="mars_merger_fixed.onnx",
    nonquantized_model_path="mars_merger_nonquant.onnx"
):
    """
    Test both the quantized and non-quantized Mars merger models with sample data.
    """

    q_session = ort.InferenceSession(quantized_model_path)
    nq_session = ort.InferenceSession(nonquantized_model_path)

    test_configs = [
        {"out_features": 512, "in_features": 256, "shared_rank": 4, "rank": 8, "N": 3},
        {"out_features": 256, "in_features": 128, "shared_rank": 2, "rank": 4, "N": 5},
    ]

    for i, config in enumerate(test_configs):
        print(f"\n=== Test {i+1}: {config} ===")

        of = config["out_features"]
        inf = config["in_features"]
        shared_rank = config["shared_rank"]
        rank = config["rank"]
        N = config["N"]

        # Generate base weight and quantize it with better precision
        base_weight_fp32 = np.random.randn(of, inf).astype(np.float32) * 0.1

        # --- QUANTIZED TEST ---
        print("\n[ Quantized model test ]")

        weight_min = float(base_weight_fp32.min())
        weight_max = float(base_weight_fp32.max())
        if weight_max - weight_min == 0:
            weight_max = weight_min + 1e-6

        scale = np.float32((weight_max - weight_min) / 255.0)
        zero_point = np.uint8(np.clip(np.round(-weight_min / scale), 0, 255))

        weight_quantized = np.clip(np.round(base_weight_fp32 / scale) + zero_point, 0, 255).astype(np.uint8)

        inputs_q = {
            "weight_quantized": weight_quantized,
            "x_zero_point": np.array(zero_point, dtype=np.uint8),
            "x_scale": np.array(scale, dtype=np.float32),
            "shared_A": np.random.randn(shared_rank, inf).astype(np.float32) * 0.01,
            "intermediate": np.random.randn(N * rank, shared_rank).astype(np.float32) * 0.01,
            "adapter_B": np.random.randn(of, rank).astype(np.float32) * 0.01,
            "alpha": np.array(16.0, dtype=np.float32),
            "adapter_index": np.array(1, dtype=np.int64),
            "rank": np.array(rank, dtype=np.int64)
        }

        # reuse
        shared_A = inputs_q["shared_A"]
        intermediate = inputs_q["intermediate"]
        adapter_B = inputs_q["adapter_B"]
        alpha = inputs_q["alpha"]
        adapter_index = inputs_q["adapter_index"]

        try:
            outputs_q = q_session.run(None, inputs_q)
            merged_weight_quantized, merged_scale, merged_zero_point = outputs_q[0], outputs_q[2], outputs_q[1]

            # dequantize
            merged_dequantized = (merged_weight_quantized.astype(np.float32) - merged_zero_point) * merged_scale

            # reference
            chunk_start = adapter_index * rank
            chunk_end = (adapter_index + 1) * rank
            chunked_intermediate = intermediate[chunk_start:chunk_end, :]
            delta = adapter_B @ chunked_intermediate @ shared_A * alpha
            expected = base_weight_fp32 + delta

            max_diff = np.max(np.abs(expected - merged_dequantized))
            rel_error = max_diff / (np.max(np.abs(expected)) + 1e-8)

            print(f"  Quantized max diff: {max_diff:.6f}, relative error: {rel_error:.6f}")
            if rel_error < 0.1:
                print("  ✓ Quantized test passed")
            else:
                print("  ✗ Quantized test failed")
        except Exception as e:
            print(f"  ✗ Quantized test error: {e}")

        # --- NON-QUANTIZED TEST ---
        print("\n[ Non-quantized model test ]")

        inputs_nq = {
            "weight": base_weight_fp32,
            "shared_A": shared_A,
            "intermediate": intermediate,
            "adapter_B": adapter_B,
            "alpha": np.array(alpha, dtype=np.float32),
            "adapter_index": np.array(adapter_index, dtype=np.int64),
            "rank": np.array(rank, dtype=np.int64),
        }

        try:
            outputs_nq = nq_session.run(None, inputs_nq)
            merged_fp32 = outputs_nq[0]

            # expected
            chunk_start = adapter_index * rank
            chunk_end = (adapter_index + 1) * rank
            chunked_intermediate = intermediate[chunk_start:chunk_end, :]
            delta = adapter_B @ chunked_intermediate @ shared_A * alpha
            expected = base_weight_fp32 + delta

            max_diff = np.max(np.abs(expected - merged_fp32))
            rel_error = max_diff / (np.max(np.abs(expected)) + 1e-8)

            print(f"  Non-quantized max diff: {max_diff:.6f}, relative error: {rel_error:.6f}")
            if rel_error < 1e-4:
                print("  ✓ Non-quantized test passed")
            else:
                print("  ✗ Non-quantized test failed")
        except Exception as e:
            print(f"  ✗ Non-quantized test error: {e}")

    return

if __name__ == "__main__":

    print("Creating basic LoRA merger model with dynamic shapes...")
    create_lora_merger_model("lora_qmerger_model.onnx", quantized=True)
    create_lora_merger_model("lora_merger_model.onnx", quantized=False)

    print("\nTesting basic LoRA merger model...")
    test_all_lora_merger_models(
        quantized_model_path="lora_qmerger_model.onnx",
        float_model_path="lora_merger_model.onnx"
    )

    # Create the basic LoRA merger model
    print("Creating basic MARS merger model with dynamic shapes...")
    basic_model = create_mars_merger_model(
        output_path="mars_qmerger_model.onnx",
        quantized=True
    )
    basic_model = create_mars_merger_model(
        output_path="mars_merger_model.onnx",
        quantized=False
    )
    
    # Test the basic model
    print("\nTesting basic MARS merger model...")
    test_results = test_mars_merger_model("mars_qmerger_model.onnx", "mars_merger_model.onnx")
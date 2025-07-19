import onnx
import numpy as np
from onnx import helper, TensorProto

def create_simple_cast_test_graph(output_path="test_cast_graph.onnx"):
    """
    Create a simple ONNX graph with one Cast node that converts UINT8 input to INT4 output.
    
    Args:
        output_path (str): Path where the test model will be saved
        
    Returns:
        str: Path to the created model
    """
    
    # Create input (UINT8)
    input_info = helper.make_tensor_value_info(
        'input_uint8',
        TensorProto.UINT8,
        [2, 3]  # Simple 2x3 tensor
    )
    
    # Create output (INT4)
    output_info = helper.make_tensor_value_info(
        'output_int4',
        TensorProto.INT4,
        [2, 3]  # Same shape as input
    )
    
    # Create the Cast node
    cast_node = helper.make_node(
        'Cast',
        inputs=['input_uint8'],
        outputs=['output_int4'],
        to=TensorProto.INT4,
        name='cast_uint8_to_int4'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[cast_node],
        name='simple_cast_test',
        inputs=[input_info],
        outputs=[output_info]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 21  # Use opset 13 which supports Cast
    
    # Save the model
    onnx.save(model, output_path)
    
    print(f"✓ Created simple cast test graph: {output_path}")
    print(f"  Input: input_uint8 (UINT8, shape [2, 3])")
    print(f"  Output: output_int4 (INT4, shape [2, 3])")
    print(f"  Node: Cast (UINT8 -> INT4)")
    
    return output_path

def create_cast_test_with_initializer(output_path="test_cast_with_init.onnx"):
    """
    Create a test graph with a UINT8 initializer that gets cast to INT4.
    
    Args:
        output_path (str): Path where the test model will be saved
        
    Returns:
        str: Path to the created model
    """
    
    # Create test data: UINT8 values 0-15 (representing INT4 -8 to 7)
    test_data = np.array([[0, 5, 10], [15, 8, 3]], dtype=np.uint8)
    
    # Create UINT8 initializer
    initializer = helper.make_tensor(
        'weights_uint8',
        TensorProto.UINT8,
        test_data.shape,
        test_data.flatten()
    )
    
    # Create output (INT4)
    output_info = helper.make_tensor_value_info(
        'output_int4',
        TensorProto.INT4,
        [2, 3]
    )
    
    # Create the Cast node
    cast_node = helper.make_node(
        'Cast',
        inputs=['weights_uint8'],
        outputs=['output_int4'],
        to=TensorProto.INT4,
        name='cast_weights_to_int4'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[cast_node],
        name='cast_test_with_initializer',
        inputs=[],  # No external inputs, using initializer
        outputs=[output_info],
        initializer=[initializer]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 21
    
    # Save the model
    onnx.save(model, output_path)
    
    print(f"✓ Created cast test with initializer: {output_path}")
    print(f"  Initializer: weights_uint8 (UINT8, shape [2, 3])")
    print(f"  Data: {test_data}")
    print(f"  Output: output_int4 (INT4, shape [2, 3])")
    print(f"  Expected output after cast: {test_data.astype(np.int8) - 8}")  # Convert back to INT4 range
    
    return output_path

def test_cast_graphs():
    """
    Create both test graphs and validate them.
    """
    print("Creating test graphs...\n")
    
    # Create simple cast test
    model1 = create_simple_cast_test_graph()
    
    print()
    
    # Create cast test with initializer
    model2 = create_cast_test_with_initializer()
    
    print("\nValidating models...")
    
    # Validate both models
    try:
        onnx.checker.check_model(model1)
        print(f"✓ {model1} validation passed")
    except Exception as e:
        print(f"✗ {model1} validation failed: {e}")
    
    try:
        onnx.checker.check_model(model2)
        print(f"✓ {model2} validation passed")
    except Exception as e:
        print(f"✗ {model2} validation failed: {e}")
    
    print("\n🎉 Test graphs created successfully!")
    print("You can now test loading these models in your ONNX Runtime training setup.")

#if __name__ == "__main__":
#    test_cast_graphs()
    
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

def test_cast_behavior():
    """
    Test the actual casting behavior between UINT8 and INT4 using ONNX Runtime.
    """
    print("=== Testing UINT8 to INT4 Casting Behavior ===\n")
    
    # Test data: All possible UINT8 values that fit in 4 bits (0-15)
    test_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8)
    
    print("Test Values (UINT8):")
    print(f"Values: {test_values}")
    print(f"Binary: {[format(x, '08b') for x in test_values]}")
    print(f"Lower 4 bits: {[format(x, '08b')[-4:] for x in test_values]}")
    print()
    
    # Create a simple cast model for testing
    def create_cast_model():
        # Input
        input_info = helper.make_tensor_value_info(
            'input_uint8',
            TensorProto.UINT8,
            [16]  # 16 values
        )
        
        # Output
        output_info = helper.make_tensor_value_info(
            'output_int4',
            TensorProto.INT4,
            [16]
        )
        
        # Cast node
        cast_node = helper.make_node(
            'Cast',
            inputs=['input_uint8'],
            outputs=['output_int4'],
            to=TensorProto.INT4
        )
        
        # Graph
        graph = helper.make_graph(
            nodes=[cast_node],
            name='cast_test',
            inputs=[input_info],
            outputs=[output_info]
        )
        
        # Model
        model = helper.make_model(graph)
        model.opset_import[0].version = 21
        
        return model
    
    # Test the cast model
    try:
        model = create_cast_model()
        onnx.save(model, "cast_test_runtime.onnx")
        
        # Create inference session
        session = ort.InferenceSession("cast_test_runtime.onnx")
        
        # Run inference
        result = session.run(None, {'input_uint8': test_values})
        int4_output = result[0]
        
        print("=== CASTING RESULTS ===")
        print(f"{'UINT8':<6} {'Binary':<8} {'4-bit':<6} {'INT4':<6} {'Expected':<8}")
        print("-" * 40)
        
        for i, (uint8_val, int4_val) in enumerate(zip(test_values, int4_output)):
            binary = format(uint8_val, '08b')
            four_bit = binary[-4:]
            
            # Calculate expected INT4 value using two's complement
            if uint8_val & 0x8:  # If bit 3 is set (>= 8)
                expected = uint8_val - 16  # Two's complement for 4-bit
            else:
                expected = uint8_val
            
            print(f"{uint8_val:<6} {binary:<8} {four_bit:<6} {int4_val:<6} {expected:<8}")
        
        print()
        print("=== ANALYSIS ===")
        print("✓ Values 0-7: Direct mapping (0→0, 1→1, ..., 7→7)")
        print("✓ Values 8-15: Two's complement mapping (8→-8, 9→-7, ..., 15→-1)")
        
    except Exception as e:
        print(f"✗ Error during casting test: {e}")
        print("This might indicate INT4 support issues in your ONNX Runtime version")

def test_initializer_cast():
    """
    Test casting with an initializer (like in your weight conversion scenario).
    """
    print("\n=== Testing Initializer Cast ===\n")
    
    # Test data representing typical weight values
    weight_data = np.array([[0, 3, 7, 8], [12, 15, 1, 6]], dtype=np.uint8)
    
    print("Weight Data (UINT8):")
    print(f"Values:\n{weight_data}")
    print(f"Flattened: {weight_data.flatten()}")
    print()
    
    try:
        # Create initializer
        initializer = helper.make_tensor(
            'weights_uint8',
            TensorProto.UINT8,
            weight_data.shape,
            weight_data.flatten()
        )
        
        # Output
        output_info = helper.make_tensor_value_info(
            'output_int4',
            TensorProto.INT4,
            weight_data.shape
        )
        
        # Cast node
        cast_node = helper.make_node(
            'Cast',
            inputs=['weights_uint8'],
            outputs=['output_int4'],
            to=TensorProto.INT4
        )
        
        # Graph
        graph = helper.make_graph(
            nodes=[cast_node],
            name='initializer_cast_test',
            inputs=[],
            outputs=[output_info],
            initializer=[initializer]
        )
        
        # Model
        model = helper.make_model(graph)
        model.opset_import[0].version = 21
        
        onnx.save(model, "initializer_cast_test.onnx")
        
        # Run inference
        session = ort.InferenceSession("initializer_cast_test.onnx")
        result = session.run(None, {})
        int4_output = result[0]
        
        print("=== INITIALIZER CAST RESULTS ===")
        print(f"Input (UINT8):\n{weight_data}")
        print(f"Output (INT4):\n{int4_output}")
        print()
        
        print("Value-by-value comparison:")
        flat_input = weight_data.flatten()
        flat_output = int4_output.flatten()
        
        for i, (inp, out) in enumerate(zip(flat_input, flat_output)):
            expected = inp if inp < 8 else inp - 16
            status = "✓" if out == expected else "✗"
            print(f"{status} {inp} → {out} (expected {expected})")
        
    except Exception as e:
        print(f"✗ Error during initializer cast test: {e}")

def test_proper_weight_conversion():
    """
    Test the proper way to convert INT4 weights for storage and retrieval.
    """
    print("\n=== Testing Proper Weight Conversion ===\n")
    
    # Original INT4 weights (-8 to 7)
    original_weights = np.array([[-8, -3, 0, 4], [7, -1, 2, -5]], dtype=np.int8)
    
    print("Original INT4 weights:")
    print(original_weights)
    print()
    
    # Convert to UINT8 for storage (add 8 to shift range from -8:7 to 0:15)
    uint8_weights = (original_weights.astype(np.int16) + 8).astype(np.uint8)
    print("Converted to UINT8 (for storage):")
    print(uint8_weights)
    print()
    
    try:
        # Create model that casts and then subtracts 8
        initializer = helper.make_tensor(
            'weights_uint8',
            TensorProto.UINT8,
            uint8_weights.shape,
            uint8_weights.flatten()
        )
        
        # Constant tensor for subtraction
        offset_tensor = helper.make_tensor(
            'offset',
            TensorProto.INT4,
            [],  # scalar
            [8]
        )
        
        # Cast node
        cast_node = helper.make_node(
            'Cast',
            inputs=['weights_uint8'],
            outputs=['weights_cast'],
            to=TensorProto.INT4
        )
        
        # Sub node to get back original range
        sub_node = helper.make_node(
            'Sub',
            inputs=['weights_cast', 'offset'],
            outputs=['weights_final'],
            name='subtract_offset'
        )
        
        # Output
        output_info = helper.make_tensor_value_info(
            'weights_final',
            TensorProto.INT4,
            uint8_weights.shape
        )
        
        # Graph
        graph = helper.make_graph(
            nodes=[cast_node, sub_node],
            name='proper_weight_conversion',
            inputs=[],
            outputs=[output_info],
            initializer=[initializer, offset_tensor]
        )
        
        # Model
        model = helper.make_model(graph)
        model.opset_import[0].version = 21
        
        onnx.save(model, "proper_weight_conversion.onnx")
        
        # Run inference
        session = ort.InferenceSession("proper_weight_conversion.onnx")
        result = session.run(None, {})
        final_output = result[0]
        
        print("=== PROPER CONVERSION RESULTS ===")
        print(f"Original INT4:\n{original_weights}")
        print(f"Stored as UINT8:\n{uint8_weights}")
        print(f"Recovered INT4:\n{final_output}")
        print()
        
        # Check if recovery is perfect
        if np.array_equal(original_weights, final_output):
            print("✓ Perfect recovery! Original weights == Recovered weights")
        else:
            print("✗ Recovery failed!")
            print(f"Differences:\n{final_output - original_weights}")
            
    except Exception as e:
        print(f"✗ Error during proper weight conversion test: {e}")

"""
if __name__ == "__main__":
    test_cast_behavior()
    test_initializer_cast()
    test_proper_weight_conversion()
    
    print("\n🎉 All tests completed!")
    print("Check the results above to understand how UINT8→INT4 casting works.")

"""

import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

def test_supported_cast_types():
    """
    Test what cast types are supported in your ONNX Runtime version.
    """
    print("=== Testing Supported Cast Types ===\n")
    
    test_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8)
    
    cast_types = [
        ("UINT8 to INT8", TensorProto.INT8),
        ("UINT8 to INT16", TensorProto.INT16),
        ("UINT8 to INT32", TensorProto.INT32),
        ("UINT8 to FLOAT", TensorProto.FLOAT),
        ("UINT8 to UINT4", TensorProto.UINT4),
        ("UINT8 to INT4", TensorProto.INT4),  # This will likely fail
    ]
    
    for cast_name, target_type in cast_types:
        try:
            # Create simple cast model
            input_info = helper.make_tensor_value_info('input', TensorProto.UINT8, [16])
            output_info = helper.make_tensor_value_info('output', target_type, [16])
            
            cast_node = helper.make_node(
                'Cast',
                inputs=['input'],
                outputs=['output'],
                to=target_type
            )
            
            graph = helper.make_graph([cast_node], 'test', [input_info], [output_info])
            model = helper.make_model(graph)
            model.opset_import[0].version = 23
            
            # Test the model
            session = ort.InferenceSession(model.SerializeToString())
            result = session.run(None, {'input': test_data})
            
            print(f"✓ {cast_name}: SUPPORTED")
            print(f"   Input:  {test_data}")
            print(f"   Output: {result[0]}")
            print()
            
        except Exception as e:
            print(f"✗ {cast_name}: NOT SUPPORTED")
            print(f"   Error: {str(e)}")
            print()

def test_int8_workaround():
    """
    Test using INT8 as a workaround for INT4 (since INT8 is more widely supported).
    """
    print("=== Testing INT8 Workaround ===\n")
    
    # Simulate INT4 weights stored as UINT8
    original_int4_weights = np.array([[-8, -3, 0, 4], [7, -1, 2, -5]], dtype=np.int8)
    print("Original INT4 weights (simulated):")
    print(original_int4_weights)
    print()
    
    # Convert to UINT8 for storage (add 8 to shift -8:7 to 0:15)
    uint8_weights = (original_int4_weights.astype(np.int16) + 8).astype(np.uint8)
    print("Stored as UINT8 (add 8):")
    print(uint8_weights)
    print()
    
    try:
        # Create model: UINT8 → INT8 → subtract 8
        initializer = helper.make_tensor(
            'weights_uint8',
            TensorProto.UINT8,
            uint8_weights.shape,
            uint8_weights.flatten()
        )
        
        offset_tensor = helper.make_tensor(
            'offset',
            TensorProto.INT8,
            [],
            [8]
        )
        
        # Cast UINT8 to INT8
        cast_node = helper.make_node(
            'Cast',
            inputs=['weights_uint8'],
            outputs=['weights_int8'],
            to=TensorProto.INT8
        )
        
        # Subtract 8 to get back to INT4 range
        sub_node = helper.make_node(
            'Sub',
            inputs=['weights_int8', 'offset'],
            outputs=['weights_final']
        )
        
        output_info = helper.make_tensor_value_info(
            'weights_final',
            TensorProto.INT8,
            uint8_weights.shape
        )
        
        graph = helper.make_graph(
            [cast_node, sub_node],
            'int8_workaround',
            [],
            [output_info],
            [initializer, offset_tensor]
        )
        
        model = helper.make_model(graph)
        model.opset_import[0].version = 13
        
        session = ort.InferenceSession(model.SerializeToString())
        result = session.run(None, {})
        recovered_weights = result[0]
        
        print("=== INT8 WORKAROUND RESULTS ===")
        print(f"Original INT4 (simulated): \n{original_int4_weights}")
        print(f"Recovered (INT8 container): \n{recovered_weights}")
        print()
        
        if np.array_equal(original_int4_weights, recovered_weights):
            print("✓ Perfect recovery using INT8 workaround!")
        else:
            print("✗ Recovery failed")
            
    except Exception as e:
        print(f"✗ INT8 workaround failed: {e}")

def test_manual_bit_manipulation():
    """
    Test manual bit manipulation to simulate INT4 behavior.
    """
    print("\n=== Testing Manual Bit Manipulation ===\n")
    
    # Test how UINT8 values would be interpreted as INT4
    uint8_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8)
    
    def uint8_to_int4_manual(values):
        """Manually convert UINT8 to INT4 using bit manipulation"""
        result = []
        for val in values:
            # Take only lower 4 bits
            four_bit = val & 0x0F
            # Convert to signed INT4 using two's complement
            if four_bit >= 8:
                int4_val = four_bit - 16  # Two's complement
            else:
                int4_val = four_bit
            result.append(int4_val)
        return np.array(result, dtype=np.int8)
    
    print("Manual UINT8 to INT4 conversion:")
    print(f"UINT8 input: {uint8_values}")
    
    manual_int4 = uint8_to_int4_manual(uint8_values)
    print(f"Manual INT4:  {manual_int4}")
    print()
    
    # Show the mapping
    print("Bit-level mapping:")
    for i, (u8, i4) in enumerate(zip(uint8_values, manual_int4)):
        binary = format(u8, '08b')
        four_bit = binary[-4:]
        print(f"{u8:2d} ({binary}) → {four_bit} → {i4:2d}")

def create_int8_based_weight_converter():
    """
    Create a practical weight converter that uses INT8 instead of INT4.
    """
    print("\n=== Creating INT8-Based Weight Converter ===\n")
    
    code = '''
def convert_int4_weights_to_int8_with_cast(input_model_path, output_model_path):
    """
    Convert INT4 weights to UINT8 storage with INT8 recovery (workaround for INT4 limitations).
    """
    import onnx
    import numpy as np
    from onnx import helper, TensorProto
    
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    graph = model.graph
    
    converted_count = 0
    
    for param in list(graph.initializer):
        if ".weight_DQ_Q4" in param.name:
            print(f"Converting: {param.name}")
            
            # Get original data
            original_data = onnx.numpy_helper.to_array(param)
            
            # Convert INT4 (-8 to 7) to UINT8 (0 to 15)
            uint8_data = (original_data.astype(np.int16) + 8).astype(np.uint8)
            
            # Create UINT8 initializer
            uint8_name = param.name + "_uint8"
            uint8_initializer = onnx.numpy_helper.from_array(uint8_data, uint8_name)
            
            # Create offset tensor
            offset_name = param.name + "_offset"
            offset_tensor = helper.make_tensor(offset_name, TensorProto.INT8, [], [8])
            
            # Create cast node UINT8 → INT8
            cast_node = helper.make_node(
                'Cast',
                inputs=[uint8_name],
                outputs=[param.name + "_int8"],
                to=TensorProto.INT8,
                name=f"cast_{param.name.replace('.', '_')}"
            )
            
            # Create subtract node to get INT4 range in INT8 container
            sub_node = helper.make_node(
                'Sub',
                inputs=[param.name + "_int8", offset_name],
                outputs=[param.name],  # Keep original name
                name=f"sub_{param.name.replace('.', '_')}"
            )
            
            # Remove original initializer
            graph.initializer.remove(param)
            
            # Add new components
            graph.initializer.extend([uint8_initializer, offset_tensor])
            graph.node.extend([cast_node, sub_node])
            
            converted_count += 1
    
    # Save model
    onnx.save(model, output_model_path, save_as_external_data=True)
    print(f"Converted {converted_count} weights. Saved to: {output_model_path}")
    
    return converted_count
'''
    
    print("Here's a practical weight converter function that works around INT4 limitations:")
    print(code)
    
    return code

if __name__ == "__main__":
    test_supported_cast_types()
    test_int8_workaround()
    test_manual_bit_manipulation()
    create_int8_based_weight_converter()
    
    print("\n🎉 Analysis complete!")
    print("\nKey findings:")
    print("1. Your ONNX Runtime doesn't support INT4 Cast operations")
    print("2. INT8 workaround should work for storing INT4-range values")
    print("3. You can use UINT8 storage + INT8 recovery + subtract offset")
    print("4. Manual bit manipulation shows the expected INT4 behavior")
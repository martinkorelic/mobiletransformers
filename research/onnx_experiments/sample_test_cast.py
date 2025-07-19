import onnx
from onnx import helper, TensorProto, ValueInfoProto, numpy_helper
import numpy as np

def create_uint8_to_int4_cast_graph():
    """
    Creates an ONNX graph that casts uint8 input to int4 output using Cast operator.
    """
    
    # Define input tensor info (uint8)
    input_tensor = helper.make_tensor_value_info(
        name='input',
        elem_type=TensorProto.INT8,
        shape=[1, 4]  # Example shape: batch_size=1, features=4
    )
    
    # Define output tensor info (int4)
    output_tensor = helper.make_tensor_value_info(
        name='output',
        elem_type=TensorProto.INT4,  # int4 type
        shape=[1, 4]  # Same shape as input
    )
    
    # Create the Cast node
    cast_node = helper.make_node(
        op_type='Cast',
        inputs=['input'],
        outputs=['output'],
        to=TensorProto.INT4  # Cast to int4
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[cast_node],
        name='uint8_to_int4_cast',
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph=graph,
        producer_name='onnx-cast-example',
        opset_imports=[helper.make_opsetid("", 21)]  # Use opset 21 for int4 support
    )
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

def test_cast_model():
    """
    Test the created model with sample data.
    """
    # Create the model
    model = create_uint8_to_int4_cast_graph()
    
    # Save the model
    onnx.save(model, "uint8_to_int4_cast.onnx")
    print("Model saved as 'uint8_to_int4_cast.onnx'")
    
    # Print model info
    print("\nModel Information:")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Graph name: {model.graph.name}")
    
    print("\nInputs:")
    for input_tensor in model.graph.input:
        print(f"  {input_tensor.name}: {TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)}")
    
    print("\nOutputs:")
    for output_tensor in model.graph.output:
        print(f"  {output_tensor.name}: {TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)}")
    
    print("\nNodes:")
    for node in model.graph.node:
        print(f"  {node.op_type}: {node.input} -> {node.output}")
        for attr in node.attribute:
            if attr.name == 'to':
                print(f"    Cast to: {TensorProto.DataType.Name(attr.i)}")
    
    return model

def create_extended_cast_chain():
    """
    Creates a more complex graph with multiple cast operations.
    """
    
    # Input: uint8
    input_tensor = helper.make_tensor_value_info(
        name='input',
        elem_type=TensorProto.UINT8,
        shape=[1, 4]
    )
    
    # Intermediate: int8
    intermediate_tensor = helper.make_tensor_value_info(
        name='intermediate',
        elem_type=TensorProto.INT8,
        shape=[1, 4]
    )
    
    # Output: int4
    output_tensor = helper.make_tensor_value_info(
        name='output',
        elem_type=TensorProto.INT4,
        shape=[1, 4]
    )
    
    # Cast uint8 to int8
    cast_node1 = helper.make_node(
        op_type='Cast',
        inputs=['input'],
        outputs=['intermediate'],
        to=TensorProto.INT8
    )
    
    # Cast int8 to int4
    cast_node2 = helper.make_node(
        op_type='Cast',
        inputs=['intermediate'],
        outputs=['output'],
        to=TensorProto.INT4
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[cast_node1, cast_node2],
        name='extended_cast_chain',
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph=graph,
        producer_name='onnx-cast-chain-example',
        opset_imports=[helper.make_opsetid("", 21)]
    )
    
    onnx.checker.check_model(model)
    return model

def run_model_inference():
    """
    Run inference on the created models and compare outputs.
    """
    try:
        import onnxruntime as ort
        print("\n" + "="*60)
        print("RUNNING MODEL INFERENCE")
        print("="*60)
        
        # Test data - various uint8 values
        test_inputs = [
            np.array([[0, 1, 7, 8]], dtype=np.uint8),      # Values within int4 range
            np.array([[15, 16, 31, 255]], dtype=np.uint8), # Values outside int4 range
            np.array([[100, 150, 200, 250]], dtype=np.uint8), # Large values
        ]
        
        print("Testing basic cast model (uint8 -> int4)...")
        try:
            session_basic = ort.InferenceSession("uint8_to_int4_cast.onnx")
            
            for i, input_data in enumerate(test_inputs):
                print(f"\nTest {i+1}:")
                print(f"  Input (uint8):  {input_data}")
                print(f"  Input range:    {input_data.min()} to {input_data.max()}")
                
                try:
                    outputs = session_basic.run(None, {"input": input_data})
                    print(f"  Output (int4):  {outputs[0]}")
                    print(f"  Output range:   {outputs[0].min()} to {outputs[0].max()}")
                    
                    # Show the casting behavior
                    original_values = input_data.flatten()
                    cast_values = outputs[0].flatten()
                    print(f"  Cast mapping:   {list(zip(original_values, cast_values))}")
                    
                except Exception as e:
                    print(f"  Error during inference: {e}")
                    
        except Exception as e:
            print(f"Error loading basic model: {e}")
        
        print("\n" + "-"*40)
        print("Testing extended cast chain model (uint8 -> int8 -> int4)...")
        try:
            session_extended = ort.InferenceSession("extended_cast_chain.onnx")
            
            for i, input_data in enumerate(test_inputs):
                print(f"\nTest {i+1}:")
                print(f"  Input (uint8):  {input_data}")
                
                try:
                    outputs = session_extended.run(None, {"input": input_data})
                    print(f"  Final Output (int4): {outputs[0]}")
                    
                except Exception as e:
                    print(f"  Error during inference: {e}")
                    
        except Exception as e:
            print(f"Error loading extended model: {e}")
            
    except ImportError:
        print("\nonnxruntime not available. Install with: pip install onnxruntime")
        print("Skipping inference tests.")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("This might be due to int4 support not being available in your ONNX Runtime version.")

def analyze_type_conversion():
    """
    Analyze how different values are converted from uint8 to int4.
    """
    print("\n" + "="*60)
    print("TYPE CONVERSION ANALYSIS")
    print("="*60)
    
    print("uint8 range: 0 to 255")
    print("int4 range:  -8 to 7")
    print("\nExpected conversion behavior:")
    print("- Values 0-7: Direct mapping")
    print("- Values 8-15: Likely wrapped to -8 to -1")
    print("- Values 16+: Modulo operation or saturation")
    
    # Show theoretical conversion for some values
    print("\nTheoretical conversions (implementation may vary):")
    test_values = [0, 1, 7, 8, 15, 16, 31, 127, 255]
    for val in test_values:
        # This is a simplified representation - actual behavior depends on implementation
        int4_approx = ((val + 8) % 16) - 8 if val > 7 else val
        print(f"  uint8 {val:3d} -> int4 ~{int4_approx:2d}")

def create_comparison_model():
    """
    Create a model that outputs both original and cast values for comparison.
    """
    print("\n" + "="*60)
    print("CREATING COMPARISON MODEL")
    print("="*60)
    
    # Input: uint8
    input_tensor = helper.make_tensor_value_info(
        name='input',
        elem_type=TensorProto.UINT8,
        shape=[1, 4]
    )
    
    # Output 1: Original uint8 (for comparison)
    output1_tensor = helper.make_tensor_value_info(
        name='original',
        elem_type=TensorProto.UINT8,
        shape=[1, 4]
    )
    
    # Output 2: Cast to int4
    output2_tensor = helper.make_tensor_value_info(
        name='cast_int4',
        elem_type=TensorProto.INT4,
        shape=[1, 4]
    )
    
    # Identity node (pass through original)
    identity_node = helper.make_node(
        op_type='Identity',
        inputs=['input'],
        outputs=['original']
    )
    
    # Cast node
    cast_node = helper.make_node(
        op_type='Cast',
        inputs=['input'],
        outputs=['cast_int4'],
        to=TensorProto.INT4
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[identity_node, cast_node],
        name='comparison_model',
        inputs=[input_tensor],
        outputs=[output1_tensor, output2_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph=graph,
        producer_name='onnx-comparison-example',
        opset_imports=[helper.make_opsetid("", 21)]
    )
    
    onnx.checker.check_model(model)
    onnx.save(model, "comparison_model.onnx")
    print("Comparison model saved as 'comparison_model.onnx'")
    
    return model

def run_comparison_inference():
    """
    Run inference on the comparison model to see original vs cast values.
    """
    try:
        import onnxruntime as ort
        print("\n" + "="*60)
        print("COMPARISON MODEL INFERENCE")
        print("="*60)
        
        session = ort.InferenceSession("comparison_model.onnx")
        
        # Test with a range of values
        test_data = np.array([[0, 7, 8, 15, 31, 63, 127, 255]], dtype=np.uint8)
        
        print(f"Test input: {test_data}")
        
        try:
            outputs = session.run(None, {"input": test_data})
            original = outputs[0]
            cast_int4 = outputs[1]
            
            print(f"\nResults:")
            print(f"Original (uint8): {original}")
            print(f"Cast (int4):      {cast_int4}")
            
            print(f"\nValue-by-value comparison:")
            for i in range(test_data.shape[1]):
                orig_val = original[0, i]
                cast_val = cast_int4[0, i]
                print(f"  {orig_val:3d} (uint8) -> {cast_val:3d} (int4)")
                
        except Exception as e:
            print(f"Error during comparison inference: {e}")
            
    except ImportError:
        print("onnxruntime not available for comparison test.")
    except Exception as e:
        print(f"Error in comparison test: {e}")

if __name__ == "__main__":
    print("Creating ONNX graph with uint8 to int4 cast...")
    
    # Create and test the basic cast model
    model = test_cast_model()
    
    print("\n" + "="*50)
    print("Creating extended cast chain model...")
    
    # Create extended model
    extended_model = create_extended_cast_chain()
    onnx.save(extended_model, "extended_cast_chain.onnx")
    
    print("Extended model saved as 'extended_cast_chain.onnx'")
    print("\nExtended model nodes:")
    for i, node in enumerate(extended_model.graph.node):
        cast_to = next(attr.i for attr in node.attribute if attr.name == 'to')
        print(f"  Node {i+1}: {node.op_type} -> {TensorProto.DataType.Name(cast_to)}")
    
    # Create comparison model
    comparison_model = create_comparison_model()
    
    # Analyze type conversion
    analyze_type_conversion()
    
    # Run inference tests
    run_model_inference()
    
    # Run comparison inference
    run_comparison_inference()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Created models:")
    print("  1. uint8_to_int4_cast.onnx - Basic cast model")
    print("  2. extended_cast_chain.onnx - Multi-step cast model") 
    print("  3. comparison_model.onnx - Side-by-side comparison model")
    print("\nNote: int4 support requires ONNX Runtime with opset 21+ support.")
    print("If inference fails, your runtime may not support int4 types yet.")
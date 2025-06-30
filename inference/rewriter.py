import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer, DefaultWeightOnlyQuantConfig
from onnx import numpy_helper


def analyze_base_layer_connections(model):
    """
    Analyze the connection patterns for base_layer nodes to understand the graph structure.
    """
    graph = model.graph
    
    # Create mappings for easier analysis
    node_map = {node.name: node for node in graph.node}
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node
    
    # Find all base_layer nodes
    base_layer_nodes = [node for node in graph.node if 'base_layer' in node.name]
    
    print(f"Found {len(base_layer_nodes)} base_layer nodes")
    print("\nAnalyzing base_layer connections:")
    
    for node in base_layer_nodes:
        print(f"\nNode: {node.name} (type: {node.op_type})")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        
        # For each input, show what feeds it
        for i, input_name in enumerate(node.input):
            if input_name in output_to_node:
                feeding_node = output_to_node[input_name]
                print(f"    Input[{i}] '{input_name}' <- {feeding_node.name} ({feeding_node.op_type})")
            else:
                # Check if it's an initializer
                init_names = [init.name for init in graph.initializer]
                if input_name in init_names:
                    print(f"    Input[{i}] '{input_name}' <- Initializer")
                else:
                    print(f"    Input[{i}] '{input_name}' <- Unknown source")

def fuse_base_layer_transpose_matmul(model):
    """
    Fuse Transpose->MatMul patterns specifically for base_layer nodes.
    Based on the pattern: MatMul(activation, Transpose(weight)) -> MatMul(activation, transposed_weight)
    """
    graph = model.graph
    nodes_to_remove = []
    initializers_to_remove = []
    fused_count = 0
    
    # Find all base_layer Transpose nodes
    base_layer_transposes = [node for node in graph.node 
                           if node.op_type == 'Transpose' and 'base_layer' in node.name]
    
    print(f"Found {len(base_layer_transposes)} base_layer Transpose nodes")
    
    for transpose_node in base_layer_transposes:
        print(f"\nProcessing Transpose: {transpose_node.name}")
        
        # Find all nodes that use this transpose's output
        transpose_output = transpose_node.output[0]
        
        # Look for MatMul nodes that use this transpose output
        for matmul_node in graph.node:
            if (matmul_node.op_type == 'MatMul' and 
                'base_layer' in matmul_node.name and 
                transpose_output in matmul_node.input):
                
                # Find which input of MatMul uses the transpose (should be input[1] based on your output)
                transpose_input_idx = None
                for i, input_name in enumerate(matmul_node.input):
                    if input_name == transpose_output:
                        transpose_input_idx = i
                        break
                
                print(f"  Found connected MatMul: {matmul_node.name}")
                print(f"    Transpose feeds MatMul input[{transpose_input_idx}]")
                
                # Get the weight initializer that feeds the Transpose
                weight_name = transpose_node.input[0]
                weight_init = None
                
                for init in graph.initializer:
                    if init.name == weight_name:
                        weight_init = init
                        break
                
                if weight_init is None:
                    print(f"    Warning: Could not find initializer {weight_name}")
                    continue
                
                try:
                    # Get weight array and transpose it
                    weight_array = numpy_helper.to_array(weight_init)
                    print(f"    Original weight shape: {weight_array.shape}")
                    
                    # The transpose node applies transpose, so we pre-apply it to the weight
                    transposed_weight = weight_array.T
                    print(f"    Pre-transposed weight shape: {transposed_weight.shape}")
                    
                    # Create new initializer with pre-transposed weight
                    #new_weight_name = weight_name.replace('.weight', '_fused.weight')
                    new_init = numpy_helper.from_array(
                        transposed_weight, 
                        name=weight_name
                    )
                    graph.initializer.append(new_init)
                    
                    # Update MatMul to use the new pre-transposed weight directly
                    matmul_node.input[transpose_input_idx] = weight_name
                    
                    # Mark nodes for removal
                    if transpose_node not in nodes_to_remove:
                        nodes_to_remove.append(transpose_node)
                    if weight_init not in initializers_to_remove:
                        initializers_to_remove.append(weight_init)
                    
                    fused_count += 1
                    print(f"    ✓ Successfully fused: {transpose_node.name} -> {matmul_node.name}")
                    
                except Exception as e:
                    print(f"    ✗ Error processing: {e}")
                    continue
    
    # Remove the old transpose nodes
    for node in nodes_to_remove:
        graph.node.remove(node)
        print(f"Removed transpose node: {node.name}")
    
    # Remove the old initializers
    for init in initializers_to_remove:
        graph.initializer.remove(init)
        print(f"Removed old initializer: {init.name}")
    
    print(f"\n🎉 Fusion complete!")
    print(f"   Fused {fused_count} Transpose->MatMul pairs")
    print(f"   Removed {len(nodes_to_remove)} Transpose nodes")
    print(f"   Removed {len(initializers_to_remove)} old initializers")
    
    return model

def onnx_matmul_quantization(onnx_model_path, onnx_model_quant_output, block_size=32, accuracy_level=4, exclude_weights=["q_proj", "k_proj"], exclude_extra_layers=["embed_tokens"]):

    onnx_model = onnx.load(onnx_model_path)

    nodes_to_not_quantize = []

    # Exclude trainable nodes
    for param in onnx_model.graph.node:
        if any((allowed_layer in param.name for allowed_layer in exclude_weights)):
            nodes_to_not_quantize.append(param.name)
        if any(allowed_layer in param.name for allowed_layer in exclude_extra_layers):
            nodes_to_not_quantize.append(param.name)

    quant = MatMul4BitsQuantizer(
            model=onnx_model,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=accuracy_level,
            # Exclude trainable LoRA layers from quantization
            nodes_to_exclude=nodes_to_not_quantize,
            quant_format=QuantFormat.QDQ,
        )
    quant.process()

    onnx.save_model(quant.model.model, onnx_model_quant_output, save_as_external_data=True)

if __name__ == "__main__":

    model = onnx.load('build/train_models/model.onnx')

    fused_model = fuse_base_layer_transpose_matmul(model)
    onnx.save_model(fused_model, 'quantization_ready.onnx', save_as_external_data=True)

    #onnx_matmul_quantization('quantization_ready.onnx', 'build/train_models/quant_model.onnx')
import onnx
import os
from typing import List, Dict, Tuple, Optional

def print_external_tensors(model_path: str, verbose: bool = False) -> Dict[str, Dict]:
    """
    Load an ONNX model and print all external tensor names and their properties.
    
    Args:
        model_path: Path to the ONNX model file
        verbose: If True, print detailed information about each tensor
        
    Returns:
        Dictionary containing external tensor information
    """
    try:
        # Load the model
        print(f"Loading model from: {model_path}")
        model = onnx.load(model_path, load_external_data=False)  # Don't load external data yet
        
        external_tensors = {}
        internal_tensors = {}
        
        print(f"\n=== ONNX Model External Tensor Analysis ===")
        print(f"Model IR Version: {model.ir_version}")
        print(f"Producer: {model.producer_name} {model.producer_version}")
        
        # Check initializers in the main graph
        print(f"\n--- Main Graph Initializers ---")
        external_count, internal_count = analyze_graph_tensors(
            model.graph, external_tensors, internal_tensors, verbose
        )
        
        # Check subgraphs (if any)
        subgraph_external = 0
        subgraph_internal = 0
        for i, node in enumerate(model.graph.node):
            for j, attr in enumerate(node.attribute):
                if attr.type == onnx.AttributeProto.GRAPH and attr.g:
                    print(f"\n--- Subgraph in node {node.name or f'node_{i}'} ---")
                    ext_count, int_count = analyze_graph_tensors(
                        attr.g, external_tensors, internal_tensors, verbose, 
                        prefix=f"subgraph_{i}_{j}_"
                    )
                    subgraph_external += ext_count
                    subgraph_internal += int_count
        
        # Summary
        total_external = external_count + subgraph_external
        total_internal = internal_count + subgraph_internal
        
        print(f"\n=== SUMMARY ===")
        print(f"External tensors: {total_external}")
        print(f"Internal tensors: {total_internal}")
        print(f"Total tensors: {total_external + total_internal}")
        
        if total_external > 0:
            print(f"\n=== EXTERNAL TENSOR NAMES ===")
            for name, info in external_tensors.items():
                print(f"  • {name}")
                if verbose and info['external_data']:
                    for entry in info['external_data']:
                        print(f"    - {entry['key']}: {entry['value']}")
        
        return {
            'external_tensors': external_tensors,
            'internal_tensors': internal_tensors,
            'summary': {
                'external_count': total_external,
                'internal_count': total_internal,
                'total_count': total_external + total_internal
            }
        }
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}


def analyze_graph_tensors(graph, external_tensors: Dict, internal_tensors: Dict, 
                         verbose: bool = False, prefix: str = "") -> Tuple[int, int]:
    """
    Analyze tensors in a specific graph (main graph or subgraph).
    
    Returns:
        Tuple of (external_count, internal_count)
    """
    external_count = 0
    internal_count = 0
    
    for initializer in graph.initializer:
        tensor_name = prefix + initializer.name
        
        # Check if tensor has external data
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            external_count += 1
            
            # Parse external data information
            external_data_info = []
            for ext_data in initializer.external_data:
                external_data_info.append({
                    'key': ext_data.key,
                    'value': ext_data.value
                })
            
            external_tensors[tensor_name] = {
                'name': initializer.name,
                'data_type': initializer.data_type,
                'dims': list(initializer.dims),
                'external_data': external_data_info
            }
            
            if verbose:
                print(f"  🔗 EXTERNAL: {tensor_name}")
                print(f"     Type: {onnx.TensorProto.DataType.Name(initializer.data_type)}")
                print(f"     Shape: {list(initializer.dims)}")
                for ext_data in initializer.external_data:
                    print(f"     {ext_data.key}: {ext_data.value}")
                print()
        else:
            internal_count += 1
            
            # Calculate size for internal tensors
            element_count = 1
            for dim in initializer.dims:
                element_count *= dim
            
            internal_tensors[tensor_name] = {
                'name': initializer.name,
                'data_type': initializer.data_type,
                'dims': list(initializer.dims),
                'element_count': element_count
            }
            
            if verbose:
                print(f"  📦 INTERNAL: {tensor_name}")
                print(f"     Type: {onnx.TensorProto.DataType.Name(initializer.data_type)}")
                print(f"     Shape: {list(initializer.dims)} ({element_count} elements)")
                print()
    
    if not verbose:
        print(f"  External: {external_count}, Internal: {internal_count}")
    
    return external_count, internal_count


def find_external_data_files(model_path: str) -> List[str]:
    """
    Find all external data files referenced by the model.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        List of external data file paths
    """
    try:
        model = onnx.load(model_path, load_external_data=False)
        model_dir = os.path.dirname(model_path)
        external_files = set()
        
        def collect_external_files(graph):
            for initializer in graph.initializer:
                if initializer.data_location == onnx.TensorProto.EXTERNAL:
                    for ext_data in initializer.external_data:
                        if ext_data.key == "location":
                            file_path = os.path.join(model_dir, ext_data.value)
                            external_files.add(file_path)
        
        # Check main graph
        collect_external_files(model.graph)
        
        # Check subgraphs
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH and attr.g:
                    collect_external_files(attr.g)
        
        return sorted(list(external_files))
        
    except Exception as e:
        print(f"Error finding external data files: {e}")
        return []


def compare_external_tensors_with_files(model_path: str, external_weights_dir: str):
    """
    Compare external tensor names in the model with available weight files.
    
    Args:
        model_path: Path to the ONNX model
        external_weights_dir: Directory containing external weight files
    """
    print(f"\n=== COMPARING MODEL TENSORS WITH AVAILABLE FILES ===")
    
    # Get external tensors from model
    tensor_info = print_external_tensors(model_path, verbose=False)
    model_external_tensors = set(tensor_info.get('external_tensors', {}).keys())
    
    # Get available weight files
    available_weights = set()
    if os.path.exists(external_weights_dir):
        for root, dirs, files in os.walk(external_weights_dir):
            for file in files:
                if file.endswith('.tensor'):
                    # Reconstruct the tensor name from the file path
                    rel_path = os.path.relpath(root, external_weights_dir)
                    if rel_path != '.':
                        tensor_name = f"{rel_path}.{os.path.splitext(file)[0]}"
                    else:
                        tensor_name = os.path.splitext(file)[0]
                    available_weights.add(tensor_name)
    
    print(f"\nModel expects {len(model_external_tensors)} external tensors")
    print(f"Found {len(available_weights)} weight files")
    
    # Find mismatches
    missing_in_files = model_external_tensors - available_weights
    extra_in_files = available_weights - model_external_tensors
    
    if missing_in_files:
        print(f"\n❌ Tensors expected by model but not found in files ({len(missing_in_files)}):")
        for tensor in sorted(missing_in_files):
            print(f"  • {tensor}")
    
    if extra_in_files:
        print(f"\n⚠️  Files found but not expected by model ({len(extra_in_files)}):")
        for tensor in sorted(extra_in_files):
            print(f"  • {tensor}")
    
    if not missing_in_files and not extra_in_files:
        print(f"\n✅ Perfect match! All tensors align between model and files.")


# Example usage
if __name__ == "__main__":
    # Example usage:
    model_path = "build/inference/ext_quant_model.onnx.onnx"
    #external_weights_dir = "path/to/external/weights/merged"
    
    # Print all external tensors
    result = print_external_tensors(model_path, verbose=True)
    
    # Find external data files
    external_files = find_external_data_files(model_path)
    print(f"\n=== EXTERNAL DATA FILES ===")
    for file_path in external_files:
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"{exists} {file_path}")
    
    # Compare with your weight files
    #compare_external_tensors_with_files(model_path, external_weights_dir)
import numpy as np
from collections import defaultdict

from peft_models.ablation.layer import Linear

def get_ablation_linear_layers(model):
    ablation_linear_layers = []

    # Recursive function to iterate through model layers
    def recursive_layers(module, prefix=""):
        for child_name, child_module in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child_module, Linear):
                ablation_linear_layers.append((full_name, child_module))
            else:
                recursive_layers(child_module, full_name)

    recursive_layers(model)
    return ablation_linear_layers


def save_peft_metrics_to_npz(model, output_path="peft_training_metrics.npz"):
    """
    Extract stored metrics from all peft linear layers and save to NPZ file
    
    Args:
        model: The trained model containing peft layers
        output_path: Path to save the NPZ file
    """
    # Get all peft linear layers
    peft_linear_layers = get_ablation_linear_layers(model)
    
    # Dictionary to store all metrics organized by layer name
    all_metrics = {}
    
    for full_name, layer in peft_linear_layers:
        print(f"Processing layer: {full_name}")
        
        # Check if this layer has stored metrics
        if not hasattr(layer, 'stored_metrics') or not layer.stored_metrics:
            print(f"  No metrics found for {full_name}")
            continue
        
        if 'windows' not in layer.stored_metrics:
            print(f"  No window metrics found for {full_name}")
            continue
        
        # Extract metrics for this layer
        layer_metrics = extract_layer_metrics(layer.stored_metrics['windows'])
        
        # Add to overall metrics dictionary
        for metric_key, metric_data in layer_metrics.items():
            all_metrics[f"{full_name}/{metric_key}"] = metric_data
    
    # Save to NPZ file
    if all_metrics:
        np.savez_compressed(output_path, **all_metrics)
        print(f"Saved metrics for {len(peft_linear_layers)} layers to {output_path}")
        print(f"Metrics keys: {list(all_metrics.keys())}")
    else:
        print("No metrics found to save!")
    
    return all_metrics

def extract_layer_metrics(windows_data):
    """
    Extract and organize metrics from window data for a single layer
    
    Args:
        windows_data: List of window metrics from stored_metrics['windows']
        layer_name: Full name of the layer
    
    Returns:
        Dictionary with organized metrics
    """
    metrics = {
        'step_ranges': [],
        'adapters': [],
        'layer_metrics': defaultdict(list)
    }
    
    for window in windows_data:
        step_range = window['step_range']
        metrics['step_ranges'].append(step_range)
        
        # Process each layer's metrics in this window
        for layer_key, layer_data in window['layers'].items():            
            # Store metrics
            for metric_name, metric_value in layer_data.items():
                full_metric_key = f"{layer_key}/{metric_name}"
                
                # Convert to numpy if it isn't already
                if isinstance(metric_value, np.ndarray):
                    value = metric_value
                else:
                    value = np.array(metric_value)
                
                metrics['layer_metrics'][full_metric_key].append(value)
        
        # Store adapter info (assuming same adapter for all layers in window)
        if window['layers']:
            first_layer = next(iter(window['layers'].keys()))
            metrics['adapters'].append(first_layer)
    
    # Convert lists to numpy arrays
    final_metrics = {}
    final_metrics['step_ranges'] = np.array(metrics['step_ranges'])
    final_metrics['adapters'] = np.array(metrics['adapters'], dtype='U10')  # String array
    
    # Stack metrics across time windows
    for metric_key, metric_list in metrics['layer_metrics'].items():
        if metric_list:
            try:
                # Try to stack - works if all have same shape
                final_metrics[metric_key] = np.stack(metric_list)
            except ValueError:
                # If shapes differ, store as object array
                final_metrics[metric_key] = np.array(metric_list, dtype=object)
    
    return final_metrics

def load_peft_metrics(npz_path):
    """
    Load saved metrics and provide analysis utilities
    
    Args:
        npz_path: Path to the NPZ file
    
    Returns:
        Dictionary with loaded metrics and analysis functions
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Group metrics by layer
    layers = {}
    for key in data.keys():
        parts = key.split('/')
        if len(parts) >= 2:
            # Extract layer name (everything before the last metric part)
            layer_parts = []
            metric_parts = []
            
            # Common metric suffixes
            metric_suffixes = ['step_ranges', 'adapters', 'down_project', 'up_project', 
                             'intermediate', 'input_vector', 'in_magnitude', 'out_magnitude',
                             'weight_norm', 'grad_norm']
            
            found_metric = False
            for i, part in enumerate(parts):
                if any(suffix in part for suffix in metric_suffixes) and not found_metric:
                    layer_parts = parts[:i] if i > 0 else parts[:1]
                    metric_parts = parts[i:]
                    found_metric = True
                    break
            
            if not found_metric:
                # Fallback: assume last part is metric
                layer_parts = parts[:-1]
                metric_parts = [parts[-1]]
            
            layer_name = '/'.join(layer_parts)
            metric_name = '/'.join(metric_parts)
            
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name][metric_name] = data[key]
    
    return {
        'raw_data': data,
        'layers': layers,
        'layer_names': list(layers.keys())
    }
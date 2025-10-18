import matplotlib

from research.utils import load_peft_metrics
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_metric_key(metric_key, module):
    """
    Parse metric key
    """

    parts = metric_key.split('/')
    if len(parts) == 2:
        adapter = parts[0] 
        metric = parts[1]
        return {
            'module': module,
            'adapter': adapter,
            'metric': metric
        }
    return None

def plot_weight_norm_heatmaps(analysis_data, layer_name, save_dir="plots", figsize=(16, 12)):
    """
    Create heatmap plots only for weight_norm_mean metrics
    """
    if layer_name not in analysis_data['layers']:
        print(f"Layer {layer_name} not found in data")
        return
    
    layer_data = analysis_data['layers'][layer_name]

    # Find only weight_norm_mean metrics
    weight_norm_data = {}
    for metric_key, metric_data in layer_data.items():
        if isinstance(metric_data, np.ndarray) and metric_data.ndim >= 2:
            parsed = parse_metric_key(metric_key, layer_name)
            
            if parsed and parsed['metric'] == 'weight_norm_mean':
                adapter = parsed['adapter']
                # Only plot specific adapters
                #if adapter in ['up_project_ablation', 'down_project_ablation']:
                weight_norm_data[adapter] = metric_data
    
    if not weight_norm_data:
        print("No weight_norm_mean data found for specified adapters")
        return
    
    # Create plots
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_adapters = len(weight_norm_data)
    fig, axes = plt.subplots(n_adapters, 1, figsize=(figsize[0], figsize[1] * n_adapters // 3), 
                            sharex=False, squeeze=False)
    
    if n_adapters == 1:
        axes = axes.reshape(1, -1)
    
    # Main title
    fig.suptitle(f'Weight Norm Mean: {layer_name}', fontsize=16, y=0.95)
    
    for idx, (adapter, data) in enumerate(weight_norm_data.items()):
        ax = axes[idx, 0]
        
        # Create heatmap
        n_windows, n_features = data.shape
        im = ax.imshow(data, cmap='viridis', aspect='auto', origin='lower')
        
        # Customize the plot
        ax.set_title(f'{adapter}/weight_norm_mean', fontsize=14, pad=20)
        ax.set_ylabel('Step Window', fontsize=12)
        ax.set_xlabel('Feature Index', fontsize=12)
        
        # Set y-axis labels starting from 1
        ax.set_yticks(range(n_windows))
        ax.set_yticklabels([str(i+1) for i in range(n_windows)], fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('weight_norm_mean', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    # Save as PDF
    safe_layer_name = layer_name.replace('/', '_').replace('.', '_')
    filename = f"{safe_layer_name}_weight_norm_mean.pdf"
    plt.savefig(Path(save_dir) / filename, format='pdf', dpi=300, bbox_inches='tight')

def plot_all_weight_norms_comparison(analysis_data, layer_name, save_dir="plots", figsize=(10, 6)):
    """
    Create a single comparison plot with all weight norm adapters side by side
    """
    if layer_name not in analysis_data['layers']:
        print(f"Layer {layer_name} not found in data")
        return
    
    layer_data = analysis_data['layers'][layer_name]
    
    # Find weight_norm_mean metrics
    weight_norm_data = {}
    for metric_key, metric_data in layer_data.items():
        if isinstance(metric_data, np.ndarray) and metric_data.ndim >= 2:
            parsed = parse_metric_key(metric_key, layer_name)
            if parsed and parsed['metric'] == 'weight_norm_mean':
                adapter = parsed['adapter']
                #if adapter in ['up_project_ablation', 'down_project_ablation']:
                weight_norm_data[adapter] = metric_data
    
    if not weight_norm_data:
        print("No weight_norm_mean data found")
        return
    
    # Create side-by-side comparison
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_adapters = len(weight_norm_data)
    fig, axes = plt.subplots(1, n_adapters, figsize=figsize, sharey=True)
    
    if n_adapters == 1:
        axes = [axes]
    
    fig.suptitle(f'Weight Norm Comparison', fontsize=14)
    
    for idx, (adapter, data) in enumerate(weight_norm_data.items()):
        ax = axes[idx]
        
        n_windows, n_features = data.shape
        im = ax.imshow(data, cmap='viridis', aspect='auto', origin='lower')
        
        if 'down_project' in adapter:
            nice_name = 'Down projection (Matrix A)'
        elif 'up_project' in adapter:
            nice_name = 'Up projection (Matrix B)'

        ax.set_title(f'{nice_name}', fontsize=12)
        ax.set_xlabel('Feature Index', fontsize=11)
        
        if idx == 0:  # Only label y-axis on leftmost plot
            ax.set_ylabel('Step Window', fontsize=12)
        
        # Set y-axis labels starting from 1
        ax.set_yticks(range(n_windows))
        ax.set_yticklabels([str(i+1) for i in range(n_windows)], fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    # Save as PDF
    safe_layer_name = layer_name.replace('/', '_').replace('.', '_')
    filename = f"{safe_layer_name}_weight_norm_comparison.pdf"
    plt.savefig(Path(save_dir) / filename, format='pdf', dpi=300, bbox_inches='tight')

# Usage example:
def analyze_weight_norms(npz_path, layer_name, save_dir="plots"):
    """Analyze only weight norm metrics for a specific layer"""
    from research.utils import load_peft_metrics
    
    # Load data
    analysis_data = load_peft_metrics(npz_path)
    
    # Create weight norm plots
    plot_weight_norm_heatmaps(analysis_data, layer_name, save_dir)
    plot_all_weight_norms_comparison(analysis_data, layer_name, save_dir)
    
    print(f"Weight norm plots saved as PDFs in {save_dir}/")

def plot_cumulative_grad_norm_heatmaps(analysis_data, save_dir="plots", figsize=(14, 9)):
    """
    Create heatmaps showing absolute cumulative change in gradient norm mean across decoder layers and projection types.
    
    Args:
        analysis_data: Dictionary with loaded metrics from load_peft_metrics()
        save_dir: Directory to save plots
        figsize: Figure size tuple
    """
    
    # Define the projection types we're looking for
    projection_types = ['k_proj', 'v_proj', 'o_proj', 'q_proj', 'gate_proj', 'down_proj', 'up_proj']
    adapter_types = ['up_project_ablation', 'down_project_ablation']
    
    # Extract decoder layer numbers and organize data
    layer_data = {}  # {adapter: {layer_num: {proj_type: grad_norm_data}}}
    
    for adapter in adapter_types:
        layer_data[adapter] = {}
    
    # Process each layer in the data
    for layer_name, layer_metrics in analysis_data['layers'].items():
        # Extract decoder layer number from paths like "base_model.model.model.layers.18.self_attn.q_proj"
        if 'base_model.model.model.layers.' in layer_name:
            try:
                # Split by dots and find the layer number after "layers"
                parts = layer_name.split('.')
                layer_idx = None
                proj_type = None
                
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        break
                
                # Extract projection type (last part of the path)
                if parts[-1] in projection_types:
                    proj_type = parts[-1]
                
                if layer_idx is None or proj_type is None:
                    continue
                    
            except (ValueError, IndexError):
                continue
            
            # Look for grad_norm_mean metrics in this layer
            for metric_key, metric_data in layer_metrics.items():
                if 'grad_norm_mean' in metric_key and isinstance(metric_data, np.ndarray):
                    # Parse the metric key to find adapter type
                    for adapter in adapter_types:
                        if adapter in metric_key:
                            if layer_idx not in layer_data[adapter]:
                                layer_data[adapter][layer_idx] = {}
                            
                            layer_data[adapter][layer_idx][proj_type] = metric_data
                            break
    
    if not any(layer_data.values()):
        print("No gradient norm data found")
        return
    
    # Determine the range of layers and create matrices
    all_layers = set()
    for adapter_data in layer_data.values():
        all_layers.update(adapter_data.keys())
    
    if not all_layers:
        print("No decoder layers found")
        return
        
    min_layer = min(all_layers)
    max_layer = max(all_layers)
    n_layers = max_layer - min_layer + 1
    n_projections = len(projection_types)
    
    # Create matrices for heatmaps
    heatmap_data = {}
    for adapter in adapter_types:
        heatmap_data[adapter] = np.zeros((n_layers, n_projections))
        
        for layer_idx in range(min_layer, max_layer + 1):
            layer_row = layer_idx - min_layer
            
            if layer_idx in layer_data[adapter]:
                for proj_idx, proj_type in enumerate(projection_types):
                    if proj_type in layer_data[adapter][layer_idx]:
                        grad_data = layer_data[adapter][layer_idx][proj_type]
                        
                        # Calculate absolute cumulative change 
                        if len(grad_data) > 1:
                            cumulative_change = abs(grad_data[-1] - grad_data[0])
                            heatmap_data[adapter][layer_row, proj_idx] = cumulative_change
    
    # Find global min and max for consistent color scaling
    all_values = []
    for adapter in adapter_types:
        if adapter in heatmap_data:
            non_zero_values = heatmap_data[adapter][heatmap_data[adapter] > 1e-6]
            if len(non_zero_values) > 0:
                all_values.extend(non_zero_values)
    
    if all_values:
        vmin, vmax = min(all_values), max(all_values)
    else:
        vmin, vmax = 0, 1
    
    # Create the plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Absolute Cumulative Change in Gradient Norm Mean', fontsize=16, fontweight='bold')
    
    # Plot for each adapter type
    for idx, adapter in enumerate(adapter_types):
        # switch plots
        if idx == 0:
            ax = axes[1]
        else:
            ax = axes[0]
        
        # Create heatmap with consistent color scaling and same colormap
        im = ax.imshow(heatmap_data[adapter], cmap='viridis', aspect='auto', 
                      origin='lower', vmin=vmin, vmax=vmax)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, n_projections, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Customize the plot
        adapter_title = adapter.replace('_ablation', '').replace('_', ' ').title()
        ax.set_title(f'{adapter_title}', fontsize=16, fontweight='bold')

        if idx == 1:
            ax.set_ylabel('Decoder Layer', fontsize=20, fontweight='bold')
        ax.set_xlabel('Projection Type', fontsize=20, fontweight='bold')
        
        # Set ticks and labels
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(i + min_layer) for i in range(n_layers)], fontsize=20)
        
        ax.set_xticks(range(n_projections))
        ax.set_xticklabels(projection_types, rotation=45, ha='right', fontsize=20)
        
        # Add colorbar
        #if idx == 0:
        #    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            #cbar.set_label('Absolute Cumulative Change', rotation=270, labelpad=20, fontsize=20)
        #    cbar.ax.tick_params(labelsize=20)
        
        # Add text annotations for values
        #for i in range(n_layers):
        #    for j in range(n_projections):
        #        value = heatmap_data[adapter][i, j]
        #        if abs(value) > 1e-6:  # Only show non-zero values
                    # Determine text color based on relative intensity
        #            normalized_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0
                    #text_color = 'white' if normalized_value > 0.5 else 'black'
                    #ax.text(j, i, f'{value:.2e}', ha='center', va='center', 
                    #       color=text_color, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    cbar = plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=20)
    
    # Save the plot
    filename = "absolute_cumulative_grad_norm_change_heatmaps.pdf"
    plt.savefig(Path(save_dir) / filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nAbsolute Cumulative Gradient Norm Change Summary:")
    print("=" * 60)
    
    for adapter in adapter_types:
        print(f"\n{adapter.replace('_ablation', '').replace('_', ' ').title()}:")
        data = heatmap_data[adapter]
        non_zero_mask = np.abs(data) > 1e-6
        
        if np.any(non_zero_mask):
            print(f"  Range: {np.min(data[non_zero_mask]):.2e} to {np.max(data[non_zero_mask]):.2e}")
            print(f"  Mean: {np.mean(data[non_zero_mask]):.2e}")
            print(f"  Std: {np.std(data[non_zero_mask]):.2e}")
            
            # Find most changed projections
            max_change_idx = np.unravel_index(np.argmax(data), data.shape)
            max_layer = max_change_idx[0] + min_layer
            max_proj = projection_types[max_change_idx[1]]
            max_value = data[max_change_idx]
            print(f"  Largest change: Layer {max_layer}, {max_proj} ({max_value:.2e})")
        else:
            print("  No significant changes detected")

def parse_metric_key(metric_key, layer_name):
    """
    Helper function to parse metric keys
    """
    # This is a simplified parser - you may need to adjust based on your actual data structure
    parts = metric_key.split('/')
    
    result = {
        'metric': None,
        'adapter': None,
    }
    
    # Look for adapter type
    for part in parts:
        if 'ablation' in part:
            result['adapter'] = part
            break
    
    # Look for metric type
    if 'grad_norm_mean' in metric_key:
        result['metric'] = 'grad_norm_mean'
    
    return result if result['adapter'] and result['metric'] else None

# Example usage:
if __name__ == "__main__":
    #npz_path = "experiment_results/AA-TinyLlama_v1.1-abl_C-boolq-r32-a2/analysis_metrics.npz"
    #layer_name = "base_model.model.model.layers.20.self_attn.v_proj"
    
    #analyze_weight_norms(npz_path, layer_name, save_dir="analysis_plots")
    # Load your data
    analysis_data = load_peft_metrics('experiment_results/AA-TinyLlama_v1.1-abl_A-boolq-r32-a2/analysis_metrics.npz')

    # Create the heatmaps
    plot_cumulative_grad_norm_heatmaps(analysis_data, save_dir="plots")
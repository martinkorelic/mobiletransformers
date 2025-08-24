import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
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
                if adapter in ['up_project', 'down_project', 'intermediate']:
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

def plot_all_weight_norms_comparison(analysis_data, layer_name, save_dir="plots", figsize=(20, 6)):
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
                if adapter in ['up_project', 'down_project', 'intermediate']:
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
    
    fig.suptitle(f'Weight Norm Comparison: {layer_name}', fontsize=16, y=0.95)
    
    for idx, (adapter, data) in enumerate(weight_norm_data.items()):
        ax = axes[idx]
        
        n_windows, n_features = data.shape
        im = ax.imshow(data, cmap='viridis', aspect='auto', origin='lower')
        
        ax.set_title(f'{adapter}/weight_norm_mean', fontsize=12)
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

# Example usage:
if __name__ == "__main__":
    npz_path = "experiment_results/TinyLlama_v1.1-abl_A-boolq-r32-a2/analysis_metrics.npz"
    layer_name = "base_model.model.model.layers.20.self_attn.v_proj"
    
    analyze_weight_norms(npz_path, layer_name, save_dir="analysis_plots")
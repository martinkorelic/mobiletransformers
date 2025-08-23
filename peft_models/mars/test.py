from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from peft_models.mars.config import MarsConfig
from peft_models.mars.model import MarsModel

from .layerv5 import Linear

import torch, time, json
import numpy as np

import torch.utils.bottleneck as bottleneck
from fvcore.nn import FlopCountAnalysis

import matplotlib.pyplot as plt

PEFT_METHOD = "mars"
MAX_LENGTH = 1

def get_mars_linear_layers(model):
    mars_linear_layers = []

    # Recursive function to iterate through model layers
    def recursive_layers(module, prefix=""):
        for child_name, child_module in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            if isinstance(child_module, Linear):
                mars_linear_layers.append((full_name, child_module))
            else:
                recursive_layers(child_module, full_name)

    recursive_layers(model)
    return mars_linear_layers

def visualize_channels(mean_magnitudes, variances, top_k=5, top_massive_limit=128, threshold=50):

    # Identify top-k channels for magnitudes and variances
    top_magnitude_indices = np.argsort(-mean_magnitudes)
    top_variance_indices = np.argsort(-variances)

    # Compute the median activation magnitude
    median_magnitude = np.median(mean_magnitudes)
    print(f"Median magnitude across channels: {median_magnitude}")

    var_magnitude = np.var(mean_magnitudes)

    # Compute the median activation variance
    median_variance = np.median(variances)
    print(f"Median variance across channels: {median_variance}")

    # Identify massive activation channels
    massive_channels = np.where(mean_magnitudes > (median_magnitude + var_magnitude * threshold))[0]
    sorted_massive_channels = massive_channels[np.argsort(-mean_magnitudes[massive_channels])]
    massive_channels = sorted_massive_channels[:top_massive_limit]

    print(f"Amount of massive channels: {len(massive_channels)}")

    # Create the plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    # Plot magnitudes
    axs[0].bar(range(len(mean_magnitudes)), mean_magnitudes, color="blue", label="Magnitudes")
    
    axs[0].scatter(massive_channels, mean_magnitudes[massive_channels], color="orange", label="Massive Activations")
    axs[0].set_title(f"Layer name - Channel Magnitudes")
    axs[0].set_xlabel("Channel Index")
    axs[0].set_ylabel("Mean Magnitude")

    top_non_massive_magnitude_indices = []

    # Annotate top magnitude indices
    for idx in top_magnitude_indices:

        if idx in massive_channels:
            continue
        if len(top_non_massive_magnitude_indices) > top_k:
            break
        top_non_massive_magnitude_indices.append(idx)

        axs[0].annotate(
            f"{idx}",
            (idx, mean_magnitudes[idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="red"
        )

    axs[0].scatter(
        top_non_massive_magnitude_indices, 
        mean_magnitudes[top_non_massive_magnitude_indices], 
        color="red", 
        label="Top Non-Massive Magnitudes"
    )

    for idx in massive_channels:
        axs[0].annotate(f"{idx}", (idx, mean_magnitudes[idx]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="orange")
    
    axs[0].legend()
        
    # Plot variances
    axs[1].bar(range(len(variances)), variances, color="green", label="Variances")
    
    axs[1].set_title(f"Layer name - Channel Variances")
    axs[1].set_xlabel("Channel Index")
    axs[1].set_ylabel("Variance")
    axs[1].legend()

    top_non_massive_variance_indices = []

    # Annotate top variance indices
    for idx in top_variance_indices:

        if idx in massive_channels:
            continue
        if len(top_non_massive_variance_indices) > top_k:
            break
        top_non_massive_variance_indices.append(idx)

        axs[1].annotate(
            f"{idx}",
            (idx, variances[idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="orange"
        )

    axs[1].scatter(
        top_non_massive_variance_indices, 
        variances[top_non_massive_variance_indices], 
        color="orange", 
        label="Top Non-Massive Variances"
    )

    axs[2].hist(mean_magnitudes, bins=len(mean_magnitudes)//2, color="blue", alpha=0.7, edgecolor="black")
    axs[2].set_title(f"Layer name - Mean Magnitude Distribution")
    axs[2].set_xlabel("Mean Magnitude")
    axs[2].set_ylabel("Frequency")

    # Variance distribution
    axs[3].hist(variances, bins=len(variances)//2, color="green", alpha=0.7, edgecolor="black")
    axs[3].set_title(f"Layer name - Variance Distribution")
    axs[3].set_xlabel("Variance")
    axs[3].set_ylabel("Frequency")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    # Print top indices and their values for magnitudes and variances
    print(f"Top {top_k} Magnitudes (Indices:Values):")
    for idx in top_magnitude_indices[:top_k+10]:
        print(f"  Channel {idx}: Magnitude = {mean_magnitudes[idx]:.4f}")
    
    print(f"\nTop {top_k} Variances (Indices:Values):")
    for idx in top_variance_indices[:top_k+10]:
        print(f"  Channel {idx}: Variance = {variances[idx]:.4f}")

def visualize_magnitudes(feature_magnitudes, tokens):
    """
    Visualize heatmap of magnitudes across sequence tokens.

    Args:
    - feature_magnitudes (torch.Tensor): Tensor of shape (sequence_length, num_channels) representing magnitudes.
    - tokens (torch.Tensor): Sequence tokens for which magnitudes are visualized.

    Returns:
    None
    """

    # Check if feature_magnitudes is already a tensor, else convert
    if not isinstance(feature_magnitudes, torch.Tensor):
        feature_magnitudes = torch.tensor(feature_magnitudes)

    # Check if tokens are already provided as tensor, else convert
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)

    # Ensure the length of tokens matches the sequence length
    if len(tokens) < feature_magnitudes.size(1):
        tokens = tokens[1:]  # Remove the first token if needed

    # Plot heatmap of magnitudes across sequence tokens
    plt.figure(figsize=(12, 6))
    print(feature_magnitudes.shape)
    print(len(tokens))
    plt.imshow(feature_magnitudes[0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title("Per-Channel Magnitudes Across Sequence Tokens")
    plt.xlabel("Sequence Token Index")
    plt.ylabel("Channel Index")
    # Adding vertical separators at token indices
    plt.vlines(range(len(tokens)), ymin=0, ymax=feature_magnitudes.size(1), color='red', linestyle='--')

    # Set tokens on x-axis
    plt.xticks(range(len(tokens)), tokens.tolist())
    plt.show()

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Computes the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_channel_stats(channel_stats, metric_name, layer_name="Layer"):
    """
    Visualizes channel statistics in separate subplots for each window.
    
    Args:
        channel_stats (dict): A dictionary containing statistics per metric and channel.
                              Each key is a metric name (e.g., "weight_magnitudes"),
                              and its value is a list of numpy arrays, each representing
                              statistics for one window.
        metric_name (str): Name of the metric to visualize (key in `channel_stats`).
        window_size (int): The number of iterations in each window.
        layer_name (str): Name of the layer for labeling the plots.
    """
    data = channel_stats.get(metric_name, [])
    
    if not data:
        print(f"No data found for metric '{metric_name}'")
        return

    # Stack the data into a 3D array if it's not already
    try:
        data_array = np.stack([np.array(d, dtype=float) for d in data], axis=0)  # Shape: (num_windows, num_channels, other_dims)
    except ValueError:
        print(f"Inconsistent dimensions for metric '{metric_name}' data.")
        return

    
    # Collapse extra dimensions for visualization (if any)
    if data_array.ndim > 2:
        data_array = data_array.mean(axis=1)  # Average over the last axis
    
    channel_data = data_array.T

    num_channels = channel_data.shape[0]

    # Create the boxplot
    plt.figure(figsize=(16, 6))
    plt.boxplot(channel_data.T, showfliers=False, vert=True, patch_artist=True,
                boxprops=dict(facecolor="blue", alpha=0.5), 
                medianprops=dict(color="red", linewidth=2))

    # Label the plot
    plt.title(f"Channel Variance in {metric_name} ({layer_name})")
    plt.xlabel("Channel Index")
    plt.ylabel(f"{metric_name} Values")
    plt.grid(True, alpha=0.3)

    # Add x-tick labels
    plt.xticks(ticks=range(1, num_channels + 1), labels=range(num_channels), rotation=90)

    plt.tight_layout()
    plt.show()

def calculate_weight_norm_metrics(weight_array):
    """
    Calculate metrics for the weight norms of a layer.

    Args:
        weight_array (np.ndarray): Array of shape (steps, channels) representing the L1 norms of weights over time.

    Returns:
        dict: Dictionary containing the top 20 most frequent and least frequent channels, and their frequencies.
    """
    num_channels = weight_array.shape[1]
    channel_frequencies = np.zeros(num_channels)

    # For each time step, find the top 5% largest and smallest channels
    for step in range(weight_array.shape[0]):
        top_5_percent_indices = np.argsort(weight_array[step])[-int(num_channels * 0.05):]
        bottom_5_percent_indices = np.argsort(weight_array[step])[:int(num_channels * 0.05)]

        channel_frequencies[top_5_percent_indices] += 1  # Increment frequency for top channels
        channel_frequencies[bottom_5_percent_indices] -= 1  # Decrement frequency for bottom channels

    # Top 20 most frequent and least frequent channels
    top_frequent_indices = np.argsort(channel_frequencies)[-20:][::-1]
    bottom_frequent_indices = np.argsort(channel_frequencies)[:20]
    top_frequent_counts = channel_frequencies[top_frequent_indices]
    bottom_frequent_counts = channel_frequencies[bottom_frequent_indices]

    return {
        "top_frequent_channels": {
            "indices": top_frequent_indices.tolist(),
            "counts": top_frequent_counts.tolist()
        },
        "bottom_frequent_channels": {
            "indices": bottom_frequent_indices.tolist(),
            "counts": bottom_frequent_counts.tolist()
        }
    }


def calculate_layer_metrics(data_array):
    """
    Calculate metrics for a given layer's data array.

    Args:
        data_array (np.ndarray): Array of shape (steps, channels)

    Returns:
        dict: Dictionary containing the calculated metrics
    """
    num_channels = data_array.shape[1]

    # Calculate top 5% threshold indices for each step
    threshold_indices_top = np.argsort(data_array, axis=1)[:, -int(num_channels * 0.05):]
    threshold_indices_bottom = np.argsort(data_array, axis=1)[:, :int(num_channels * 0.05)]

    # Count frequency of each channel appearing in top and bottom 5%
    channel_frequencies_top = np.zeros(num_channels)
    channel_frequencies_bottom = np.zeros(num_channels)

    for indices_top, indices_bottom in zip(threshold_indices_top, threshold_indices_bottom):
        channel_frequencies_top[indices_top] += 1
        channel_frequencies_bottom[indices_bottom] += 1

    # Get top 20 most frequent channels
    top_frequent_indices = np.argsort(channel_frequencies_top)[-20:][::-1]
    bottom_frequent_indices = np.argsort(channel_frequencies_bottom)[-20:][::-1]
    top_frequent_counts = channel_frequencies_top[top_frequent_indices]
    bottom_frequent_counts = channel_frequencies_bottom[bottom_frequent_indices]

    # Calculate variance for each channel across all steps
    channel_variances = np.var(data_array, axis=0)
    top_variance_indices = np.argsort(channel_variances)[-20:][::-1]
    bottom_variance_indices = np.argsort(channel_variances)[:20]
    top_variance_values = channel_variances[top_variance_indices]
    bottom_variance_values = channel_variances[bottom_variance_indices]

    return {
        "top_frequent_channels": {
            "indices": top_frequent_indices.tolist(),
            "counts": top_frequent_counts.tolist()
        },
        "bottom_frequent_channels": {
            "indices": bottom_frequent_indices.tolist(),
            "counts": bottom_frequent_counts.tolist()
        },
        "top_variance_channels": {
            "indices": top_variance_indices.tolist(),
            "variances": top_variance_values.tolist()
        },
        "bottom_variance_channels": {
            "indices": bottom_variance_indices.tolist(),
            "variances": bottom_variance_values.tolist()
        }
    }

def calculate_scaling_metrics(scaling_array):
    """
    Calculate top 20 highest and lowest scaling factors.

    Args:
        scaling_array (np.ndarray): Array of shape (steps, channels) for scaling factors over time.

    Returns:
        dict: Dictionary containing the top 20 highest and lowest scaling factors and their channels.
    """
    avg_scaling_factors = np.mean(scaling_array, axis=0)
    top_scaling_indices = np.argsort(avg_scaling_factors)[-20:][::-1]
    bottom_scaling_indices = np.argsort(avg_scaling_factors)[:20]
    return {
        "top_scaling_channels": {
            "indices": top_scaling_indices.tolist(),
            "scaling_factors": avg_scaling_factors[top_scaling_indices].tolist()
        },
        "bottom_scaling_channels": {
            "indices": bottom_scaling_indices.tolist(),
            "scaling_factors": avg_scaling_factors[bottom_scaling_indices].tolist()
        }
    }

def plot_gradient_norms(trainable_layers, keywords, save_dir="gradient_plots"):
    """
    Plot gradient norms over time for specific groups of layers and save the plots.
    
    Parameters:
    - keywords: List of keywords to group layers (e.g., ["q_proj", "k_proj"]).
    - trainable_layers: List of tuples (layer_name, layer_object) with gradient norms in `layer.channel_stats["gradient_norm"]`.
    - save_dir: Directory to save the plots. Created if it doesn't exist.
    """
    import os
    
    # Create directory to save plots if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define a list of 20 nice colors
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", 'springgreen', 'maroon'
    ]

    # Loop over each keyword to create plots
    for keyword in keywords:
        # Filter layers matching the current keyword
        matching_layers = [
            (layer_name, layer.channel_stats["gradient_norm"])
            for layer_name, layer in trainable_layers
            if keyword in layer_name
        ]
        
        if not matching_layers:
            print(f"No layers found for keyword '{keyword}', skipping.")
            continue

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(20, 10))

        # Set title and labels
        ax.set_title(f"Gradient Norms Over Time - {keyword}", fontsize=16)
        ax.set_xlabel("Timesteps", fontsize=14)
        ax.set_ylabel("Gradient Norm", fontsize=14)

        # Plot gradient norms for each matching layer
        for idx, (layer_name, gradient_norms) in enumerate(matching_layers):
            color = colors[idx % len(colors)]
            ax.plot(range(len(gradient_norms)), gradient_norms, label=layer_name, color=color)

        # Add a legend outside the plot
        fig.legend(
            title="Decoder Layer",
            fontsize=12,
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            ncol=4,
            bbox_transform=fig.transFigure
        )

        # Add grid and adjust layout
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        # Save the plot
        save_path = os.path.join(save_dir, f"{keyword}_gradient_norms.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved for keyword '{keyword}' at: {save_path}")

        # Clear the figure to avoid overlapping
        plt.close(fig)

def visualize_layer_metrics_with_changes(trainable_layers, save_dir="channel_visualizations", plot_grad_norms=False):
    """
    Visualizes channel statistics for each layer, focusing on highlighting the top 20% of lowest-change channels
    in timestep differences for output magnitudes.
    
    Produces:
    1. Heatmap of timestep differences in output magnitudes.
    2. Visualization of top 20% of lowest-change channels in timestep differences.

    Args:
        trainable_layers (list): List of (layer_name, layer_module) tuples.
        save_dir (str): Directory to save the figures and metrics.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if plot_grad_norms:
        plot_gradient_norms(trainable_layers)

    all_layer_metrics = []

    for layer_name, layer in trainable_layers:
        channel_stats = layer.channel_stats
        layer_metrics = {"layer_name": layer_name}

        # Prepare figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(layer_name, fontsize=16)

        # Compute timestep differences from out_magnitudes
        out_magnitudes = np.array(channel_stats.get("out_magnitudes", []))  # Shape: (timesteps, channels)
        if out_magnitudes.size > 0 and out_magnitudes.shape[0] > 1:
            timestep_differences = np.abs(np.diff(out_magnitudes, axis=0))  # Shape: (timesteps - 1, channels)

            # Plot 1: Heatmap of timestep differences
            ax_heatmap = axes[0]
            heatmap = ax_heatmap.imshow(timestep_differences, aspect="auto", cmap="coolwarm", origin="lower")
            plt.colorbar(heatmap, ax=ax_heatmap)
            ax_heatmap.set_title("Timestep Differences in Output Magnitudes")
            ax_heatmap.set_xlabel("Channels")
            ax_heatmap.set_ylabel("Timesteps")
            ax_heatmap.grid(True, alpha=0.3)

            layer_metrics["timestep_differences"] = timestep_differences.tolist()

            # Plot 2: Top 20% of lowest-change channels highlighted
            num_channels = timestep_differences.shape[1]
            num_low_change_channels = int(num_channels * 0.2)

            indicator_matrix = np.zeros_like(timestep_differences, dtype=int)  # Shape: (timesteps - 1, channels)

            for t, diffs in enumerate(timestep_differences):
                low_change_indices = np.argsort(diffs)[:num_low_change_channels]  # Get top 20% of lowest differences
                indicator_matrix[t, low_change_indices] = 1  # Mark these channels in the indicator matrix

            ax_low_change = axes[1]
            ax_low_change.imshow(indicator_matrix, aspect="auto", cmap="Greys", interpolation="none", origin="lower")
            ax_low_change.set_title("Top 20% Lowest-Change Channels (Per Timestep)")
            ax_low_change.set_xlabel("Channels")
            ax_low_change.set_ylabel("Timesteps")
            ax_low_change.grid(True, alpha=0.3)

            layer_metrics["low_change_indicators"] = indicator_matrix.tolist()

        all_layer_metrics.append(layer_metrics)

        # Save the figure
        file_name = os.path.join(save_dir, f"{layer_name.replace('.', '_')}_low_change_plots.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(file_name)
        plt.close()
        print(f"Saved visualization for layer '{layer_name}' at '{file_name}'")

    # Save all metrics to a JSON file
    metrics_file = os.path.join(save_dir, "all_layer_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_layer_metrics, f, indent=4)
    print(f"Saved all layer metrics at '{metrics_file}'")



def visualize_layer_metrics(trainable_layers, lora_targets=[], save_dir="channel_visualizations", plot_grad_norms=False):
    """
    Visualizes channel statistics for each layer and saves all metrics to a single JSON file.
    Produces:
    1. Boxplots for input and output magnitudes for channels.
    2. A visualization of top 5% highest magnitude channels per timestep.
    3. Heatmaps for l1_weight_down_project and l1_weight_up_project.
    4. Saves metrics for top 20 most frequent channels in l1 norms for down_project and up_project.
    
    Args:
        trainable_layers (list): List of (layer_name, layer_module) tuples.
        save_dir (str): Directory to save the figures and metrics.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if plot_grad_norms:
        plot_gradient_norms(trainable_layers, keywords=lora_targets)
    
    # Initialize list to store all layer metrics
    all_layer_metrics = []
    
    for layer_name, layer in trainable_layers:
        channel_stats = layer.channel_stats
        layer_metrics = {"layer_name": layer_name}
        
        # Prepare figure with 6 subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle(layer_name, fontsize=16)
        
        # Boxplots for input and output magnitudes
        metric_names = ["in_magnitudes", "out_magnitudes"]
        for i, metric in enumerate(metric_names):
            data = channel_stats.get(metric, [])
            
            if not data:
                print(f"No data found for metric '{metric}' in layer {layer_name}")
                continue
                
            # Stack the data
            try:
                data_array = np.stack([np.array(d, dtype=float) for d in data], axis=0)
            except ValueError:
                print(f"Inconsistent dimensions for metric '{metric}' data in layer {layer_name}.")
                continue
            
            # Collapse extra dimensions if necessary
            if data_array.ndim > 2:
                data_array = data_array.mean(axis=2)  # Average over extra dimensions
            
            # Calculate metrics for this data
            layer_metrics[metric] = calculate_layer_metrics(data_array)
            
            # Plot boxplot with dynamic tick spacing
            ax = axes[0, i]
            num_channels = data_array.shape[1]
            ax.boxplot(data_array, positions=range(num_channels), showfliers=False)
            
            # Create approximately 10 ticks based on the number of channels
            num_ticks = 10
            step_size = max(1, num_channels // num_ticks)
            tick_positions = np.arange(0, num_channels, step_size)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions)
            
            ax.set_title(f"Boxplot of {metric.replace('_', ' ').title()}")
            ax.set_xlabel("Channel Index")
            ax.set_ylabel("Magnitude")
            ax.grid(True, alpha=0.3)
            
            # Plot top 5% visualization with transposed axes
            ax_top_5 = axes[1, i]
            threshold_indices = np.argsort(data_array, axis=1)[:, -int(data_array.shape[1] * 0.05):]
            indicator = np.zeros_like(data_array, dtype=int)
            for step, indices in enumerate(threshold_indices):
                indicator[step, indices] = 1
            
            # Display the indicator matrix with y-axis reversed and no interpolation
            ax_top_5.imshow(indicator, aspect="auto", cmap="Greys", interpolation="none", origin="lower")
            
            ax_top_5.set_title(f"Top 5% Channels Over Steps ({metric.replace('_', ' ').title()})")
            ax_top_5.set_ylabel("Steps")
            ax_top_5.set_xlabel("Channels")
            ax_top_5.grid(True, alpha=0.3)
        
        # Heatmaps for l1_weight_down_project and l1_weight_up_project
        weight_metrics = ["l1_weight_down_project", "l1_weight_up_project"]
        for i, metric in enumerate(weight_metrics):
            data = channel_stats.get(metric, [])
            
            if not data:
                print(f"No data found for metric '{metric}' in layer {layer_name}")
                continue
            
            try:
                data_array = np.stack([np.array(d, dtype=float) for d in data], axis=0)
            except ValueError:
                print(f"Inconsistent dimensions for metric '{metric}' data in layer {layer_name}.")
                continue
            
            # Calculate top 5% channels over all timesteps
            top_channels_metric = calculate_weight_norm_metrics(data_array)
            layer_metrics[metric] = top_channels_metric
            
            # Plot heatmap
            ax_heatmap = axes[i, 2]
            heatmap = ax_heatmap.imshow(data_array, aspect="auto", cmap="viridis", interpolation="none", origin="lower")
            plt.colorbar(heatmap, ax=ax_heatmap)
            
            ax_heatmap.set_title(f"Heatmap of {metric.replace('_', ' ').title()}")
            ax_heatmap.set_ylabel("Steps")
            ax_heatmap.set_xlabel("Channels")
            ax_heatmap.grid(True, alpha=0.3)
        
        # Scaling factors
        #base_scaling = np.stack(channel_stats["base_scaling"])
        adapter_scaling = np.stack(channel_stats["adapter_scaling"])

        # Histogram for adapter_scaling (last timestep)
        if "adapter_scaling" in channel_stats:
            adapter_scaling = np.array(channel_stats["adapter_scaling"])
            last_timestep_scaling = adapter_scaling[-1, :]  # Last timestep scaling factors
            
            # Calculate statistics
            mean_scaling = np.mean(last_timestep_scaling)
            variance_scaling = np.var(last_timestep_scaling)
            
            # Plot histogram
            ax_hist = axes[0, 3]
            ax_hist.hist(last_timestep_scaling, bins=100, color="blue", alpha=0.7, edgecolor="black")
            ax_hist.set_title("Adapter Scaling Factors Distribution (Last Timestep)")
            ax_hist.set_xlabel("Scaling Factor")
            ax_hist.set_ylabel("Frequency")
            ax_hist.grid(True, alpha=0.3)
            
            # Annotate mean and variance
            ax_hist.axvline(mean_scaling, color="red", linestyle="--", label=f"Mean: {mean_scaling:.4f}")
            ax_hist.legend()
            ax_hist.text(
                0.95, 0.95, f"Variance: {variance_scaling:.4f}", 
                transform=ax_hist.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right'
            )

        ax_adapter = axes[1, 3]
        heatmap = ax_adapter.imshow(adapter_scaling, aspect="auto", cmap="viridis", origin="lower", interpolation="none")
        plt.colorbar(heatmap, ax=ax_adapter)
        ax_adapter.set_title(f"Adapter Scaling Factors")
        ax_adapter.set_xlabel("Channels")
        ax_adapter.set_ylabel("Steps")
        
        # Add this layer's metrics to the list
        all_layer_metrics.append(layer_metrics)
            
        # Save the figure
        file_name = os.path.join(save_dir, f"{layer_name.replace('.', '_')}_plots.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(file_name)
        plt.close()
        print(f"Saved visualization for layer '{layer_name}' at '{file_name}'")
    
    # Save all metrics to a single JSON file
    metrics_file = os.path.join(save_dir, "all_layer_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_layer_metrics, f, indent=4)
    print(f"Saved all layer metrics at '{metrics_file}'")

if __name__ == "__main__":

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if PEFT_METHOD == "mars":
        mars_config = MarsConfig(
            peft_type="MARS",
            ranks=[32, 16],
            lora_alphas=[1],  # Scaling factor
            target_modules=["q_proj"],  # Target specific model layers
            task_type="CAUSAL_LM"
        )

        peft_model = MarsModel(base_model, mars_config)
    elif PEFT_METHOD == "lora":
        # Prepare the model
        lora_config = LoraConfig(
                r=16,
                target_modules=["q_proj"],
                task_type="CAUSAL_LM",
            )
        peft_model = get_peft_model(base_model, lora_config)

    peft_model.train()
    trainable_layers = [name for name, param in peft_model.named_parameters() if param.requires_grad]
    print(trainable_layers)
    print(count_trainable_parameters(peft_model))
    
    # Set parameters for generation
      # Maximum length of the generated text
    num_return_sequences = 1  # Number of generated sequences (if you want multiple completions)
    eos_token_id = tokenizer.eos_token_id  # End-of-sequence token (if available)

    input_text = ["Some crumpled newspaper and a lighter along with a bucket of water. A man begins demonstrating how to place the newspaper and twigs on top of that."]

    inputs = tokenizer(input_text, return_tensors="pt", padding=True)


    # Begin generation loop
    generated_text = input_text
    input_ids = inputs["input_ids"]
    attention_ids = inputs["attention_mask"]

    # Flop analysis
    #flops = FlopCountAnalysis(peft_model, inputs=(input_ids, attention_ids))
    #print(f"FLOPs: {flops.total()}")

    peft_model.eval()
    with torch.no_grad():  # Disable gradients for inference
        for _ in range(MAX_LENGTH):
            # Perform a forward pass to predict the next token

            time_start = time.time()
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_ids)
            end_time = time.time() - time_start
            print(f"Inference time: {end_time:.6f}")

            # Extract logits (predicted scores for the next token)
            logits = outputs.logits

            # Get the predicted token (argmax over the last token's logits)
            next_token_id = torch.argmax(logits[0, -1]).item()

            # Decode the predicted token into text
            predicted_token = tokenizer.decode(next_token_id)

            # Append the predicted token to the generated text
            generated_text += predicted_token

            print(generated_text)

            # Append the predicted token to the input_ids for the next iteration
            #input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=input_ids.device)], dim=1)

            # Stop generating if the end-of-sequence token is generated
            if next_token_id == eos_token_id:
                break

    # Display visualizations
            
    mars_linear_layers = get_mars_linear_layers(peft_model)

    visualize_channels(mars_linear_layers[15].in_magnitudes, mars_linear_layers[15].in_variances)
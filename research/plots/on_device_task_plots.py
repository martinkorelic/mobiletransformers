from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy_comparison(file_paths, custom_names, output_filename='accuracy_comparison.pdf'):
    """
    Create horizontal bar plots comparing accuracy across categories for two JSON files.
    
    Args:
        file_paths (list): List of paths to JSON files (should be exactly 2)
        custom_names (list): List of custom names for the datasets (should be exactly 2)
        output_filename (str): Name of the output PDF file
    
    Returns:
        None: Saves the plot as PDF
    """
    if len(file_paths) != 2 or len(custom_names) != 2:
        raise ValueError("This function requires exactly 2 files and 2 custom names")
    
    # Read the JSON files
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data.append(json.load(file))
    
    # Extract per_category_accuracy from both files
    categories_data = []
    all_categories = set()
    
    for dataset in data:
        if 'per_category_accuracy' in dataset:
            categories_data.append(dataset['per_category_accuracy'])
            all_categories.update(dataset['per_category_accuracy'].keys())
        else:
            raise ValueError("JSON file must contain 'per_category_accuracy' field")
    
    # Sort categories alphabetically for consistent ordering
    categories = sorted(list(all_categories))
    
    # Prepare data for plotting
    dataset1_values = [categories_data[0].get(cat, 0) for cat in categories]
    dataset2_values = [categories_data[1].get(cat, 0) for cat in categories]
    
    # Convert to percentages
    dataset1_values = [val * 100 for val in dataset1_values]
    dataset2_values = [val * 100 for val in dataset2_values]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(13, 8))
    
    # Define colors and hatches
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
    hatches = ['///', '|||']
    
    # Calculate bar positions
    y_pos = np.arange(len(categories))
    bar_height = 0.35
    
    # Create horizontal bars
    bars1 = ax.barh(y_pos - bar_height/2, dataset1_values, bar_height, 
                    label=custom_names[0], color=colors[0], alpha=0.8, 
                    hatch=hatches[0], edgecolor='black', linewidth=0.5)
    
    bars2 = ax.barh(y_pos + bar_height/2, dataset2_values, bar_height,
                    label=custom_names[1], color=colors[1], alpha=0.8,
                    hatch=hatches[1], edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, dataset1_values, dataset2_values)):
        # Label for first dataset
        ax.text(val1 + 1, bar1.get_y() + bar1.get_height()/2, f'{val1:.1f}%',
                va='center', ha='left', fontweight='bold', fontsize=14, color=colors[0])
        
        # Label for second dataset
        ax.text(val2 + 1, bar2.get_y() + bar2.get_height()/2, f'{val2:.1f}%',
                va='center', ha='left', fontweight='bold', fontsize=14, color=colors[1])
    
    # Customize the plot
    ax.set_xlabel('Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Categories', fontsize=18, fontweight='bold')
    ax.set_title('Accuracy Comparison by Category', fontsize=16, fontweight='bold', pad=20)
    
    # Set category labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=14)
    
    # Set x-axis limits with some padding
    max_val = max(max(dataset1_values), max(dataset2_values))
    ax.set_xlim(0, max_val * 1.15)
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend outside plot area on the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, framealpha=0.9)
    
    # Add overall accuracy outside plot area on the right
    overall_text = []
    for i, (dataset, name) in enumerate(zip(data, custom_names)):
        if 'results' in dataset:
            overall_acc = dataset['results'] * 100
            overall_text.append(f'{name}: {overall_acc:.1f}%')
    
    if overall_text:
        # Create colored text for overall accuracy
        overall_info = 'Overall Accuracy:'
        ax.text(1.05, 0.5, overall_info, transform=ax.transAxes,
                verticalalignment='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Add colored entries below the header
        for i, (dataset, name) in enumerate(zip(data, custom_names)):
            if 'results' in dataset:
                overall_acc = dataset['results'] * 100
                # Position each entry below the previous one
                y_offset = 0.5 - 0.08 * (i + 1)
                ax.text(1.05, y_offset, f'{name}: {overall_acc:.1f}%', 
                       transform=ax.transAxes, verticalalignment='center', 
                       fontsize=14, fontweight='bold', color=colors[i],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=colors[i], alpha=0.9))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Plot saved as: {output_filename}")

def plot_training_metrics(file_path, output_filename='training_metrics.pdf', name="temp"):
    """
    Plot training loss over steps and step duration over time.
    
    Args:
        file_path (str): Path to JSON file containing training metrics
        output_filename (str): Name of the output PDF file
    
    Returns:
        None: Saves the plot as PDF
    """
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract step metrics
    if 'step_metrics' not in data:
        raise ValueError("JSON file must contain 'step_metrics' field")
    
    step_metrics = data['step_metrics']
    
    # Extract data for plotting
    steps = [metric['step'] for metric in step_metrics]
    losses = [metric['loss'] for metric in step_metrics]
    durations = [metric['stepDurationMs'] / 1000 for metric in step_metrics]  # Convert to seconds
    
    # Create figure with 2 subplots in rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. Training Loss over Steps
    ax1.plot(steps, losses, color='#2E8B57', linewidth=2.5, alpha=0.8)
    ax1.fill_between(steps, losses, alpha=0.2, color='#2E8B57')
    
    # Add final loss annotation in top right
    final_loss = losses[-1]
    ax1.text(0.98, 0.95, f'Final Loss: {final_loss:.4f}', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold', 
             color='#2E8B57', verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E8B57', alpha=0.9))
    
    ax1.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Loss Progress', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # 2. Step Duration over Time (Line plot)
    avg_duration = np.mean(durations)
    ax2.plot(steps, durations, color='#FF6347', linewidth=2, alpha=0.8, marker='o', markersize=3)
    ax2.axhline(avg_duration, color='red', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_duration:.1f}s')
    
    ax2.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Duration (seconds)', fontsize=14, fontweight='bold')
    ax2.set_title('Step Duration Over Time', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    
    # Set main title
    filename = os.path.basename(file_path).replace('.json', '')
    fig.suptitle(f'On-device Training Metrics - {name}', fontsize=18, fontweight='bold')
    
    # Adjust spacing between subplots and add more room for title
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # More room for main title (was 0.93)
    
    # Save and show
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Training metrics plot saved as: {output_filename}")

# Plot accuracy Mini PersonalQA
#plot_accuracy_comparison(['experiment_results/train-qwen2-personalqa-mobile/base_eval_results.json', 'experiment_results/train-qwen2-personalqa-mobile/eval_results.json'], 
#                         ['Base model', 'On-device fine-tuned model'])

# Mini PersonalQA training metrics
#plot_training_metrics('experiment_results/train-qwen2-personalqa-mobile/training_logs.json', name="Mini PersonalQA")

# Mini Recommendation training metrics   
#plot_training_metrics('experiment_results/train-qwen2-recommendation-mobile/training_logs.json', name="Mini Recommendation")
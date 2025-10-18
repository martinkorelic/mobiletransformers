import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ==================== RAM USAGE PLOTTING FUNCTIONS ====================

def plot_ram_usage_filled(file_paths, custom_names=None, model_loading_time=25):
    """
    Plot RAM usage with filled areas under each line (non-stacked).
    
    Args:
        file_paths (str or list): Path to a single JSON file or list of paths to JSON files
        custom_names (list, optional): Custom names for legend entries. If None, uses filenames
        model_loading_time (int, optional): Duration of model loading phase in seconds (default: 25)
    
    Returns:
        None: Displays the plot
    """
    # Handle single file path or list of file paths
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    # Handle custom names
    if custom_names is None:
        custom_names = [os.path.basename(fp).replace('.json', '') for fp in file_paths]
    elif len(custom_names) != len(file_paths):
        raise ValueError(f"Number of custom names ({len(custom_names)}) must match number of files ({len(file_paths)})")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Colors for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    hatches = ["///", "***", "|||", "\\\\\\", "...", "+++", "xxx", "ooo"]
    
    # Find global time range to ensure all plots start from 0
    global_max_time = 0
    
    # Store legend elements and peak values
    legend_elements = []
    peak_values = []
    
    for i, file_path in enumerate(file_paths):
        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Extract timestamps and RAM usage
        timestamps = [int(entry['timestamp_sec']) for entry in data]
        ram_usage = [entry['ram_usage_mb'] for entry in data]
        
        # Convert timestamps to start from 0
        min_timestamp = min(timestamps)
        normalized_timestamps = [ts - min_timestamp for ts in timestamps]
        
        # Update global max time
        global_max_time = max(global_max_time, max(normalized_timestamps))
        
        # Get custom name for label
        custom_name = custom_names[i]
        color = colors[i % len(colors)]
        hatch = hatches[i % len(hatches)]
        
        # Find peak RAM usage
        peak_ram = max(ram_usage)
        peak_values.append((peak_ram, color, custom_name))
        
        # Fill area under the curve
        plt.fill_between(normalized_timestamps, ram_usage, alpha=1, color=color)
        plt.fill_between(normalized_timestamps, ram_usage, alpha=1, color="gray", 
                        hatch=hatch, linewidth=0.5, facecolor="none")
        
        # Create legend element - USE FACECOLOR instead of COLOR
        import matplotlib.patches as mpatches
        legend_elements.append(mpatches.Patch(facecolor=color, hatch=hatch, 
                                            edgecolor='gray', label=custom_name))
    
    # Add horizontal lines for peak values with staggered text labels
    for i, (peak_ram, color, custom_name) in enumerate(peak_values):
        plt.axhline(y=peak_ram, color=color, linestyle='--', linewidth=2, alpha=0.7, 
                   label='_nolegend_')
        
        # Calculate staggered x position for text
        text_x = 10 + (i * 20)  # Start at x=10, then +10 for each subsequent line
        
        # Add text label for peak value at staggered position
        plt.text(text_x, peak_ram + 20, f'{peak_ram:.0f} MB', 
                color=color, fontweight='bold', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, alpha=0.9), fontsize=16)
    
    # Add model loading time indicator - horizontal line at y=2000 from 0 to model_loading_time
    if model_loading_time > 0 and model_loading_time <= global_max_time:
        # Horizontal line from 0 to model_loading_time at y=2000 (not in legend)
        plt.plot([0, model_loading_time], [2000, 2000], 
                color='red', linewidth=3, alpha=0.8, label='_nolegend_')
        
        # Add text label on the horizontal line
        plt.text(model_loading_time/2, 2050, 'Model Loading', fontsize=14,
                horizontalalignment='center', color='red', fontweight='bold')
    
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Customize the plot
    plt.title('On-device RAM Usage', fontsize=18, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=18)
    plt.ylabel('RAM Usage (MB)', fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Create legend with colored patches and hatches
    plt.legend(handles=legend_elements, loc='lower center', fontsize=18)
    
    # Force x-axis to start from 0 and y-axis to start from 1000
    plt.xlim(left=0, right=global_max_time * 1.02)
    plt.ylim(bottom=1000)  # Start y-axis from 1000
    
    # Add some styling
    plt.tight_layout()
    plt.savefig('ram_usage_plot.pdf', format='pdf', bbox_inches='tight', dpi=150)

def plot_cpu_usage_individual_subplots(file_path, metric='cpu_usage_percent', name='cpu'):
    """
    Plot CPU usage for each core in separate thin subplots.
    
    Args:
        file_path (str): Path to a single JSON file
        metric (str): Which metric to plot ('cpu_usage_percent', 'cpu_time_ms')
    
    Returns:
        None: Displays the plot
    """
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Group data by CPU
    cpu_data = defaultdict(list)
    
    for entry in data:
        cpu_id = entry['cpu']
        cpu_data[cpu_id].append({
            'timestamp': int(entry['timestamp_sec']),
            'value': entry[metric]
        })
    
    # Sort CPUs numerically and ensure we have exactly 7 CPUs (0-6)
    cpu_ids = sorted(cpu_data.keys(), key=lambda x: int(x))
    expected_cpus = ['0', '1', '2', '3', '4', '5', '6']
    
    # Colors for different CPUs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2']
    
    # Create 7 thin subplots stacked vertically
    fig, axes = plt.subplots(7, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f'Individual CPU Core Usage - {name}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each CPU on its own subplot
    for i, cpu_id in enumerate(expected_cpus):
        ax = axes[i]
        color = colors[i]
        
        if cpu_id in cpu_data:
            # Sort by timestamp
            cpu_data[cpu_id].sort(key=lambda x: x['timestamp'])
            
            timestamps = [entry['timestamp'] for entry in cpu_data[cpu_id]]
            values = [entry['value'] for entry in cpu_data[cpu_id]]
            
            # Normalize timestamps to start from 0
            if timestamps:
                min_timestamp = min(timestamps)
                normalized_timestamps = [ts - min_timestamp for ts in timestamps]
                
                # Plot line with filled area
                ax.plot(normalized_timestamps, values, color=color, linewidth=2, alpha=0.8)
                ax.fill_between(normalized_timestamps, values, alpha=0.3, color=color)
        
        # Customize each subplot
        ax.set_ylabel(f'CPU {cpu_id}', fontsize=12, fontweight='bold', rotation=0, 
                     ha='right', va='center')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Make subplots really thin
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=14)
        
        # Only show x-axis label on bottom subplot
        if i == 6:  # Last subplot
            ax.set_xlabel('Time (seconds)', fontsize=12)
        
        # Set consistent y-axis range based on metric
        if metric == 'cpu_usage_percent':
            ax.set_ylim(0, 100)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust layout to make subplots thin and remove gaps
    plt.subplots_adjust(hspace=0.1)  # Minimal space between subplots
    plt.tight_layout()
    plt.savefig('cpu_usage_plot.pdf', format='pdf', bbox_inches='tight', dpi=300)


def plot_temperature_individual_subplots(file_path, name="temp"):
    """
    Plot temperature data for each thermal zone in separate thin subplots.
    
    Args:
        file_path (str): Path to a single JSON file
    
    Returns:
        None: Displays the plot
    """
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Group data by thermal zone
    thermal_data = defaultdict(list)
    
    for entry in data:
        thermal_zone = entry['thermal_zone']
        thermal_data[thermal_zone].append({
            'timestamp': int(entry['timestamp_sec']),
            'temperature': entry['temperature_celsius']
        })
    
    # Expected thermal zones in order
    expected_zones = ['BIG Temperature', 'MID Temperature', 'LITTLE Temperature']
    zone_labels = ['Big Cores', 'Middle Cores', 'Little Cores']
    
    # Colors for different thermal zones
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    
    # Create 3 thin subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'CPU Core Temperatures - {name}', 
                 fontsize=16, fontweight='bold')
    
    # Find global time range for consistent x-axis
    all_timestamps = []
    for zone_data in thermal_data.values():
        all_timestamps.extend([entry['timestamp'] for entry in zone_data])
    
    if all_timestamps:
        global_min_time = min(all_timestamps)
        
        # Plot each thermal zone on its own subplot
        for i, (zone, label) in enumerate(zip(expected_zones, zone_labels)):
            ax = axes[i]
            color = colors[i]
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            if zone in thermal_data:
                # Sort by timestamp
                thermal_data[zone].sort(key=lambda x: x['timestamp'])
                
                timestamps = [entry['timestamp'] for entry in thermal_data[zone]]
                temperatures = [entry['temperature'] for entry in thermal_data[zone]]
                
                # Normalize timestamps to start from 0
                if timestamps:
                    normalized_timestamps = [ts - global_min_time for ts in timestamps]
                    
                    # Plot line with filled area
                    ax.plot(normalized_timestamps, temperatures, color=color, linewidth=2, alpha=0.8)
                    ax.fill_between(normalized_timestamps, temperatures, alpha=0.3, color=color)
                    
                    # Calculate average temperature
                    avg_temp = sum(temperatures) / len(temperatures)
                    
                    # Add average temperature annotation
                    ax.text(0.02, 0.95, f'Avg: {avg_temp:.1f}°C', 
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor=color, alpha=0.9),
                           verticalalignment='top', color=color)
                    
                    # Add peak temperature annotation
                    peak_temp = max(temperatures)
                    peak_time = normalized_timestamps[temperatures.index(peak_temp)]
                    ax.annotate(f'Peak: {peak_temp:.1f}°C', 
                               xy=(peak_time, peak_temp), 
                               xytext=(peak_time + 5, peak_temp + 2),
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=color, alpha=0.8),
                               fontsize=9, fontweight='bold', color=color)
            else:
                # If no data for this zone, show empty plot
                ax.text(0.5, 0.5, f'No data for {label}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray', style='italic')
            
            # Customize each subplot with better labels
            ax.set_ylabel(f'{label}\n(°C)', fontsize=12, fontweight='bold', rotation=0, 
                         ha='right', va='center')
            ax.grid(True, alpha=0.3)
            
            # Set reasonable temperature range (adjust based on your data)
            ax.set_ylim(60, 100)  # Typical CPU temperature range
            
            # Make subplots really thin
            ax.tick_params(axis='y', labelsize=14)
            ax.tick_params(axis='x', labelsize=14)
            
            # Only show x-axis label on bottom subplot
            if i == 2:  # Last subplot
                ax.set_xlabel('Time (seconds)', fontsize=14)
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add horizontal line at critical temperature (e.g., 85°C)
            ax.axhline(y=85, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.98, 85, '85°C Critical', transform=ax.get_yaxis_transform(),
                   fontsize=12, color='red', va='bottom', ha='left')
    
    # Adjust layout to make subplots thin and remove gaps
    plt.subplots_adjust(hspace=0.15)  # Minimal space between subplots
    plt.tight_layout()
    plt.savefig('temp_usage_plot.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Temperature plot

#plot_temperature_individual_subplots('experiment_results/on-device-benchmarks/mars32-opt4-temp-usage.json', name="MARS Q-OPT1 ($r$ = 32)")

# CPU usage plot
#plot_cpu_usage_individual_subplots('experiment_results/on-device-benchmarks/mars32-opt4-cpu-usage.json', metric='cpu_usage_percent', name="MARS Q-OPT1 ($r$ = 32)")

# RAM usage plots
#plot_ram_usage_filled(['experiment_results/on-device-benchmarks/lora32-mem-usage.json', 'experiment_results/on-device-benchmarks/mars32-opt3-mem-usage.json', 'experiment_results/on-device-benchmarks/mars32-opt4-mem-usage.json'],
#                      custom_names=['LoRA ($r$ = 32)', 'MARS Q-OPT0 ($r$ = 32)', 'MARS Q-OPT1 ($r$ = 32)'])

#plot_ram_usage_filled(['experiment_results/on-device-benchmarks/lora8-mem-usage.json', 'experiment_results/on-device-benchmarks/mars8-opt3-mem-usage.json', 'experiment_results/on-device-benchmarks/mars8-opt4-mem-usage.json'],
#                      custom_names=['LoRA ($r$ = 8)', 'MARS Q-OPT0 ($r$ = 8)', 'MARS Q-OPT1 ($r$ = 8)'])

plot_ram_usage_filled(['experiment_results/on-device-benchmarks/lora2-mem-usage.json', 'experiment_results/on-device-benchmarks/mars2-opt3-mem-usage.json', 'experiment_results/on-device-benchmarks/mars2-opt4-mem-usage.json'],
                      custom_names=['LoRA ($r$ = 2)', 'MARS Q-OPT0 ($r$ = 2)', 'MARS O-OPT1 ($r$ = 2)'])
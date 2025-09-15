import json
import matplotlib
from matplotlib.transforms import Bbox
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def plot_peft_training_efficiency(peft_directories, peft_names, task_name, ranks, name="temp",
                                 output_filename='peft_training_efficiency.pdf', 
                                 x_axis_start=0, x_axis_end=None, 
                                 y_axis_start=0, y_axis_end=None):
    """
    Create a scatter plot showing GPU memory vs training runtime for different PEFT methods and ranks.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        task_name (str): Task name to filter by (e.g., 'arc_e', 'winogrande')
        ranks (list): List of ranks to plot (e.g., [2, 8, 32])
        output_filename (str): Name of the output PDF file
        x_axis_start (float): Minimum value for x-axis (training time in minutes). Default is 0.
        x_axis_end (float): Maximum value for x-axis (training time in minutes). Default is auto.
        y_axis_start (float): Minimum value for y-axis (GPU memory in GB). Default is 0.
        y_axis_end (float): Maximum value for y-axis (GPU memory in GB). Default is auto.
    
    Returns:
        None: Saves the plot as PDF
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Colors and markers for different PEFT methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Marker sizes for different ranks (made larger and more distinct)
    min_rank = min(ranks) if ranks else 2
    max_rank = max(ranks) if ranks else 32
    rank_sizes = {}
    for rank in ranks:
        # Scale sizes from 150 to 500 based on rank
        if max_rank > min_rank:
            normalized = (rank - min_rank) / (max_rank - min_rank)
        else:
            normalized = 0.5
        rank_sizes[rank] = 150 + normalized * 600  # Range: 150-500
    
    # Store results
    results = {}  # {peft_name: {rank: {'gpu_mem': X, 'train_runtime': Y}}}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        results[peft_name] = {}
        
        for rank in ranks:
            # Search for subdirectories with the pattern for this rank and task
            # Determine ranks for this method
            if peft_name == 'LoRA-XS':
                lookup_rank = rank * 8 
            else:
                lookup_rank = rank

            search_pattern = os.path.join(peft_dir, f"*-{task_name}-r{lookup_rank}-*")
            subdirs = glob.glob(search_pattern)
            
            for subdir in subdirs:
                if os.path.isdir(subdir):
                    # Look for training_logs.json in this directory
                    logs_file = os.path.join(subdir, "training_logs.json")
                    if os.path.exists(logs_file):
                        try:
                            with open(logs_file, 'r') as f:
                                logs_data = json.load(f)
                                
                                # Get the last entry (summary data)
                                if logs_data and isinstance(logs_data, list):
                                    last_entry = logs_data[-1]
                                    
                                    # Extract GPU memory (convert to GB) and training runtime (convert to minutes)
                                    gpu_mem_gb = last_entry.get('gpu_mem', 0)  # Convert bytes to GB
                                    train_runtime_minutes = last_entry.get('train_runtime', 0) / 60  # Convert seconds to minutes
                                    
                                    results[peft_name][rank] = {
                                        'gpu_mem': gpu_mem_gb,
                                        'train_runtime': train_runtime_minutes
                                    }
                                    
                                    print(f"Found {peft_name} r={rank}: {gpu_mem_gb:.2f}GB, {train_runtime_minutes:.2f}min")
                                    break  # Found the data for this rank
                                    
                        except Exception as e:
                            print(f"Error reading {logs_file}: {e}")
    
    # Create the scatter plot with extra space for legends
    fig, ax = plt.subplots(figsize=(13, 10))  # Made wider to accommodate larger legends
    plt.subplots_adjust(right=0.30)  # Leave more space for legends
    ax.tick_params(axis='both', which='major', labelsize=20)
    # Plot connecting lines first (underneath dots)
    for i, peft_name in enumerate(peft_names):
        if peft_name in results:
            color = colors[i % len(colors)]
            
            # Collect data points for this method and sort by rank
            line_data = []
            for rank in sorted(ranks):
                if rank in results[peft_name]:
                    data = results[peft_name][rank]
                    line_data.append((data['train_runtime'], data['gpu_mem']))
            
            if len(line_data) > 1:  # Only draw line if we have multiple points
                x_line = [point[0] for point in line_data]
                y_line = [point[1] for point in line_data]
                ax.plot(x_line, y_line, color=color, alpha=0.5, linewidth=2, zorder=1)
    
    # Plot each PEFT method (dots on top)
    for i, peft_name in enumerate(peft_names):
        if peft_name in results:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Collect data points for this method
            x_values = []  # training runtime
            y_values = []  # gpu memory
            sizes = []     # marker sizes based on rank
            
            for rank in ranks:
                if rank in results[peft_name]:
                    data = results[peft_name][rank]
                    x_values.append(data['train_runtime'])
                    y_values.append(data['gpu_mem'])
                    sizes.append(rank_sizes.get(rank, 200))
            
            if x_values:  # Only plot if we have data
                # Plot the points (on top of lines)
                scatter = ax.scatter(x_values, y_values, s=sizes, c=[color]*len(x_values), 
                                   marker=marker, alpha=0.7, edgecolors='black', linewidth=1.5, 
                                   label=peft_name, zorder=2)
    
    # Customize the plot
    ax.set_xlabel('Training Runtime (minutes)', fontsize=20, fontweight='bold')
    ax.set_ylabel('GPU Memory Usage (GB)', fontsize=20, fontweight='bold')
    ax.set_title(f'{name}', 
                fontsize=20, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend for methods on the right with more spacing
    legend1 = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, 
                       framealpha=0.9, title='PEFT Methods', title_fontsize=20,
                       labelspacing=1.5)  # Increased spacing and font sizes
    
    # Create a second legend for rank sizes on the right with more spacing (moved down and larger)
    rank_legend_elements = []
    for rank in sorted(set(ranks)):
        if rank in rank_sizes:
            size = rank_sizes[rank]
            rank_legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.7, 
                                                   edgecolors='black', linewidth=1,
                                                   label=f'Rank {rank}'))
    
    if rank_legend_elements:
        legend2 = ax.legend(handles=rank_legend_elements, bbox_to_anchor=(1.05, 0.3), 
                           loc='upper left', fontsize=14, framealpha=0.9, title='Rank Sizes',
                           title_fontsize=14, labelspacing=1.2,  # Increased spacing and font sizes
                           markerscale=0.9)  # Make the legend markers slightly larger
        ax.add_artist(legend1)  # Add the first legend back
    
    # Set axis limits with some padding
    if any(results.values()):
        all_runtimes = [data['train_runtime'] for method_data in results.values() 
                       for data in method_data.values()]
        all_gpu_mems = [data['gpu_mem'] for method_data in results.values() 
                       for data in method_data.values()]
        
        if all_runtimes and all_gpu_mems:
            x_max = x_axis_end if x_axis_end is not None else max(all_runtimes) * 1.1
            y_max = y_axis_end if y_axis_end is not None else max(all_gpu_mems) * 1.1
            ax.set_xlim(x_axis_start, x_max)
            ax.set_ylim(y_axis_start, y_max)
    
    # Adjust layout and save
    plt.tight_layout()
    #plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    extended_bbox = Bbox.from_bounds(
        fig_bbox.x0,  # left (no change)
        fig_bbox.y0,  # bottom (no change) 
        fig_bbox.width + 50,  # width + extra pixels on right
        fig_bbox.height  # height (no change)
    )
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight', bbox_extra_artists=[legend1])
    plt.show()
    
    # Print summary table
    print(f"\nTraining Efficiency Summary for {task_name.upper()}:")
    print("=" * 70)
    print(f"{'Method':<15} {'Rank':<6} {'GPU Memory (GB)':<15} {'Runtime (minutes)':<16}")
    print("-" * 70)
    
    for peft_name in peft_names:
        if peft_name in results:
            for rank in sorted(results[peft_name].keys()):
                data = results[peft_name][rank]
                print(f"{peft_name:<15} {rank:<6} {data['gpu_mem']:<15.2f} {data['train_runtime']:<16.1f}")
    
    print(f"\nScatter plot saved as: {output_filename}")

def plot_peft_3d_comparison(peft_directories, peft_names, title='PEFT Methods 3D Performance Comparison',
                           output_filename='peft_3d_comparison.pdf'):
    """
    Create a 3D scatter plot showing training runtime vs GPU memory vs task accuracy for different PEFT methods.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        title (str): Title for the plot
        output_filename (str): Name of the output PDF file
    
    Returns:
        None: Saves the plot as PDF
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Define benchmark datasets and ranks
    benchmarks = ['arc_e', 'arc_c', 'winogrande', 'boolq', 'logiqa', 'hellaswag']
    ranks = [2, 8, 32]
    
    # Colors and markers for different PEFT methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#45b7d1', '#96ceb4', '#bb8fce', '#85c1e9',
          '#f8c471', '#82e0aa', '#f1948a', '#85929e', '#d7bde2',
          '#a9dfbf', '#f9e79f', '#d5a6bd', '#aed6f1', '#f4d03f',
          '#cd6155', '#58d68d', '#5dade2', '#af7ac5', '#f7dc6f']
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd', 'H',
           '$\\clubsuit$', '$\\diamondsuit$', '$\\heartsuit$', '$\\spadesuit$',
           '$\\bigstar$', '$\\bigcirc$', '$\\bigoplus$', '$\\bigotimes$']
    
    # Store results for each method
    method_results = {}  # {peft_name: {'runtime': avg, 'gpu_mem': avg, 'accuracy': avg}}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        print(f"Processing {peft_name}...")
        
        # Determine ranks for this method
        if peft_name == 'LoRA-XS':
            search_ranks = [16, 64, 256]
        else:
            search_ranks = ranks
        
        # Collect all data for this method across all ranks and tasks
        all_runtimes = []
        all_gpu_mems = []
        all_accuracies = []
        
        for rank in search_ranks:
            # Search for subdirectories with the pattern for this rank
            search_pattern = os.path.join(peft_dir, f"*-r{rank}-*")
            subdirs = glob.glob(search_pattern)
            
            for subdir in subdirs:
                if os.path.isdir(subdir):
                    dir_name = os.path.basename(subdir)
                    
                    # Find which benchmark this is
                    benchmark_found = None
                    for benchmark in benchmarks:
                        if f"-{benchmark}-" in dir_name:
                            benchmark_found = benchmark
                            break
                    
                    if benchmark_found:
                        # Read training logs for runtime and GPU memory
                        logs_file = os.path.join(subdir, "training_logs.json")
                        if os.path.exists(logs_file):
                            try:
                                with open(logs_file, 'r') as f:
                                    logs_data = json.load(f)
                                    
                                    if logs_data and isinstance(logs_data, list):
                                        last_entry = logs_data[-1]
                                        
                                        # Extract GPU memory (convert to GB) and training runtime (convert to minutes)
                                        gpu_mem_gb = last_entry.get('gpu_mem', 0)
                                        train_runtime_minutes = last_entry.get('train_runtime', 0) / 60
                                        
                                        all_runtimes.append(train_runtime_minutes)
                                        all_gpu_mems.append(gpu_mem_gb)
                                        
                            except Exception as e:
                                print(f"Error reading {logs_file}: {e}")
                        
                        # Read eval results for accuracy
                        eval_file = os.path.join(subdir, "eval_results.json")
                        if os.path.exists(eval_file):
                            try:
                                with open(eval_file, 'r') as f:
                                    eval_data = json.load(f)
                                    accuracy = eval_data.get('results', 0.0)
                                    all_accuracies.append(accuracy)
                                    
                            except Exception as e:
                                print(f"Error reading {eval_file}: {e}")
        
        # Calculate averages for this method
        if all_runtimes and all_gpu_mems and all_accuracies:
            avg_runtime = sum(all_runtimes) / len(all_runtimes)
            avg_gpu_mem = sum(all_gpu_mems) / len(all_gpu_mems)
            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
            
            method_results[peft_name] = {
                'runtime': avg_runtime,
                'gpu_mem': avg_gpu_mem,
                'accuracy': avg_accuracy
            }
            
            print(f"  {peft_name}: Runtime={avg_runtime:.2f}min, GPU={avg_gpu_mem:.2f}GB, Accuracy={avg_accuracy:.3f}")
        else:
            print(f"  Warning: No complete data found for {peft_name}")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlim(30, 80) 
    ax.set_ylim(0, 5)
    ax.set_zlim(0.3, 0.6) 
            
    # After plotting each scatter point, add a vertical line
    for i, peft_name in enumerate(peft_names):
        if peft_name in method_results:
            data = method_results[peft_name]
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Plot the point
            ax.scatter(data['runtime'], data['gpu_mem'], data['accuracy'],
                      c=color, marker=marker, s=300, alpha=0.8, 
                      edgecolors='black', linewidth=1.5, label=peft_name)

            # Add vertical line from point down to bottom
            ax.plot([data['runtime'], data['runtime']], 
                    [data['gpu_mem'], data['gpu_mem']], 
                   [0.3, data['accuracy']], 
                    color=color, linestyle='-', alpha=0.6, linewidth=2)


    # Customize the plot
    ax.set_xlabel('Training Runtime (minutes)', fontsize=16, fontweight='bold')
    ax.set_ylabel('GPU Memory Usage (GB)', fontsize=16, fontweight='bold')
    ax.set_zlabel('Task Accuracy', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, columnspacing=3, labelspacing=1.5,
              framealpha=0.9, title='PEFT Methods', title_fontsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', 
                pad_inches=0.3, dpi=300)
    plt.show()
    
    # Print summary
    print(f"\n3D Scatter Plot Summary:")
    print("=" * 60)
    print(f"{'Method':<15} {'Runtime (min)':<15} {'GPU Mem (GB)':<15} {'Accuracy':<10}")
    print("-" * 60)
    
    for peft_name in peft_names:
        if peft_name in method_results:
            data = method_results[peft_name]
            print(f"{peft_name:<15} {data['runtime']:<15.2f} {data['gpu_mem']:<15.2f} {data['accuracy']:<10.3f}")
    
    print(f"\n3D scatter plot saved as: {output_filename}")
    
    return method_results

def plot_peft_parameter_efficiency(peft_directories, peft_names, title='Parameter Efficiency (Accuracy Gain vs Baseline)',
                                              output_filename='peft_param_efficiency.pdf'):
    """
    Create horizontal bar plots showing accuracy gain (vs baseline) per million parameters for different PEFT methods and ranks.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        title (str): Title for the plot
        output_filename (str): Name of the output PDF file
    
    Returns:
        dict: Results dictionary with efficiency scores
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Define benchmark datasets and their baselines (random guessing accuracy)
    benchmark_baselines = {
        'arc_e': 0.25,      # 4 choices: 1/4 = 25%
        'arc_c': 0.25,      # 4 choices: 1/4 = 25%
        'winogrande': 0.50, # 2 choices: 1/2 = 50%
        'boolq': 0.50,      # 2 choices (True/False): 1/2 = 50%
        'logiqa': 0.25,     # 4 choices: 1/4 = 25%
        'hellaswag': 0.25   # 4 choices: 1/4 = 25%
    }
    
    benchmarks = list(benchmark_baselines.keys())
    ranks = [2, 8, 32]
    
    # Store results for each method and rank
    results = {}  # {peft_name: {rank: {'accuracy': avg, 'baseline_gain': gain, 'params': count, 'efficiency': score}}}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        results[peft_name] = {}
        print(f"Processing {peft_name}...")
        
        # Determine ranks for this method
        if peft_name == 'LoRA-XS':
            search_ranks = [16, 64, 256]
        else:
            search_ranks = ranks
        
        for rank in search_ranks:
            accuracies = []
            benchmark_names = []
            trainable_params = None
            
            # Search for subdirectories with the pattern for this rank
            search_pattern = os.path.join(peft_dir, f"*-r{rank}-*")
            subdirs = glob.glob(search_pattern)
            
            for subdir in subdirs:
                if os.path.isdir(subdir):
                    dir_name = os.path.basename(subdir)
                    
                    # Find which benchmark this is
                    benchmark_found = None
                    for benchmark in benchmarks:
                        if f"-{benchmark}-" in dir_name:
                            benchmark_found = benchmark
                            break
                    
                    if benchmark_found:
                        # Read eval results for accuracy
                        eval_file = os.path.join(subdir, "eval_results.json")
                        if os.path.exists(eval_file):
                            try:
                                with open(eval_file, 'r') as f:
                                    eval_data = json.load(f)
                                    accuracy = eval_data.get('results', 0.0)
                                    accuracies.append(accuracy)
                                    benchmark_names.append(benchmark_found)
                            except Exception as e:
                                print(f"Error reading {eval_file}: {e}")
                        
                        # Read trainable parameters (only need once per rank)
                        if trainable_params is None:
                            config_file = os.path.join(subdir, "training_configuration.json")
                            if os.path.exists(config_file):
                                try:
                                    with open(config_file, 'r') as f:
                                        config_data = json.load(f)
                                        peft_config = config_data.get('peft_config', {})
                                        trainable_params = peft_config.get('trainable_parameter_count', 0)
                                except Exception as e:
                                    print(f"Error reading {config_file}: {e}")
            
            # Calculate average accuracy gain for this rank
            if accuracies and trainable_params and len(accuracies) == len(benchmark_names):
                # Calculate baseline-corrected accuracy gains
                baseline_gains = []
                for accuracy, benchmark in zip(accuracies, benchmark_names):
                    baseline = benchmark_baselines[benchmark]
                    gain = accuracy - baseline
                    baseline_gains.append(gain)
                
                avg_accuracy = sum(accuracies) / len(accuracies)
                avg_baseline_gain = sum(baseline_gains) / len(baseline_gains)
                params_millions = trainable_params / 1_000_000
                
                # Calculate efficiency: accuracy gain (vs baseline) per million parameters
                efficiency = avg_baseline_gain / params_millions if params_millions > 0 else 0
                
                results[peft_name][rank] = {
                    'accuracy': avg_accuracy,
                    'baseline_gain': avg_baseline_gain,
                    'params_millions': params_millions,
                    'efficiency': efficiency
                }
                
                print(f"  Rank {rank}: Accuracy={avg_accuracy:.3f}, Baseline Gain={avg_baseline_gain:.3f}, "
                      f"Params={params_millions:.2f}M, Efficiency={efficiency:.2f}")
    
    # Create horizontal bar plots in vertical layout (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.4)
    
    # Colors for different PEFT methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Hatching patterns for different methods
    hatches = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo', '***', '///']
    
    # Plot for each rank
    for rank_idx, rank in enumerate(ranks):
        ax = axes[rank_idx]
        
        # Collect data for this rank
        methods = []
        efficiencies = []
        method_colors = []
        method_hatches = []
        
        for i, peft_name in enumerate(peft_names):
            if peft_name in results:
                # Handle different ranks for LoRA-XS
                if peft_name == 'LoRA-XS':
                    lora_xs_ranks = [16, 64, 256]
                    if rank_idx < len(lora_xs_ranks):
                        actual_rank = lora_xs_ranks[rank_idx]
                    else:
                        continue
                else:
                    actual_rank = rank
                
                if actual_rank in results[peft_name]:
                    methods.append(peft_name)
                    efficiencies.append(results[peft_name][actual_rank]['efficiency'])
                    method_colors.append(colors[i % len(colors)])
                    method_hatches.append(hatches[i % len(hatches)])
        
        # Create horizontal bar plot
        if methods and efficiencies:
            bars = ax.barh(range(len(methods)), efficiencies, color=method_colors, 
                          alpha=0.8, edgecolor='black', linewidth=1,
                          hatch=method_hatches)
            
            # Customize subplot
            ax.set_xlabel('Accuracy Gain per Million Parameters', fontsize=14, fontweight='bold')
            
            # Use scientific notation for x-axis
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            # Increase tick label font sizes
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=14)
            
            # Set rank title
            if rank_idx < len(ranks):
                ax.set_title(f'Rank {ranks[rank_idx]}', fontsize=14, fontweight='bold')
            
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels at the end of bars
            for bar, efficiency in zip(bars, efficiencies):
                width = bar.get_width()
                
                # Format label based on magnitude
                #if abs(efficiency) >= 100:
                #    label = f'{efficiency:.0f}'
                #elif abs(efficiency) >= 10:
                #    label = f'{efficiency:.1f}'
                #else:
                label = f'{efficiency:.3f}'
                
                ax.text(width + abs(width)*0.01, bar.get_y() + bar.get_height()/2.,
                       label, ha='left', va='center', fontsize=14, fontweight='bold')
        current_xlim = ax.get_xlim()
        x_range = current_xlim[1] - current_xlim[0]
        ax.set_xlim(current_xlim[0], current_xlim[1] + 0.1 * x_range)  # Extend by 10%
    
    # Save the plot
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', 
                pad_inches=0.3, dpi=300)
    plt.show()
    
    # Print summary table
    print(f"\nParameter Efficiency Summary (Baseline-Corrected):")
    print("=" * 95)
    print(f"{'Method':<15} {'Rank':<8} {'Accuracy':<10} {'Baseline Gain':<14} {'Params (M)':<12} {'Efficiency':<12}")
    print("-" * 95)
    
    for peft_name in peft_names:
        if peft_name in results:
            # Get appropriate ranks for this method
            if peft_name == 'LoRA-XS':
                method_ranks = [16, 64, 256]
            else:
                method_ranks = ranks
            
            for rank in method_ranks:
                if rank in results[peft_name]:
                    data = results[peft_name][rank]
                    print(f"{peft_name:<15} {rank:<8} {data['accuracy']:<10.3f} "
                          f"{data['baseline_gain']:<14.3f} {data['params_millions']:<12.2f} "
                          f"{data['efficiency']:<12.2f}")
    
    print(f"\nBaseline-corrected parameter efficiency plot saved as: {output_filename}")
    
    # Print baseline information for reference
    print(f"\nDataset Baselines (Random Guessing):")
    print("-" * 40)
    for dataset, baseline in benchmark_baselines.items():
        print(f"{dataset:<15}: {baseline:>6.1f}%")
    
    return results


def plot_peft_memory_efficiency(peft_directories, peft_names, title='PEFT Memory Efficiency: Accuracy Gain vs Baseline per MB GPU Memory',
                                             output_filename='peft_memory_efficiency.pdf'):
    """
    Create horizontal bar plots showing accuracy gain (vs baseline) per MB GPU memory for different PEFT methods and ranks.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        title (str): Title for the plot
        output_filename (str): Name of the output PDF file
    
    Returns:
        dict: Results dictionary with efficiency scores
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Define benchmark datasets and their baselines (random guessing accuracy in decimal form)
    benchmark_baselines = {
        'arc_e': 0.25,      # 4 choices: 1/4 = 0.25
        'arc_c': 0.25,      # 4 choices: 1/4 = 0.25
        'winogrande': 0.5,  # 2 choices: 1/2 = 0.5
        'boolq': 0.5,       # 2 choices (True/False): 1/2 = 0.5
        'logiqa': 0.25,     # 4 choices: 1/4 = 0.25
        'hellaswag': 0.25   # 4 choices: 1/4 = 0.25
    }
    
    benchmarks = list(benchmark_baselines.keys())
    ranks = [2, 8, 32]
    
    # Store results for each method and rank
    results = {}  # {peft_name: {rank: {'accuracy': avg, 'baseline_gain': gain, 'gpu_mem_mb': avg, 'efficiency': score}}}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        results[peft_name] = {}
        print(f"Processing {peft_name}...")
        
        # Determine ranks for this method
        if peft_name == 'LoRA-XS':
            search_ranks = [16, 64, 256]
        else:
            search_ranks = ranks
        
        for rank in search_ranks:
            accuracies = []
            benchmark_names = []
            gpu_memories = []
            
            # Search for subdirectories with the pattern for this rank
            search_pattern = os.path.join(peft_dir, f"*-r{rank}-*")
            subdirs = glob.glob(search_pattern)
            
            for subdir in subdirs:
                if os.path.isdir(subdir):
                    dir_name = os.path.basename(subdir)
                    
                    # Find which benchmark this is
                    benchmark_found = None
                    for benchmark in benchmarks:
                        if f"-{benchmark}-" in dir_name:
                            benchmark_found = benchmark
                            break
                    
                    if benchmark_found:
                        # Read eval results for accuracy
                        eval_file = os.path.join(subdir, "eval_results.json")
                        if os.path.exists(eval_file):
                            try:
                                with open(eval_file, 'r') as f:
                                    eval_data = json.load(f)
                                    accuracy = eval_data.get('results', 0.0)
                                    accuracies.append(accuracy)
                                    benchmark_names.append(benchmark_found)
                            except Exception as e:
                                print(f"Error reading {eval_file}: {e}")
                        
                        # Read training logs for GPU memory
                        logs_file = os.path.join(subdir, "training_logs.json")
                        if os.path.exists(logs_file):
                            try:
                                with open(logs_file, 'r') as f:
                                    logs_data = json.load(f)
                                    
                                    if logs_data and isinstance(logs_data, list):
                                        last_entry = logs_data[-1]
                                        # Extract GPU memory (already in appropriate units)
                                        gpu_mem_gb = last_entry.get('gpu_mem', 0)
                                        gpu_memories.append(gpu_mem_gb)
                            except Exception as e:
                                print(f"Error reading {logs_file}: {e}")
            
            # Calculate averages for this rank
            if (accuracies and gpu_memories and benchmark_names and 
                len(accuracies) == len(gpu_memories) == len(benchmark_names)):
                
                # Calculate baseline-corrected accuracy gains
                baseline_gains = []
                for accuracy, benchmark in zip(accuracies, benchmark_names):
                    baseline = benchmark_baselines[benchmark]
                    gain = accuracy - baseline
                    baseline_gains.append(gain)
                
                avg_accuracy = sum(accuracies) / len(accuracies)
                avg_baseline_gain = sum(baseline_gains) / len(baseline_gains)
                avg_gpu_mem_gb = sum(gpu_memories) / len(gpu_memories)
                
                # Convert GB to MB and calculate memory efficiency: accuracy gain (vs baseline) per MB of GPU memory
                avg_gpu_mem_mb = avg_gpu_mem_gb * 1024  # Convert GB to MB
                efficiency = avg_baseline_gain / avg_gpu_mem_mb if avg_gpu_mem_mb > 0 else 0
                
                results[peft_name][rank] = {
                    'accuracy': avg_accuracy,
                    'baseline_gain': avg_baseline_gain,
                    'gpu_mem_gb': avg_gpu_mem_gb,
                    'gpu_mem_mb': avg_gpu_mem_mb,
                    'efficiency': efficiency
                }
                
                print(f"  Rank {rank}: Accuracy={avg_accuracy:.3f}, Baseline Gain={avg_baseline_gain:.3f}, "
                      f"GPU Memory={avg_gpu_mem_gb:.2f}GB ({avg_gpu_mem_mb:.0f}MB), Efficiency={efficiency:.6f}")
    
    # Create horizontal bar plots in vertical layout (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Colors for different PEFT methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Hatching patterns for different methods
    hatches = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo', '***', '///']
    
    # Plot for each rank
    for rank_idx, rank in enumerate(ranks):
        ax = axes[rank_idx]
        
        # Collect data for this rank
        methods = []
        efficiencies = []
        method_colors = []
        method_hatches = []
        
        for i, peft_name in enumerate(peft_names):
            if peft_name in results:
                # Handle different ranks for LoRA-XS
                if peft_name == 'LoRA-XS':
                    lora_xs_ranks = [16, 64, 256]
                    if rank_idx < len(lora_xs_ranks):
                        actual_rank = lora_xs_ranks[rank_idx]
                    else:
                        continue
                else:
                    actual_rank = rank
                
                if actual_rank in results[peft_name]:
                    methods.append(peft_name)
                    efficiencies.append(results[peft_name][actual_rank]['efficiency'])
                    method_colors.append(colors[i % len(colors)])
                    method_hatches.append(hatches[i % len(hatches)])
        
        # Create horizontal bar plot
        if methods and efficiencies:
            bars = ax.barh(range(len(methods)), efficiencies, color=method_colors, 
                          alpha=0.8, edgecolor='black', linewidth=1,
                          hatch=method_hatches)
            
            # Customize subplot
            ax.set_xlabel('Accuracy Gain per MB GPU Memory', fontsize=14, fontweight='bold')
            
            # Use scientific notation for x-axis
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            # Increase tick label font sizes
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=14)
            
            # Set rank title
            if rank_idx < len(ranks):
                ax.set_title(f'Rank {ranks[rank_idx]}', fontsize=14, fontweight='bold')
            
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels at the end of bars
            for bar, efficiency in zip(bars, efficiencies):
                width = bar.get_width()
                
                # Format label based on magnitude (scientific notation scaling)
                if abs(efficiency) >= 1e-4:
                    label = f'{efficiency*1000:.3f}'
                elif abs(efficiency) >= 1e-5:
                    label = f'{efficiency*1000:.3f}'
                else:
                    label = f'{efficiency*1000:.3f}'
                
                ax.text(width + abs(width)*0.01, bar.get_y() + bar.get_height()/2.,
                       label, ha='left', va='center', fontsize=14, fontweight='bold')
                
        current_xlim = ax.get_xlim()
        x_range = current_xlim[1] - current_xlim[0]
        ax.set_xlim(current_xlim[0], current_xlim[1] + 0.1 * x_range)  # Extend by 10%
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.4)
    
    # Save the plot
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', 
                pad_inches=0.3, dpi=300)
    plt.show()
    
    # Print summary table
    print(f"\nMemory Efficiency Summary (Baseline-Corrected):")
    print("=" * 110)
    print(f"{'Method':<15} {'Rank':<8} {'Accuracy':<10} {'Baseline Gain':<14} {'GPU Mem (GB)':<12} {'GPU Mem (MB)':<12} {'Efficiency':<12}")
    print("-" * 110)
    
    for peft_name in peft_names:
        if peft_name in results:
            # Get appropriate ranks for this method
            if peft_name == 'LoRA-XS':
                method_ranks = [16, 64, 256]
            else:
                method_ranks = ranks
            
            for rank in method_ranks:
                if rank in results[peft_name]:
                    data = results[peft_name][rank]
                    print(f"{peft_name:<15} {rank:<8} {data['accuracy']:<10.3f} "
                          f"{data['baseline_gain']:<14.3f} {data['gpu_mem_gb']:<12.2f} "
                          f"{data['gpu_mem_mb']:<12.0f} {data['efficiency']:<12.6f}")
    
    print(f"\nBaseline-corrected memory efficiency plot saved as: {output_filename}")
    
    # Print baseline information for reference
    print(f"\nDataset Baselines (Random Guessing):")
    print("-" * 40)
    for dataset, baseline in benchmark_baselines.items():
        print(f"{dataset:<15}: {baseline:>6.2f}")
    
    return results

def quant_hellaswag_plot():
    plot_peft_training_efficiency([
        'experiment_results/TinyLlama_v1.1-loraq4',
        'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q8',
        'experiment_results/TinyLlama_v1.1-qlora',
        'experiment_results/TinyLlama_v1.1-qmars'
    ], [
        'LoRA (fp4)',
        'LoRA (int8)',
        'MARS Q-OPT0 (fp4)',
        'MARS Q-OPT0 (int8)',
        'MARS Q-OPT1 (fp4)',
        'MARS Q-OPT1 (int8)',
        'QLoRA',
        'QMARS'
    ], 'hellaswag', [2, 8, 32], x_axis_start=130, x_axis_end=140, y_axis_start=1, name="Quantized PEFT Training - HellaSwag", output_filename="quant_peft_efficiency.pdf")

def non_quant_hellaswag_plot():
    plot_peft_training_efficiency([
        'experiment_results/TinyLlama_v1.1-lora',
        'experiment_results/TinyLlama_v1.1-mars-opt0',
        'experiment_results/TinyLlama_v1.1-mars-opt1',
        'experiment_results/TinyLlama_v1.1-lora_xs',
        'experiment_results/TinyLlama_v1.1-vb_lora',
        #'experiment_results/TinyLlama_v1.1-loha'
    ], [
        'LoRA',
        'MARS OPT0',
        'MARS OPT1',
        'LoRA-XS',
        'VB LoRA',
        #'LoHA'
    ], 'hellaswag', [2, 8, 32], x_axis_start=130, x_axis_end=160, y_axis_start=4.2, y_axis_end=5.5, name="Full-precision PEFT Training - HellaSwag", output_filename="non_quant_peft_efficiency.pdf")

def plot_3d_comparison():
    plot_peft_3d_comparison([
            'experiment_results/TinyLlama_v1.1-lora',
            'experiment_results/TinyLlama_v1.1-mars-opt0',
            'experiment_results/TinyLlama_v1.1-mars-opt1',
            'experiment_results/TinyLlama_v1.1-lora_xs',
            'experiment_results/TinyLlama_v1.1-vb_lora',
            #'experiment_results/TinyLlama_v1.1-loha',
            'experiment_results/TinyLlama_v1.1-loraq4',
            'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
            'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
            'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
            'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
            'experiment_results/TinyLlama_v1.1-mars-opt4-q8',
            'experiment_results/TinyLlama_v1.1-qlora',
            'experiment_results/TinyLlama_v1.1-qmars'
        ], [
            'LoRA',
            'MARS OPT0',
            'MARS OPT1',
            'LoRA-XS',
            'VB LoRA',
            #'LoHA',
            'LoRA (fp4)',
            'LoRA (int8)',
            'MARS Q-OPT0 (fp4)',
            'MARS Q-OPT0 (int8)',
            'MARS Q-OPT1 (fp4)',
            'MARS Q-OPT1 (int8)',
            'QLoRA',
            'QMARS'
        ], title="PEFT Method Performance Comparison")



def parameter_efficiency_plot():
    plot_peft_parameter_efficiency([
                'experiment_results/TinyLlama_v1.1-lora',
                'experiment_results/TinyLlama_v1.1-mars-opt0',
                'experiment_results/TinyLlama_v1.1-mars-opt1',
                'experiment_results/TinyLlama_v1.1-lora_xs',
                'experiment_results/TinyLlama_v1.1-vb_lora',
                'experiment_results/TinyLlama_v1.1-loha',
                'experiment_results/TinyLlama_v1.1-loraq4',
                'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
                'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
                'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
                'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
                'experiment_results/TinyLlama_v1.1-mars-opt4-q8',
                'experiment_results/TinyLlama_v1.1-qlora',
                'experiment_results/TinyLlama_v1.1-qmars'
            ], [
                'LoRA',
                'MARS OPT0',
                'MARS OPT1',
                'LoRA-XS',
                'VB LoRA',
                'LoHA',
                'LoRA (fp4)',
                'LoRA (int8)',
                'MARS Q-OPT0 (fp4)',
                'MARS Q-OPT0 (int8)',
                'MARS Q-OPT1 (fp4)',
                'MARS Q-OPT1 (int8)',
                'QLoRA',
                'QMARS'
            ], title="Accuracy per Trainable Parameter")

def memory_efficiency_plot():
    plot_peft_memory_efficiency([
                'experiment_results/TinyLlama_v1.1-lora',
                'experiment_results/TinyLlama_v1.1-mars-opt0',
                'experiment_results/TinyLlama_v1.1-mars-opt1',
                'experiment_results/TinyLlama_v1.1-lora_xs',
                'experiment_results/TinyLlama_v1.1-vb_lora',
                'experiment_results/TinyLlama_v1.1-loha',
                'experiment_results/TinyLlama_v1.1-loraq4',
                'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
                'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
                'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
                'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
                'experiment_results/TinyLlama_v1.1-mars-opt4-q8',
                'experiment_results/TinyLlama_v1.1-qlora',
                'experiment_results/TinyLlama_v1.1-qmars'
            ], [
                'LoRA',
                'MARS OPT0',
                'MARS OPT1',
                'LoRA-XS',
                'VB LoRA',
                'LoHA',
                'LoRA (fp4)',
                'LoRA (int8)',
                'MARS Q-OPT0 (fp4)',
                'MARS Q-OPT0 (int8)',
                'MARS Q-OPT1 (fp4)',
                'MARS Q-OPT1 (int8)',
                'QLoRA',
                'QMARS'
            ], title="Accuracy per Memory size")

# Quant / Non quant PEFT plots
#quant_hellaswag_plot()
#non_quant_hellaswag_plot()
#memory_efficiency_plot()
parameter_efficiency_plot()
#plot_3d_comparison()
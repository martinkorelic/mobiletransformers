import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os
import glob
from math import pi

def plot_peft_radar_multiplots(peft_methods, ranks=[2, 8, 32], title="PEFT Methods Performance Comparison", output_filename='peft_radar_subplots.pdf'):
    """
    Create multiple radar chart subplots comparing PEFT methods across benchmark datasets for different ranks.
    
    Args:
        peft_methods (list): List of tuples containing (peft_dir, peft_name)
        ranks (list): List of ranks to analyze (e.g., [2, 8, 32])
        title (str): Custom title for the overall plot
        output_filename (str): Name of the output PDF file
    
    Returns:
        None: Saves the plot as PDF
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from math import pi
    import json
    import os
    import glob
    
    # Define benchmark datasets
    benchmarks = ['arc_c', 'arc_e', 'winogrande', 'boolq', 'logiqa', 'hellaswag']
    benchmark_labels = ['ARC-C', 'ARC-E', 'WinoGrande', 'BoolQ', 'LogiQA', 'HellaSwag']
    
    # Colors for different PEFT methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    # Colors for benchmark labels (replaced yellow with purple)
    benchmark_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#9B59B6', '#DDA0DD']
    
    # Calculate subplot layout: 2 columns, n rows
    n_subplots = len(ranks)
    n_cols = 2
    n_rows = (n_subplots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(12, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Extract PEFT method names for unified legend
    peft_names = [peft_name for _, peft_name in peft_methods]
    
    # Calculate angles for radar chart
    angles = [n / float(len(benchmarks)) * 2 * pi for n in range(len(benchmarks))]
    angles += angles[:1]  # Complete the circle
    
    # Process each rank
    for rank_idx, rank in enumerate(ranks):
        # Calculate subplot position using GridSpec - this properly centers odd subplots
        if len(ranks) % 2 == 1 and rank_idx == len(ranks) - 1:
            # Last subplot when odd number - span both columns to center it
            ax = fig.add_subplot(gs[n_rows - 1, :], projection='polar')
        else:
            # Normal positioning for first subplots
            row = rank_idx // n_cols
            col = rank_idx % n_cols
            ax = fig.add_subplot(gs[row, col], projection='polar')
        
        # Store results for all PEFT methods for this rank
        all_results = {}
        
        # Process each PEFT method
        for peft_dir, peft_name in peft_methods:
            all_results[peft_name] = {}
            
            # Adjust rank for LoRA-XS
            current_rank = rank
            if peft_name == 'LoRA-XS':
                if rank == 2:
                    current_rank = 16
                elif rank == 8:
                    current_rank = 64
                elif rank == 32:
                    current_rank = 256
            
            # Search for subdirectories with the pattern
            search_pattern = os.path.join(peft_dir, f"*-r{current_rank}-*")
            subdirs = glob.glob(search_pattern)
            
            for subdir in subdirs:
                if os.path.isdir(subdir):
                    # Extract benchmark name from directory name
                    dir_name = os.path.basename(subdir)
                    
                    # Find which benchmark this is
                    benchmark_found = None
                    for benchmark in benchmarks:
                        if f"-{benchmark}-" in dir_name:
                            benchmark_found = benchmark
                            break
                    
                    if benchmark_found:
                        # Look for eval_results.json in this directory
                        result_file = os.path.join(subdir, "eval_results.json")
                        if os.path.exists(result_file):
                            try:
                                with open(result_file, 'r') as f:
                                    data = json.load(f)
                                    all_results[peft_name][benchmark_found] = data.get('results', 0.0)
                            except Exception as e:
                                print(f"Error reading {result_file}: {e}")
                                all_results[peft_name][benchmark_found] = 0.0
            
            # Ensure all benchmarks are present (fill missing with 0)
            for benchmark in benchmarks:
                if benchmark not in all_results[peft_name]:
                    all_results[peft_name][benchmark] = 0.0
        
        # Plot each PEFT method on this subplot
        for i, peft_name in enumerate(peft_names):
            if peft_name in all_results:
                # Get values for this PEFT method
                values = [all_results[peft_name][benchmark] for benchmark in benchmarks]
                values += values[:1]  # Complete the circle
                
                # Get color
                color = colors[i % len(colors)]
                
                # Plot the line
                ax.plot(angles, values, 'o-', linewidth=2.5, label=peft_name, 
                       color=color, markersize=6)
                
                # Fill the area
                ax.fill(angles, values, alpha=0.15, color=color)
        
        # Customize the subplot
        ax.set_xticks(angles[:-1])
        
        # Set radial limits first
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
        
        # Add colored benchmark labels around the circle
        for i, (angle, label) in enumerate(zip(angles[:-1], benchmark_labels)):
            color = benchmark_colors[i % len(benchmark_colors)]
            # Position labels further out from the circle
            label_radius = 1.25
            ax.text(angle, label_radius, label, 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=11, fontweight='bold', color='black',  # Keep text black
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=2))
        
        # Hide the default tick labels since we're using custom colored ones
        ax.set_xticklabels([])
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        # Set subplot title with LaTeX formatting
        ax.set_title(f'Rank $r={rank}$', size=14, fontweight='bold', pad=35)
    
    # Add overall title
    fig.suptitle(title, size=20, fontweight='bold', y=0.98)
    
    # Create unified legend
    handles = []
    labels = []
    for i, peft_name in enumerate(peft_names):
        color = colors[i % len(colors)]
        # Create a line handle for the legend
        line = plt.Line2D([0], [0], color=color, linewidth=2.5, marker='o', markersize=6)
        handles.append(line)
        labels.append(peft_name)
    
    # Add legend to the right of the entire figure
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.2), 
               fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and colored labels
    plt.subplots_adjust(right=0.85, hspace=0.4, wspace=0.3)
    
    # Save and show
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Print summary
    print(f"\\nRadar chart subplots saved as: {output_filename}")
    print(f"Created {len(ranks)} subplots for ranks: {ranks}")
    print(f"PEFT methods included: {', '.join(peft_names)}")

def plot_peft_radar_chart(peft_directories, peft_names, rank=8, title="PEFT Methods Performance Comparison", output_filename='peft_radar_comparison.pdf'):
    """
    Create a radar chart comparing PEFT methods across benchmark datasets.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        rank (int): Rank to analyze (2, 8, or 32)
        title (str): Custom title for the plot
        output_filename (str): Name of the output PDF file
    
    Returns:
        None: Saves the plot as PDF
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Define benchmark datasets
    benchmarks = ['arc_c', 'arc_e', 'winogrande', 'boolq', 'logiqa', 'hellaswag']
    benchmark_labels = ['ARC-C', 'ARC-E', 'WinoGrande', 'BoolQ', 'LogiQA', 'HellaSwag']
    
    # Colors for different PEFT methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Store results for each PEFT method
    peft_results = {}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        peft_results[peft_name] = {}
        
        # Search for subdirectories with the pattern
        # Example: TinyLlama_v1.1-abl_B-arc_c-r2-a2
        search_pattern = os.path.join(peft_dir, f"*-r{rank}-*")
        subdirs = glob.glob(search_pattern)
        
        for subdir in subdirs:
            if os.path.isdir(subdir):
                # Extract benchmark name from directory name
                dir_name = os.path.basename(subdir)
                
                # Find which benchmark this is
                benchmark_found = None
                for benchmark in benchmarks:
                    if f"-{benchmark}-" in dir_name:
                        benchmark_found = benchmark
                        break
                
                if benchmark_found:
                    # Look for eval_results.json in this directory
                    result_file = os.path.join(subdir, "eval_results.json")
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r') as f:
                                data = json.load(f)
                                peft_results[peft_name][benchmark_found] = data.get('results', 0.0)
                                print(f"Found {peft_name} - {benchmark_found}: {data.get('results', 0.0):.4f}")
                        except Exception as e:
                            print(f"Error reading {result_file}: {e}")
                            peft_results[peft_name][benchmark_found] = 0.0
                    else:
                        print(f"Warning: No eval_results.json found in {subdir}")
    
    # Ensure all PEFT methods have all benchmarks (fill missing with 0)
    for peft_name in peft_results:
        for benchmark in benchmarks:
            if benchmark not in peft_results[peft_name]:
                peft_results[peft_name][benchmark] = 0.0
                print(f"Warning: No results found for {peft_name} on {benchmark}")
    
    # Calculate averages for each PEFT method
    peft_averages = {}
    for peft_name in peft_results:
        scores = [peft_results[peft_name][benchmark] for benchmark in benchmarks if peft_results[peft_name][benchmark] > 0]
        if scores:
            peft_averages[peft_name] = sum(scores) / len(scores)
        else:
            peft_averages[peft_name] = 0.0
    
    # Set up the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Calculate angles for each benchmark
    angles = [n / float(len(benchmarks)) * 2 * pi for n in range(len(benchmarks))]
    angles += angles[:1]  # Complete the circle
    
    # Plot each PEFT method
    for i, peft_name in enumerate(peft_names):
        if peft_name in peft_results:
            # Get values for this PEFT method
            values = [peft_results[peft_name][benchmark] for benchmark in benchmarks]
            values += values[:1]  # Complete the circle
            
            # Plot the line
            color = colors[i % len(colors)]
            
            ax.plot(angles, values, 'o-', linewidth=2.5, label=peft_name, 
                   color=color, markersize=6)
            
            # Fill the area
            ax.fill(angles, values, alpha=0.15, color=color)
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmark_labels, fontsize=16, fontweight='bold')
    
    # Push benchmark labels further from circle
    ax.tick_params(axis='x', pad=25)
    
    # Set radial limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, color='gray')
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    # Add custom title
    plt.title(title, size=18, fontweight='bold', pad=40, x=0.6)
    
    # Add legend outside the plot
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=16)
    
    # Add individual colored entries for each method
    for i, peft_name in enumerate(peft_names):
        if peft_name in peft_averages:
            avg_score = peft_averages[peft_name]
            color = colors[i % len(colors)]
            
            # Position each entry below the previous one
            y_offset = 0.5 - 0.08 - 0.06 * (i + 1)
            
            entry_text = f"{peft_name}: {avg_score:.3f}"
            ax.text(1.05, y_offset, entry_text, transform=ax.transAxes,
                   verticalalignment='center', horizontalalignment='left', 
                   fontsize=16, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=2))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Print detailed results
    print(f"\nDetailed Results for Rank {rank}:")
    print("=" * 60)
    for peft_name in peft_names:
        if peft_name in peft_results:
            print(f"\n{peft_name}:")
            print("-" * 30)
            total_score = 0
            valid_benchmarks = 0
            for benchmark, label in zip(benchmarks, benchmark_labels):
                score = peft_results[peft_name][benchmark]
                print(f"  {label:12}: {score:.4f}")
                if score > 0:
                    total_score += score
                    valid_benchmarks += 1
            
            if valid_benchmarks > 0:
                avg_score = total_score / valid_benchmarks
                print(f"  {'Average':12}: {avg_score:.4f}")
    
    print(f"\nRadar chart saved as: {output_filename}")

# Example usage:
    
def ablation_plots():
    plot_peft_radar_chart(['experiment_results/TinyLlama_v1.1-abl_A',
                        'experiment_results/TinyLlama_v1.1-abl_B',
                        'experiment_results/TinyLlama_v1.1-abl_C',
                        'experiment_results/TinyLlama_v1.1-abl_D',
                        'experiment_results/TinyLlama_v1.1-loraq4',
                        'experiment_results/TinyLlama_v1.1-loraq8'],
                        ['Variant A',
                            'Variant B',
                            'Variant C',
                            'Variant D',
                            'Variant E (LoRA int4)',
                            'Variant F (LoRA int8)'], rank=2, title='Ablation study benchmarks ($r = 2$)', output_filename='ablation_radar_r2.pdf')

    plot_peft_radar_chart(['experiment_results/TinyLlama_v1.1-abl_A',
                        'experiment_results/TinyLlama_v1.1-abl_B',
                        'experiment_results/TinyLlama_v1.1-abl_C',
                        'experiment_results/TinyLlama_v1.1-abl_D',
                        'experiment_results/TinyLlama_v1.1-loraq4',
                        'experiment_results/TinyLlama_v1.1-loraq8'],
                        ['Variant A',
                            'Variant B',
                            'Variant C',
                            'Variant D',
                            'Variant E (LoRA int4)',
                            'Variant F (LoRA int8)'], rank=8, title='Ablation study benchmarks ($r = 8$)', output_filename='ablation_radar_r8.pdf')

    plot_peft_radar_chart(['experiment_results/TinyLlama_v1.1-abl_A',
                        'experiment_results/TinyLlama_v1.1-abl_B',
                        'experiment_results/TinyLlama_v1.1-abl_C',
                        'experiment_results/TinyLlama_v1.1-abl_D',
                        'experiment_results/TinyLlama_v1.1-loraq4',
                        'experiment_results/TinyLlama_v1.1-loraq8'],
                        ['Variant A',
                            'Variant B',
                            'Variant C',
                            'Variant D',
                            'Variant E (LoRA int4)',
                            'Variant F (LoRA int8)'], rank=32, title='Ablation study benchmarks ($r = 32$)', output_filename='ablation_radar_r32.pdf')


def mars_plots():
    plot_peft_radar_chart(['experiment_results/TinyLlama_v1.1-mars-opt0',
                        'experiment_results/TinyLlama_v1.1-mars-opt1',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
                        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
                        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'],
                        ['MARS OPT0',
                            'MARS OPT1',
                            'MARS OPT3 (int8)',
                            'MARS OPT3 (int4)',
                            'MARS OPT4 (int8)',
                            'MARS OPT4 (int4)'], rank=2, title='MARS benchmarks ($r = 2$)', output_filename='mars_radar_r2.pdf')
    
    plot_peft_radar_chart(['experiment_results/TinyLlama_v1.1-mars-opt0',
                        'experiment_results/TinyLlama_v1.1-mars-opt1',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
                        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
                        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'],
                        ['MARS OPT0',
                            'MARS OPT1',
                            'MARS OPT3 (int8)',
                            'MARS OPT3 (int4)',
                            'MARS OPT4 (int8)',
                            'MARS OPT4 (int4)'], rank=8, title='MARS benchmarks ($r = 8$)', output_filename='mars_radar_r8.pdf')
    
    plot_peft_radar_chart(['experiment_results/TinyLlama_v1.1-mars-opt0',
                        'experiment_results/TinyLlama_v1.1-mars-opt1',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
                        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
                        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'],
                        ['MARS OPT0',
                            'MARS OPT1',
                            'MARS OPT3 (int8)',
                            'MARS OPT3 (int4)',
                            'MARS OPT4 (int8)',
                            'MARS OPT4 (int4)'], rank=32, title='MARS benchmarks ($r = 32$)', output_filename='mars_radar_r32.pdf')
    
def combined_plots():
    plot_peft_radar_chart([
                        'experiment_results/TinyLlama_v1.1-abl_B',
                        'experiment_results/TinyLlama_v1.1-mars-opt1'],
                        ['Variant B',
                            'MARS OPT1'], rank=2, title='Comparison benchmark ($r = 2$)', output_filename='comparison_radar_r2.pdf')
    
    plot_peft_radar_chart([
                        'experiment_results/TinyLlama_v1.1-abl_C',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q4'],
                        ['Variant C',
                            'MARS OPT3 (int4)'], rank=8, title='Comparison benchmark ($r = 8$)', output_filename='comparison_radar_r8.pdf')
    
    plot_peft_radar_chart([
                        'experiment_results/TinyLlama_v1.1-loraq4',
                        'experiment_results/TinyLlama_v1.1-mars-opt3-q8'],
                        ['Variant E',
                            'MARS OPT3 (int8)'], rank=32, title='Comparison benchmark ($r = 32$)', output_filename='comparison_radar_r32.pdf')
    
def q_combined_plots():
    plot_peft_radar_chart([
                        'experiment_results/TinyLlama_v1.1-qmars',
                        'experiment_results/TinyLlama_v1.1-qlora'],
                        ['QMARS',
                            'QLoRA'], rank=2, title='QMARS vs QLoRA ($r = 2$)', output_filename='qmars_qlora_radar_r2.pdf')
    
    plot_peft_radar_chart([
                        'experiment_results/TinyLlama_v1.1-qmars',
                        'experiment_results/TinyLlama_v1.1-qlora'],
                        ['QMARS',
                            'QLoRA'], rank=8, title='QMARS vs QLoRA ($r = 8$)', output_filename='qmars_qlora_radar_r8.pdf')
    
    plot_peft_radar_chart([
                        'experiment_results/TinyLlama_v1.1-qmars',
                        'experiment_results/TinyLlama_v1.1-qlora'],
                        ['QMARS',
                            'QLoRA'], rank=32, title='QMARS vs QLoRA ($r = 32$)', output_filename='qmars_qlora_radar_r32.pdf')
    
def multiplot_qmars_qlora():
    peft_methods = [
    ('experiment_results/TinyLlama_v1.1-qmars', 'QMARS'),
    ('experiment_results/TinyLlama_v1.1-qlora', 'QLoRA')
    ]
    plot_peft_radar_multiplots(peft_methods, ranks=[2, 8, 32], title='QMARS vs QLoRA', output_filename='qlora_qmars_radar.pdf')

def multiplot_mars():
    peft_methods = [
        ('experiment_results/TinyLlama_v1.1-mars-opt0', 'MARS OPT0'),
        ('experiment_results/TinyLlama_v1.1-mars-opt1', 'MARS OPT1'),
        ('experiment_results/TinyLlama_v1.1-mars-opt3-q4', 'MARS Q-OPT0 (fp4)'),
        ('experiment_results/TinyLlama_v1.1-mars-opt3-q8', 'MARS Q-OPT0 (int8)'),
        ('experiment_results/TinyLlama_v1.1-mars-opt4-q4', 'MARS Q-OPT1 (fp4)'),
        ('experiment_results/TinyLlama_v1.1-mars-opt4-q8', 'MARS Q-OPT1 (int8)')
    ]
    plot_peft_radar_multiplots(peft_methods, ranks=[2, 8, 32], title='MARS Methods Comparison', output_filename='mars_radar.pdf')

def multiplot_mars_lora():
    peft_methods = [
        ('experiment_results/TinyLlama_v1.1-mars-opt0', 'MARS OPT0'),
        ('experiment_results/TinyLlama_v1.1-mars-opt1', 'MARS OPT1'),
        ('experiment_results/TinyLlama_v1.1-lora', 'LoRA')
    ]
    plot_peft_radar_multiplots(peft_methods, ranks=[2, 8, 32], title='MARS vs LoRA', output_filename='lora_mars_radar.pdf')

#multiplot_qmars_qlora()
multiplot_mars()
#multiplot_mars_lora()
# Plots
#ablation_plots()
#mars_plots()
#combined_plots()
    
#q_combined_plots()
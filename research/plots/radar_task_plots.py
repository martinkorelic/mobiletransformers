import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from math import pi

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
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
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
            
            # Add value labels
            #for angle, value, benchmark in zip(angles[:-1], values[:-1], benchmarks):
            #    if value > 0:  # Only show label if there's a value
            #        # Calculate label position (slightly outside the point)
            #        label_distance = value + 0.05
            #        x = angle
            #        y = label_distance
            #        ax.annotate(f'{value:.3f}', (x, y), xytext=(5, 5), 
            #                   textcoords='offset points', fontsize=9, 
            #                   color=color, fontweight='bold',
            #                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
            #                            edgecolor=color, alpha=0.8))
    
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
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.0), fontsize=16)
    
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
    

# Plots
ablation_plots()
mars_plots()
combined_plots()
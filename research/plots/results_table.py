import json
import os
import glob

def format_param_count(param_count):
    """
    Format parameter count to readable format (e.g., 1.5M, 0.3M)
    
    Args:
        param_count (int): Number of parameters
    
    Returns:
        str: Formatted parameter count
    """
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.1f}M"
    elif param_count >= 1_000:
        return f"{param_count / 1_000_000:.2f}M"
    else:
        return f"{param_count / 1_000_000:.1f}M"

def generate_peft_latex_table(peft_directories, peft_names, output_filename='peft_results_table.tex'):
    """
    Generate a LaTeX table comparing PEFT methods across ranks and benchmarks.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        output_filename (str): Name of the output LaTeX file
    
    Returns:
        None: Saves the table as LaTeX file
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Define benchmark datasets and ranks
    benchmarks = ['arc_e', 'arc_c', 'winogrande', 'boolq', 'logiqa', 'hellaswag']
    benchmark_labels = ['ARC-E', 'ARC-C', 'WinoGrande', 'BoolQ', 'LogiQA', 'HellaSwag']
    ranks = [2, 8, 32]
    
    # Store all results
    results = {}  # {peft_name: {rank: {benchmark: score, 'trainable_params': count}}}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        results[peft_name] = {}
        if peft_name == 'LoRA-XS':
            search_ranks = [16, 64, 256]
        else:
            search_ranks = ranks

        for rank in search_ranks:
            results[peft_name][rank] = {}
            
            # Search for subdirectories with the pattern for this rank
            search_pattern = os.path.join(peft_dir, f"*-r{rank}-*")
            subdirs = glob.glob(search_pattern)
            
            trainable_params = None
            
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
                        # Read eval results
                        result_file = os.path.join(subdir, "eval_results.json")
                        if os.path.exists(result_file):
                            try:
                                with open(result_file, 'r') as f:
                                    data = json.load(f)
                                    results[peft_name][rank][benchmark_found] = data.get('results', 0.0)
                            except Exception as e:
                                print(f"Error reading {result_file}: {e}")
                                results[peft_name][rank][benchmark_found] = 0.0
                        
                        # Read trainable parameters (only need to do this once per rank)
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
                                    trainable_params = 0
            
            # Store trainable parameters for this rank
            results[peft_name][rank]['trainable_params'] = trainable_params or 0
            
            # Fill missing benchmarks with 0
            for benchmark in benchmarks:
                if benchmark not in results[peft_name][rank]:
                    results[peft_name][rank][benchmark] = 0.0
    
    # Generate LaTeX table
    latex_content = []
    
    # Document header
    latex_content.append(r"\documentclass{article}")
    latex_content.append(r"\usepackage{booktabs}")
    latex_content.append(r"\usepackage{array}")
    latex_content.append(r"\usepackage{multirow}")
    latex_content.append(r"\begin{document}")
    latex_content.append("")
    
    # Table caption and label
    latex_content.append(r"\begin{table}[h!]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{PEFT Methods Performance Comparison Across Ranks and Benchmarks}")
    latex_content.append(r"\label{tab:peft_comparison}")
    
    # Table structure - calculate number of columns
    num_cols = 2 + len(benchmarks) + 1  # Method + Rank/Params + benchmarks + Average
    col_spec = "l|c|" + "c" * len(benchmarks) + "|c"
    
    latex_content.append(f"\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\toprule")
    
    # Header row
    header = "\textbf{PEFT Method} & \textbf{Rank/Params} & "
    header += " & ".join([f"\textbf{{{label}}}" for label in benchmark_labels])
    header += " & \textbf{Average} \\"
    latex_content.append(header)
    latex_content.append("\midrule")
    
    # Data rows
    method_averages = []
    
    for peft_name in peft_names:
        if peft_name in results:
            # Get the ranks for this specific method
            if peft_name == 'LoRA-XS':
                current_method_ranks = [16, 64, 256]
            else:
                current_method_ranks = ranks
            
            first_row = True
            rank_averages = []
            
            for rank in current_method_ranks:
                if rank in results[peft_name]:
                    # Get scores for this rank
                    scores = [results[peft_name][rank][benchmark] for benchmark in benchmarks]
                    valid_scores = [s for s in scores if s > 0]
                    rank_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                    rank_averages.append(rank_avg)
                    
                    # Format trainable parameters
                    trainable_params = results[peft_name][rank]['trainable_params']
                    params_str = format_param_count(trainable_params)
                    
                    # Create row
                    if first_row:
                        method_cell = f"\multirow{{{len(current_method_ranks)}}}{{*}}{{\textbf{{{peft_name}}}}}"
                        first_row = False
                    else:
                        method_cell = ""
                    
                    rank_cell = f"r={rank} ({params_str})"
                    score_cells = " & ".join([f"{score:.3f}" if score > 0 else "-" for score in scores])
                    avg_cell = f"{rank_avg:.3f}" if rank_avg > 0 else "-"
                    
                    row = f"{method_cell} & {rank_cell} & {score_cells} & {avg_cell} \\"
                    latex_content.append(row)
            
            # Calculate method average
            method_avg = sum(rank_averages) / len(rank_averages) if rank_averages else 0.0
            method_averages.append(method_avg)
            
            # Add method average row
            avg_scores = []
            for benchmark in benchmarks:
                benchmark_scores = [results[peft_name][rank][benchmark] for rank in current_method_ranks 
                                  if rank in results[peft_name] and results[peft_name][rank][benchmark] > 0]
                benchmark_avg = sum(benchmark_scores) / len(benchmark_scores) if benchmark_scores else 0.0
                avg_scores.append(benchmark_avg)
            
            avg_row_cells = " & ".join([f"{score:.3f}" if score > 0 else "-" for score in avg_scores])
            avg_row = f" & \\textbf{{Average}} & {avg_row_cells} & \\textbf{{{method_avg:.3f}}} \\\\"
            latex_content.append(avg_row)
            latex_content.append(r"\midrule")
    
    # Overall average row
    if method_averages:
        overall_avg = sum(method_averages) / len(method_averages)
        
        # Calculate overall averages per benchmark
        overall_benchmark_avgs = []
        for benchmark in benchmarks:
            all_scores = []
            for peft_name in peft_names:
                if peft_name in results:
                    # Get appropriate ranks for this method
                    if peft_name == 'LoRA-XS':
                        method_ranks_for_overall = [16, 64, 256]
                    else:
                        method_ranks_for_overall = ranks
                    
                    for rank in method_ranks_for_overall:
                        if rank in results[peft_name] and results[peft_name][rank][benchmark] > 0:
                            all_scores.append(results[peft_name][rank][benchmark])
            benchmark_overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
            overall_benchmark_avgs.append(benchmark_overall_avg)
        
        overall_cells = " & ".join([f"{score:.3f}" if score > 0 else "-" for score in overall_benchmark_avgs])
        overall_row = f"\\textbf{{Overall Average}} & & {overall_cells} & \\textbf{{{overall_avg:.3f}}} \\\\"
        latex_content.append(overall_row)
    
    # Table footer
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    latex_content.append("")
    latex_content.append(r"\end{document}")
    
    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX table saved as: {output_filename}")
    
    # Also print summary statistics
    print(f"\nSummary Statistics:")
    print("=" * 50)
    for i, peft_name in enumerate(peft_names):
        if i < len(method_averages):
            print(f"{peft_name}: {method_averages[i]:.3f}")
    if method_averages:
        print(f"Overall Average: {sum(method_averages) / len(method_averages):.3f}")

def generate_peft_runtime_table(peft_directories, peft_names, output_filename='peft_runtime_table.tex'):
    """
    Generate a LaTeX table comparing PEFT methods training runtime across ranks and tasks.
    
    Args:
        peft_directories (list): List of directories containing PEFT results
        peft_names (list): List of display names for PEFT methods
        output_filename (str): Name of the output LaTeX file
    
    Returns:
        None: Saves the table as LaTeX file
    """
    if len(peft_directories) != len(peft_names):
        raise ValueError("Number of directories must match number of names")
    
    # Define benchmark datasets and ranks
    benchmarks = ['arc_e', 'arc_c', 'winogrande', 'boolq', 'logiqa', 'hellaswag']
    benchmark_labels = ['ARC-E', 'ARC-C', 'WinoGrande', 'BoolQ', 'LogiQA', 'HellaSwag']
    ranks = [2, 8, 32]
    
    # Store all results
    results = {}  # {peft_name: {rank: {benchmark: runtime, 'trainable_params': count}}}
    
    for peft_dir, peft_name in zip(peft_directories, peft_names):
        results[peft_name] = {}
        if peft_name == 'LoRA-XS':
            search_ranks = [16, 64, 256]
        else:
            search_ranks = ranks

        for rank in search_ranks:
            results[peft_name][rank] = {}
            
            # Search for subdirectories with the pattern for this rank
            search_pattern = os.path.join(peft_dir, f"*-r{rank}-*")
            subdirs = glob.glob(search_pattern)
            
            trainable_params = None
            
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
                        # Read training logs for runtime
                        logs_file = os.path.join(subdir, "training_logs.json")
                        if os.path.exists(logs_file):
                            try:
                                with open(logs_file, 'r') as f:
                                    logs_data = json.load(f)
                                    
                                    if logs_data and isinstance(logs_data, list):
                                        last_entry = logs_data[-1]
                                        # Training runtime in minutes
                                        runtime = last_entry.get('train_runtime', 0) / 60
                                        results[peft_name][rank][benchmark_found] = runtime
                            except Exception as e:
                                print(f"Error reading {logs_file}: {e}")
                                results[peft_name][rank][benchmark_found] = 0.0
                        
                        # Read trainable parameters (only need to do this once per rank)
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
                                    trainable_params = 0
            
            # Store trainable parameters for this rank
            results[peft_name][rank]['trainable_params'] = trainable_params or 0
            
            # Fill missing benchmarks with 0
            for benchmark in benchmarks:
                if benchmark not in results[peft_name][rank]:
                    results[peft_name][rank][benchmark] = 0.0
    
    # Generate LaTeX table
    latex_content = []
    
    # Document header
    latex_content.append(r"\documentclass{article}")
    latex_content.append(r"\usepackage{booktabs}")
    latex_content.append(r"\usepackage{array}")
    latex_content.append(r"\usepackage{multirow}")
    latex_content.append(r"\begin{document}")
    latex_content.append("")
    
    # Table caption and label
    latex_content.append(r"\begin{table}[h!]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{PEFT Methods Training Runtime Comparison Across Ranks and Tasks (minutes)}")
    latex_content.append(r"\label{tab:peft_runtime_comparison}")
    
    # Table structure - calculate number of columns
    num_cols = 2 + len(benchmarks) + 1  # Method + Rank/Params + benchmarks + Average
    col_spec = "l|c|" + "c" * len(benchmarks) + "|c"
    
    latex_content.append(f"\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\toprule")
    
    # Header row
    header = "\\textbf{PEFT Method} & \\textbf{Rank/Params} & "
    header += " & ".join([f"\\textbf{{{label}}}" for label in benchmark_labels])
    header += " & \\textbf{Average} \\\\"
    latex_content.append(header)
    latex_content.append("\\midrule")
    
    # Data rows
    method_averages = []
    
    for peft_name in peft_names:
        if peft_name in results:
            # Get the ranks for this specific method
            if peft_name == 'LoRA-XS':
                current_method_ranks = [16, 64, 256]
            else:
                current_method_ranks = ranks
            
            first_row = True
            rank_averages = []
            
            for rank in current_method_ranks:
                if rank in results[peft_name]:
                    # Get runtime values for this rank
                    runtimes = [results[peft_name][rank][benchmark] for benchmark in benchmarks]
                    valid_runtimes = [r for r in runtimes if r > 0]
                    rank_avg = sum(valid_runtimes) / len(valid_runtimes) if valid_runtimes else 0.0
                    rank_averages.append(rank_avg)
                    
                    # Format trainable parameters
                    trainable_params = results[peft_name][rank]['trainable_params']
                    if trainable_params >= 1_000_000:
                        params_str = f"{trainable_params / 1_000_000:.1f}M"
                    elif trainable_params >= 1_000:
                        params_str = f"{trainable_params / 1_000_000:.2f}M"
                    else:
                        params_str = f"{trainable_params / 1_000_000:.1f}M"
                    
                    # Create row
                    if first_row:
                        method_cell = f"\\multirow{{{len(current_method_ranks)}}}{{*}}{{\\textbf{{{peft_name}}}}}"
                        first_row = False
                    else:
                        method_cell = ""
                    
                    rank_cell = f"r={rank} ({params_str})"
                    runtime_cells = " & ".join([f"{runtime:.1f}" if runtime > 0 else "-" for runtime in runtimes])
                    avg_cell = f"{rank_avg:.1f}" if rank_avg > 0 else "-"
                    
                    row = f"{method_cell} & {rank_cell} & {runtime_cells} & {avg_cell} \\\\"
                    latex_content.append(row)
            
            # Calculate method average
            method_avg = sum(rank_averages) / len(rank_averages) if rank_averages else 0.0
            method_averages.append(method_avg)
            
            # Add method average row
            avg_runtimes = []
            for benchmark in benchmarks:
                benchmark_runtimes = [results[peft_name][rank][benchmark] for rank in current_method_ranks 
                                    if rank in results[peft_name] and results[peft_name][rank][benchmark] > 0]
                benchmark_avg = sum(benchmark_runtimes) / len(benchmark_runtimes) if benchmark_runtimes else 0.0
                avg_runtimes.append(benchmark_avg)
            
            avg_row_cells = " & ".join([f"{runtime:.1f}" if runtime > 0 else "-" for runtime in avg_runtimes])
            avg_row = f" & \\textbf{{Average}} & {avg_row_cells} & \\textbf{{{method_avg:.1f}}} \\\\"
            latex_content.append(avg_row)
            latex_content.append("\\midrule")
    
    # Table footer
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    latex_content.append("\\end{document}")
    
    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX runtime table saved as: {output_filename}")
    
    # Also print summary statistics
    print(f"\nRuntime Summary Statistics:")
    print("=" * 50)
    for i, peft_name in enumerate(peft_names):
        if i < len(method_averages):
            print(f"{peft_name}: {method_averages[i]:.1f} minutes")
    
    if method_averages:
        overall_avg = sum(method_averages) / len(method_averages)
        print(f"Overall Average: {overall_avg:.1f} minutes")


def ablation_table():
    generate_peft_latex_table([
        'experiment_results/TinyLlama_v1.1-abl_A',
        'experiment_results/TinyLlama_v1.1-abl_B',
        'experiment_results/TinyLlama_v1.1-abl_C',
        'experiment_results/TinyLlama_v1.1-abl_D'
    ], [
        'Variant A',
        'Variant B',
        'Variant C',
        'Variant D',
    ], 'peft_ablation_table.tex')


def mars_table():
    generate_peft_latex_table([
        'experiment_results/TinyLlama_v1.1-mars-opt0',
        'experiment_results/TinyLlama_v1.1-mars-opt1',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'
    ], [
        'MARS OPT0',
        'MARS OPT1',
        'MARS OPT3 (int4)',
        'MARS OPT3 (int8)',
        'MARS OPT4 (int4)',
        'MARS OPT4 (int8)',
    ], 'mars_table.tex')


def non_quant_table():
    generate_peft_latex_table([
        'experiment_results/TinyLlama_v1.1-lora',
        'experiment_results/TinyLlama_v1.1-lora_xs',
        'experiment_results/TinyLlama_v1.1-loha',
        'experiment_results/TinyLlama_v1.1-vb_lora',
        'experiment_results/TinyLlama_v1.1-mars-opt0',
        'experiment_results/TinyLlama_v1.1-mars-opt1'
    ], [
        'LoRA',
        'LoRA-XS',
        'LoHA',
        'VB LoRA',
        'MARS OPT0',
        'MARS OPT1'
    ], 'non_quant_table.tex')

def quant_table():
    generate_peft_latex_table([
        'experiment_results/TinyLlama_v1.1-loraq4',
        # TODO: LoRA 8
        'experiment_results/TinyLlama_v1.1-qlora',
        'experiment_results/TinyLlama_v1.1-qmars',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'
    ], [
        'LoRA (int4)',
        'QLoRA',
        'QMARS',
        'MARS OPT3 (int4)',
        'MARS OPT3 (int8)',
        'MARS OPT4 (int4)',
        'MARS OPT4 (int8)'
    ], 'quant_table.tex')


# Generate training runtime table  
def runtime_table(peft_dirs, peft_names):
    generate_peft_runtime_table(peft_dirs, peft_names,
                            output_filename='training_time_table.tex')
    
QUANT_PEFT_DIRS = [
        'experiment_results/TinyLlama_v1.1-loraq4',
        # TODO: LoRA 8
        'experiment_results/TinyLlama_v1.1-qlora',
        'experiment_results/TinyLlama_v1.1-qmars',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'
    ]
QUANT_PEFT_NAMES = [
        'LoRA (int4)',
        # 'LoRA (int8)',
        'QLoRA',
        'QMARS',
        'MARS OPT3 (int4)',
        'MARS OPT3 (int8)',
        'MARS OPT4 (int4)',
        'MARS OPT4 (int8)'
    ]

NON_QUANT_PEFT_DIRS = [
    'experiment_results/TinyLlama_v1.1-lora',
        'experiment_results/TinyLlama_v1.1-lora_xs',
        'experiment_results/TinyLlama_v1.1-loha',
        'experiment_results/TinyLlama_v1.1-vb_lora',
        'experiment_results/TinyLlama_v1.1-mars-opt0',
        'experiment_results/TinyLlama_v1.1-mars-opt1'
]

NON_QUANT_PEFT_NAMES = [
    'LoRA',
    'LoRA-XS',
    'LoHA',
    'VB LoRA',
    'MARS OPT0',
    'MARS OPT1'
]

ABLATION_PEFT_DIRS = [
    'experiment_results/TinyLlama_v1.1-abl_A',
    'experiment_results/TinyLlama_v1.1-abl_B',
    'experiment_results/TinyLlama_v1.1-abl_C',
    'experiment_results/TinyLlama_v1.1-abl_D'
]

ABLATION_PEFT_NAMES = [
    'Variant A',
    'Variant B',
    'Variant C',
    'Variant D'
]

#runtime_table(QUANT_PEFT_DIRS, QUANT_PEFT_NAMES)
#runtime_table(NON_QUANT_PEFT_DIRS, NON_QUANT_PEFT_NAMES)
#runtime_table(ABLATION_PEFT_DIRS, ABLATION_PEFT_NAMES)
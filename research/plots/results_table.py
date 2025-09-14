import json
import os
import glob
from pathlib import Path
import subprocess

def compile_and_crop_latex(output_filename):
    """
    Compile LaTeX to PDF using pdflatex and crop the result using pdfcrop.
    
    Args:
        output_filename (str): The LaTeX filename (with .tex extension)
    """
    # Get the base filename without extension
    base_name = Path(output_filename).stem
    pdf_name = f"{base_name}.pdf"
    cropped_pdf_name = f"{base_name}-crop.pdf"
    
    try:
        # Step 1: Compile LaTeX to PDF using pdflatex
        print(f"Compiling {output_filename} to PDF...")
        result = subprocess.run([
            'pdflatex', 
            '-interaction=nonstopmode',  # Don't stop for errors
            '-output-directory=.',       # Output to current directory
            output_filename
        ], capture_output=True, text=True, check=True)
        
        print(f"✓ PDF compilation successful: {pdf_name}")
        
        # Step 2: Crop the PDF using pdfcrop
        print(f"Cropping {pdf_name}...")
        result = subprocess.run([
            'pdfcrop',
            pdf_name,
            cropped_pdf_name
        ], capture_output=True, text=True, check=True)
        
        print(f"✓ PDF cropping successful: {cropped_pdf_name}")
        
        # Step 3: Delete the uncropped file and rename cropped file
        if os.path.exists(pdf_name):
            os.remove(pdf_name)
            print(f"✓ Deleted uncropped file: {pdf_name}")
        
        if os.path.exists(cropped_pdf_name):
            os.rename(cropped_pdf_name, pdf_name)
            print(f"✓ Renamed cropped file to: {pdf_name}")
        
        # Clean up auxiliary LaTeX files
        aux_extensions = ['.aux', '.log', '.out', '.fdb_latexmk', '.fls']
        for ext in aux_extensions:
            aux_file = f"{base_name}{ext}"
            if os.path.exists(aux_file):
                os.remove(aux_file)
        
        print(f"✓ Final cropped PDF available as: {pdf_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during compilation/cropping:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except FileNotFoundError as e:
        print(f"✗ Command not found. Please ensure pdflatex and pdfcrop are installed:")
        print(f"Error: {e}")

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
    benchmark_labels = ['ARC-E', 'ARC-C', 'Winogrande', 'BoolQ', 'LogiQA', 'HellaSwag']
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
    latex_content.append(r"\usepackage[margin=0cm]{geometry}")
    latex_content.append(r"\pagestyle{empty}")
    latex_content.append(r"\begin{document}")
    
    latex_content.append("")
    
    # Table caption and label
    latex_content.append(r"\begin{table}[h!]")
    
    # Table structure - calculate number of columns
    num_cols = 1 + len(benchmarks) + 1  # Rank/Params + benchmarks + Average
    col_spec = "|l|" + "c" * len(benchmarks) + "|c|"
    
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\hline")
    
    # Header row
    header = "\\textbf{Rank/Params} & "
    header += " & ".join([f"\\textbf{{{label}}}" for label in benchmark_labels])
    header += " & \\textbf{Average} \\\\"
    latex_content.append(header)
    latex_content.append("\hline")
    
    # Data rows
    method_averages = []
    
    for peft_name in peft_names:
        if peft_name in results:
            # Get the ranks for this specific method
            if peft_name == 'LoRA-XS':
                current_method_ranks = [16, 64, 256]
            else:
                current_method_ranks = ranks
            
            # Method header row with enhanced spacing
            #latex_content.append(f"\\multicolumn{{{num_cols}}}{{|c|}}{{}} \\\\[0.3pt]")  # Empty row with spacing above
            method_header = f"\\multicolumn{{{num_cols}}}{{|c|}}{{\\raisebox{{0pt}}[15pt][7pt]{{\\textbf{{\\Large {peft_name}}}}}}} \\\\"
            latex_content.append(method_header)
            latex_content.append("\hline")
            
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
                    rank_cell = f"$r={rank}$ ({params_str})"
                    score_cells = " & ".join([f"{score:.3f}" if score > 0 else "-" for score in scores])
                    avg_cell = f"{rank_avg:.3f}" if rank_avg > 0 else "-"
                    
                    row = f"{rank_cell} & {score_cells} & {avg_cell} \\\\"
                    latex_content.append(row)
            
            # Calculate method average
            method_avg = sum(rank_averages) / len(rank_averages) if rank_averages else 0.0
            method_averages.append(method_avg)
            
            # Add method average row with top border
            avg_scores = []
            for benchmark in benchmarks:
                benchmark_scores = [results[peft_name][rank][benchmark] for rank in current_method_ranks 
                                  if rank in results[peft_name] and results[peft_name][rank][benchmark] > 0]
                benchmark_avg = sum(benchmark_scores) / len(benchmark_scores) if benchmark_scores else 0.0
                avg_scores.append(benchmark_avg)
            
            avg_row_cells = " & ".join([f"{score:.3f}" if score > 0 else "-" for score in avg_scores])
            avg_row = f"\\textbf{{Average}} & {avg_row_cells} & \\textbf{{{method_avg:.3f}}}" + r"\\"
            latex_content.append("\\cline{1-" + str(num_cols) + "}")  # Top border for average row
            latex_content.append(avg_row)
            latex_content.append(r"\hline")
    
    # Table footer
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    latex_content.append("")
    latex_content.append(r"\end{document}")
    
    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX table saved as: {output_filename}")

    compile_and_crop_latex(output_filename)
    
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
    benchmark_labels = ['ARC-E', 'ARC-C', 'Winogrande', 'BoolQ', 'LogiQA', 'HellaSwag']
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
    latex_content.append(r"\usepackage[margin=0cm]{geometry}")
    latex_content.append(r"\pagestyle{empty}")
    latex_content.append(r"\begin{document}")
    latex_content.append("")
    
    # Table caption and label
    latex_content.append(r"\begin{table}[h!]")
    
    # Table structure - calculate number of columns (no method column)
    num_cols = 1 + len(benchmarks) + 1  # Rank/Params + benchmarks + Average
    col_spec = "|l|" + "c" * len(benchmarks) + "|c|"
    
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\hline")
    
    # Header row
    header = "\\textbf{Rank/Params} & "
    header += " & ".join([f"\\textbf{{{label}}}" for label in benchmark_labels])
    header += " & \\textbf{Average} \\\\"
    latex_content.append(header)
    latex_content.append("\\hline")
    
    # Data rows
    method_averages = []
    
    for peft_name in peft_names:
        if peft_name in results:
            # Get the ranks for this specific method
            if peft_name == 'LoRA-XS':
                current_method_ranks = [16, 64, 256]
            else:
                current_method_ranks = ranks
            
            # Method header row with enhanced spacing using struts
            method_header = f"\\multicolumn{{{num_cols}}}{{|c|}}{{\\raisebox{{0pt}}[15pt][7pt]{{\\textbf{{\\Large {peft_name}}}}}}} \\\\"
            latex_content.append(method_header)
            latex_content.append("\\hline")
            
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
                    rank_cell = f"$r={rank}$ ({params_str})"
                    runtime_cells = " & ".join([f"{runtime:.1f}" if runtime > 0 else "-" for runtime in runtimes])
                    avg_cell = f"{rank_avg:.1f}" if rank_avg > 0 else "-"
                    
                    row = f"{rank_cell} & {runtime_cells} & {avg_cell} \\\\"
                    latex_content.append(row)
            
            # Calculate method average
            method_avg = sum(rank_averages) / len(rank_averages) if rank_averages else 0.0
            method_averages.append(method_avg)
            
            # Add method average row with top border
            avg_runtimes = []
            for benchmark in benchmarks:
                benchmark_runtimes = [results[peft_name][rank][benchmark] for rank in current_method_ranks 
                                    if rank in results[peft_name] and results[peft_name][rank][benchmark] > 0]
                benchmark_avg = sum(benchmark_runtimes) / len(benchmark_runtimes) if benchmark_runtimes else 0.0
                avg_runtimes.append(benchmark_avg)
            
            avg_row_cells = " & ".join([f"{runtime:.1f}" if runtime > 0 else "-" for runtime in avg_runtimes])
            avg_row = f"\\textbf{{Average}} & {avg_row_cells} & \\textbf{{{method_avg:.1f}}} \\\\"
            latex_content.append("\\cline{1-" + str(num_cols) + "}")  # Top border for average row
            latex_content.append(avg_row)
            latex_content.append("\\hline")
    
    # Table footer
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    latex_content.append("\\end{document}")
    
    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX runtime table saved as: {output_filename}")

    compile_and_crop_latex(output_filename)
    
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
        'experiment_results/TinyLlama_v1.1-abl_D',
        'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
        'experiment_results/TinyLlama_v1.1-loraq4'
    ], [
        'Variant A - Intermediate layer adaptation',
    'Variant B - Adapter vector with frozen A matrix',
    'Variant C - Frozen A matrix with intermediate layer',
    'Variant D - Shared and frozen A matrix',
    'Variant E - LoRA (int8)',
    'Variant F - LoRA (fp4)'
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
        'MARS OPT3 (fp4)',
        'MARS OPT3 (int8)',
        'MARS OPT4 (fp4)',
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
        'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
        'experiment_results/TinyLlama_v1.1-qlora',
        'experiment_results/TinyLlama_v1.1-qmars',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'
    ], [
        'LoRA (fp4)',
        'LoRA (int8)',
        'QLoRA',
        'QMARS',
        'MARS OPT3 (fp4)',
        'MARS OPT3 (int8)',
        'MARS OPT4 (fp4)',
        'MARS OPT4 (int8)'
    ], 'quant_table.tex')


# Generate training runtime table  
def runtime_table(peft_dirs, peft_names, output):
    generate_peft_runtime_table(peft_dirs, peft_names,
                            output_filename=output)
    
QUANT_PEFT_DIRS = [
        'experiment_results/TinyLlama_v1.1-loraq4',
        'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
        'experiment_results/TinyLlama_v1.1-qlora',
        'experiment_results/TinyLlama_v1.1-qmars',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt3-q8',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q4',
        'experiment_results/TinyLlama_v1.1-mars-opt4-q8'
    ]
QUANT_PEFT_NAMES = [
        'LoRA (fp4)',
        'LoRA (int8)',
        'QLoRA',
        'QMARS',
        'MARS Q-OPT0 (fp4)',
        'MARS Q-OPT0 (int8)',
        'MARS Q-OPT1 (fp4)',
        'MARS Q-OPT1 (int8)',
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
    'experiment_results/TinyLlama_v1.1-abl_D',
    'experiment_results/TinyLlama_v1.1-abl_G-loraq8',
    'experiment_results/TinyLlama_v1.1-loraq4'
    
]

ABLATION_PEFT_NAMES = [
    'Variant A - Intermediate layer adaptation',
    'Variant B - Adapter vector with frozen A matrix',
    'Variant C - Frozen A matrix with intermediate layer',
    'Variant D - Shared and frozen A matrix',
    'Variant E - LoRA (int8)',
    'Variant F - LoRA (fp4)'
]

quant_table()
#non_quant_table()
#ablation_table()
#mars_table()

runtime_table(QUANT_PEFT_DIRS, QUANT_PEFT_NAMES, 'quant_time_table.tex')
#runtime_table(NON_QUANT_PEFT_DIRS, NON_QUANT_PEFT_NAMES, 'non_quant_time_table.tex')
#runtime_table(ABLATION_PEFT_DIRS, ABLATION_PEFT_NAMES, 'ablation_time_table.tex')
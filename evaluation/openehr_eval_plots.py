import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Set font parameters for PDF export
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# UPDATE THESE WITH YOUR ACTUAL FILES
model_json_pairs = [
    ("TinyLlama", "data/ehr_eval/medical_llm_evaluation_tinyllama_complex.json"),
    ("Phi-3-Mini-4k", "data/ehr_eval/medical_llm_evaluation_phi3_complex.json"),
]

def load_evaluation_data(model_json_pairs):
    """
    Load evaluation data from JSON files.
    
    Args:
        model_json_pairs: List of tuples (slm_model_name, json_file_path)
    
    Returns:
        Dictionary with model names as keys and evaluation data as values
    """
    evaluation_data = {}
    
    for model_name, json_file in model_json_pairs:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                evaluation_data[model_name] = data['evaluation_results']
                print(f"Loaded data for {model_name} from {json_file}")
        except FileNotFoundError:
            print(f"Warning: File {json_file} not found for model {model_name}")
        except KeyError:
            print(f"Warning: Invalid JSON structure in {json_file} for model {model_name}")
    
    return evaluation_data

def create_comparison_plot(evaluation_data, comparison_type, title, filename):
    """
    Create a comparison plot for SLM vs LLM performance.
    
    Args:
        evaluation_data: Dictionary with evaluation results
        comparison_type: Either 'slm_vs_llm_document' or 'slm_vs_llm_chunked'
        title: Plot title
        filename: Output filename
    """
    models = list(evaluation_data.keys())
    faithfulness_scores = []
    clinical_quality_scores = []
    
    # Extract scores for each model
    for model in models:
        if comparison_type in evaluation_data[model]:
            data = evaluation_data[model][comparison_type]
            faithfulness_scores.append(data['faithfulness']['average'])
            clinical_quality_scores.append(data['clinical_quality']['average'])
        else:
            print(f"Warning: {comparison_type} not found for model {model}")
            faithfulness_scores.append(0)
            clinical_quality_scores.append(0)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar positions
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, faithfulness_scores, width, 
                   label='Faithfulness', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, clinical_quality_scores, width, 
                   label='Clinical Quality', color='lightcoral', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('SLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    ax.set_ylim(0, max(max(faithfulness_scores), max(clinical_quality_scores)) * 1.1)
    
    # Add value labels on bars
    def add_value_labels(bars, scores):
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    add_value_labels(bars1, faithfulness_scores)
    add_value_labels(bars2, clinical_quality_scores)
    
    # Add threshold lines
    #ax.axhline(y=0.9, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)
    #ax.axhline(y=7.0, color='lightcoral', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add threshold labels
    #ax.text(len(models)-0.1, 0.92, 'Faithfulness Threshold (0.9)', 
    #        fontsize=9, ha='right', va='bottom', color='steelblue')
    #ax.text(len(models)-0.1, 7.2, 'Clinical Quality Threshold (7.0)', 
    #        fontsize=9, ha='right', va='bottom', color='lightcoral')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n{title} - Summary Statistics:")
    print("-" * 50)
    for i, model in enumerate(models):
        print(f"{model}:")
        print(f"  Faithfulness: {faithfulness_scores[i]:.3f}")
        print(f"  Clinical Quality: {clinical_quality_scores[i]:.2f}")
        faithfulness_pass = "✓" if faithfulness_scores[i] >= 0.9 else "✗"
        clinical_pass = "✓" if clinical_quality_scores[i] >= 7.0 else "✗"
        print(f"  Thresholds: Faithfulness {faithfulness_pass}, Clinical Quality {clinical_pass}")
        print()

def create_summary_table(evaluation_data):
    """Create a summary table of all results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    print(f"{'Model':<15} {'Document':<20} {'Chunked':<20}")
    print(f"{'':15} {'Faith':<8} {'Clin':<8} {'Faith':<8} {'Clin':<8}")
    print("-" * 65)
    
    for model in evaluation_data.keys():
        doc_data = evaluation_data[model].get('slm_vs_llm_document', {})
        chunk_data = evaluation_data[model].get('slm_vs_llm_chunked', {})
        
        doc_faith = doc_data.get('faithfulness', {}).get('average', 0)
        doc_clin = doc_data.get('clinical_quality', {}).get('average', 0)
        chunk_faith = chunk_data.get('faithfulness', {}).get('average', 0)
        chunk_clin = chunk_data.get('clinical_quality', {}).get('average', 0)
        
        print(f"{model:<15} {doc_faith:<8.3f} {doc_clin:<8.2f} {chunk_faith:<8.3f} {chunk_clin:<8.2f}")

def main():
    """
    Main function to generate plots from evaluation JSON files.
    
    Usage:
        # Define your model-json pairs
        model_json_pairs = [
            ("Llama-3B", "evaluation_results_llama3b.json"),
            ("Phi-3-Mini", "evaluation_results_phi3.json"),
            ("Gemma-2B", "evaluation_results_gemma2b.json"),
        ]
        
        # Run the plotting script
        plot_evaluation_results(model_json_pairs)
    """
    
    
    print("Medical LLM Evaluation Results Plotting")
    print("=" * 50)
    
    # Load evaluation data
    evaluation_data = load_evaluation_data(model_json_pairs)
    
    if not evaluation_data:
        print("No evaluation data loaded. Please check your file paths.")
        return
    
    # Create plots
    print(f"\nGenerating plots for {len(evaluation_data)} models...")
    
    # Plot 1: SLM vs LLM (Document Context)
    create_comparison_plot(
        evaluation_data=evaluation_data,
        comparison_type='slm_vs_llm_document',
        title='SLM vs LLM Performance Comparison (Full Document Context)',
        filename='slm_vs_llm_document_comparison.pdf'
    )
    
    # Plot 2: SLM vs LLM (Chunked Context)
    create_comparison_plot(
        evaluation_data=evaluation_data,
        comparison_type='slm_vs_llm_chunked',
        title='SLM vs LLM Performance Comparison (Chunked Context)',
        filename='slm_vs_llm_chunked_comparison.pdf'
    )
    
    # Create summary table
    create_summary_table(evaluation_data)
    
    print("\nPlots generated successfully!")
    print("Files created:")
    print("- slm_vs_llm_document_comparison.pdf")
    print("- slm_vs_llm_chunked_comparison.pdf")

def plot_evaluation_results(model_json_pairs):
    """
    Convenient function to plot results with custom model-json pairs.
    
    Args:
        model_json_pairs: List of tuples (model_name, json_file_path)
    """
    evaluation_data = load_evaluation_data(model_json_pairs)
    
    if not evaluation_data:
        print("No evaluation data loaded. Please check your file paths.")
        return
    
    # Create plots
    create_comparison_plot(
        evaluation_data=evaluation_data,
        comparison_type='slm_vs_llm_document',
        title='SLM vs LLM Performance Comparison (Full Document Context)',
        filename='slm_vs_llm_document_comparison.pdf'
    )
    
    create_comparison_plot(
        evaluation_data=evaluation_data,
        comparison_type='slm_vs_llm_chunked', 
        title='SLM vs LLM Performance Comparison (Chunked Context)',
        filename='slm_vs_llm_chunked_comparison.pdf'
    )
    
    create_summary_table(evaluation_data)

if __name__ == "__main__":
    main()
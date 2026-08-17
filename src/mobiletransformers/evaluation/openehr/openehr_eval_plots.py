"""
Script for generating scatter plots comparing faithfulness and clinical quality from OpenEHR evaluation JSON files.
"""

import json

import matplotlib
import matplotlib.pyplot as plt

# Set font parameters for PDF export
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

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
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                evaluation_data[model_name] = data["evaluation_results"]
                print(f"Loaded data for {model_name} from {json_file}")
        except FileNotFoundError:
            print(f"Warning: File {json_file} not found for model {model_name}")
        except KeyError:
            print(f"Warning: Invalid JSON structure in {json_file} for model {model_name}")

    return evaluation_data


def create_scatter_plot(evaluation_data, title, filename):
    """
    Create a scatter plot showing faithfulness vs clinical quality for all models and contexts.

    Args:
        evaluation_data: Dictionary with evaluation results
        title: Plot title
        filename: Output filename
    """
    # Set up the plot with appropriate figure size
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define colors for each model (you can expand this list as needed)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    # Define markers for document vs chunked
    document_marker = "o"  # circle
    chunked_marker = "s"  # square

    # Track data for legend
    model_handles = []
    context_handles = []

    # Extract and plot data for each model
    for i, model in enumerate(evaluation_data.keys()):
        color = colors[i % len(colors)]

        # Process document context data
        if "slm_vs_llm_document" in evaluation_data[model]:
            doc_data = evaluation_data[model]["slm_vs_llm_document"]
            doc_faithfulness = doc_data["faithfulness"]["average"] * 10  # Scale 0-1 to 0-10
            doc_clinical = doc_data["clinical_quality"]["average"] * 10  # Scale 0-1 to 0-10

            scatter_doc = ax.scatter(
                doc_faithfulness,
                doc_clinical,
                c=color,
                marker=document_marker,
                s=200,
                alpha=0.8,
                edgecolors="black",
                linewidth=1,
            )

        # Process chunked context data
        if "slm_vs_llm_chunked" in evaluation_data[model]:
            chunk_data = evaluation_data[model]["slm_vs_llm_chunked"]
            chunk_faithfulness = chunk_data["faithfulness"]["average"] * 10  # Scale 0-1 to 0-10
            chunk_clinical = chunk_data["clinical_quality"]["average"] * 10  # Scale 0-1 to 0-10

            scatter_chunk = ax.scatter(
                chunk_faithfulness,
                chunk_clinical,
                c=color,
                marker=chunked_marker,
                s=200,
                alpha=0.8,
                edgecolors="black",
                linewidth=1,
            )

    # Customize the plot with larger fonts
    ax.set_xlabel("Faithfulness Score (0-10)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Clinical Quality Score (0-10)", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)

    # Set axis limits and ticks
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Create custom legend
    # Legend for context types
    from matplotlib.lines import Line2D

    context_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=document_marker,
            color="gray",
            linestyle="None",
            markersize=12,
            markerfacecolor="gray",
            markeredgecolor="black",
            label="Document Context",
        ),
        Line2D(
            [0],
            [0],
            marker=chunked_marker,
            color="gray",
            linestyle="None",
            markersize=12,
            markerfacecolor="gray",
            markeredgecolor="black",
            label="Chunked Context",
        ),
    ]

    # Legend for models
    model_legend_elements = []
    for i, model in enumerate(evaluation_data.keys()):
        color = colors[i % len(colors)]
        model_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                linestyle="None",
                markersize=12,
                markerfacecolor=color,
                markeredgecolor="black",
                label=model,
            )
        )

    # Create two separate legends
    context_legend = ax.legend(
        handles=context_legend_elements,
        title="Context Type",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        bbox_to_anchor=(1.45, 1.0),
    )
    context_legend.get_title().set_fontweight("bold")

    model_legend = ax.legend(
        handles=model_legend_elements,
        title="Models",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        bbox_to_anchor=(1.45, 0.7),
    )
    model_legend.get_title().set_fontweight("bold")

    # Add the first legend back (matplotlib removes it when adding the second)
    ax.add_artist(context_legend)

    # Add reference lines at common thresholds
    ax.axhline(y=7.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=9.0, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    # Add threshold labels
    ax.text(0.2, 7.2, "Clinical Quality Threshold", fontsize=12, color="red", alpha=0.7, fontweight="bold")
    ax.text(
        9.2,
        0.5,
        "Faithfulness Threshold",
        fontsize=12,
        color="blue",
        alpha=0.7,
        fontweight="bold",
        rotation=90,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print(f"\n{title} - Summary Statistics:")
    print("-" * 60)
    for model in evaluation_data.keys():
        print(f"{model}:")

        if "slm_vs_llm_document" in evaluation_data[model]:
            doc_data = evaluation_data[model]["slm_vs_llm_document"]
            doc_faithfulness = doc_data["faithfulness"]["average"] * 10
            doc_clinical = doc_data["clinical_quality"]["average"] * 10
            print(
                f"  Document Context - Faithfulness: {doc_faithfulness:.2f}, Clinical Quality: {doc_clinical:.2f}"
            )

        if "slm_vs_llm_chunked" in evaluation_data[model]:
            chunk_data = evaluation_data[model]["slm_vs_llm_chunked"]
            chunk_faithfulness = chunk_data["faithfulness"]["average"] * 10
            chunk_clinical = chunk_data["clinical_quality"]["average"] * 10
            print(
                f"  Chunked Context  - Faithfulness: {chunk_faithfulness:.2f}, Clinical Quality: {chunk_clinical:.2f}"
            )

        print()


def main():
    """
    Main function to generate scatter plot from evaluation JSON files.
    """
    print("Medical LLM Evaluation Results - Scatter Plot Generation")
    print("=" * 60)

    # Load evaluation data
    evaluation_data = load_evaluation_data(model_json_pairs)

    if not evaluation_data:
        print("No evaluation data loaded. Please check your file paths.")
        return

    # Create scatter plot
    print(f"\nGenerating scatter plot for {len(evaluation_data)} models...")

    create_scatter_plot(
        evaluation_data=evaluation_data,
        title="SLM Performance: Faithfulness vs Clinical Quality Comparison",
        filename="slm_performance_scatter_plot.pdf",
    )

    print("\nScatter plot generated successfully!")
    print("File created: slm_performance_scatter_plot.pdf")


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

    # Create scatter plot
    create_scatter_plot(
        evaluation_data=evaluation_data,
        title="SLM Performance: Faithfulness vs Clinical Quality Comparison",
        filename="slm_performance_scatter_plot.pdf",
    )


if __name__ == "__main__":
    main()

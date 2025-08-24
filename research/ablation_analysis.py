import numpy as np
from research.utils import load_peft_metrics


def analyze_training_metrics(save_path="peft_metrics.npz"):
    """Complete workflow for saving and analyzing metrics"""
    
    # 2. Load and organize for analysis  
    analysis_data = load_peft_metrics(save_path)
    
    # 3. Print summary
    print("\n=== METRICS SUMMARY ===")
    for layer_name, layer_metrics in analysis_data['layers'].items():
        print(f"\nLayer: {layer_name}")
        for metric_name, metric_data in layer_metrics.items():
            if isinstance(metric_data, np.ndarray):
                print(f"  {metric_name}: shape {metric_data.shape}")
                if metric_data.dtype in [np.float32, np.float64]:
                    print(f"    Range: [{np.min(metric_data):.4f}, {np.max(metric_data):.4f}]")
    
    return analysis_data


if __name__ == "__main__":

    analyze_training_metrics("experiment_results/TinyLlama_v1.1-abl_A-boolq-r32-a2/analysis_metrics.npz")
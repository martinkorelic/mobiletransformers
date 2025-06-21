from ast import Dict, List, Tuple
import json
import os
import psutil
from pyparsing import Any, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from torch.nn.functional import cosine_similarity
from scipy.stats import wasserstein_distance

from transformers import TrainerCallback
import time

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

class MemoryUsageCallback(TrainerCallback):
    """Logs GPU, CPU, and PyTorch-allocated CPU memory usage during training."""
    def __init__(self):
        super().__init__()
        self.gpu_mem_before_fwd = 0
        self.gpu_mem_after_fwd = 0
        self.gpu_mem_after_bwd = 0
        self.gpu_peak_mem_fwd = 0
        self.gpu_peak_mem_bwd = 0
        self.cpu_mem_after_bwd = 0
        self.cpu_mem_after_fwd = 0
        self.cpu_mem_before_fwd = 0
        self.torch_cpu_mem_before = 0
        self.torch_cpu_mem_after_fwd = 0
        self.torch_cpu_mem_after_bwd = 0

    def _get_memory_usage(self):
        """Helper function to get GPU, CPU, and PyTorch-allocated CPU memory usage."""
        gpu_mem, gpu_peak_mem = (0.0, 0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure accurate measurements
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            gpu_peak_mem = torch.cuda.max_memory_allocated() / 1e9
        cpu_mem = psutil.Process().memory_info().rss / 1e9  # Convert to GB
        return gpu_mem, gpu_peak_mem, cpu_mem

    def on_step_begin(self, args, state, control, **kwargs):
        """Capture memory before forward pass."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # Reset peak tracking
        self.gpu_mem_before_fwd, _, self.cpu_mem_before_fwd = self._get_memory_usage()

    def on_backward_begin(self, args, state, control, **kwargs):
        """Capture memory just before backward pass (when activations are present)."""
        self.gpu_mem_after_fwd, self.gpu_peak_mem_fwd, self.cpu_mem_after_fwd, self.torch_cpu_mem_after_fwd = self._get_memory_usage()

    def on_backward_end(self, args, state, control, **kwargs):
        """Capture memory after backward pass."""
        self.gpu_mem_after_bwd, self.gpu_peak_mem_bwd, self.cpu_mem_after_bwd, self.torch_cpu_mem_after_bwd = self._get_memory_usage()

    def on_log(self, args, state, control, **kwargs):
        """Logs memory usage at the end of each step."""
        print(f"\n[Memory Usage] Model training memory:")
        print(f"  GPU: {self.gpu_mem_before_fwd:.2f} GB")
        print(f"  CPU: {self.cpu_mem_before_fwd:.2f} GB")
        #print(f"  PyTorch CPU Allocated: {self.torch_cpu_mem_before:.2f} GB -> {self.torch_cpu_mem_after_bwd:.2f} GB")
        #print(f"  - After Forward   | GPU: {self.gpu_mem_after_fwd:.2f} GB (Peak: {self.gpu_peak_mem_fwd:.2f} GB), CPU: {self.cpu_mem_after_fwd:.2f} GB")
        #print(f"  - After Backward  | GPU: {self.gpu_mem_after_bwd:.2f} GB (Peak: {self.gpu_peak_mem_bwd:.2f} GB), CPU: {self.cpu_mem_after_bwd:.2f} GB\n")

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

class LogStepTimerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_log(self, args, state, control, **kwargs):
        """
        Called at the end of a logging step.
        """
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            print(f"Time for log step {state.global_step}: {elapsed_time:.4f} seconds")
        self.start_time = time.time()

class LoRAWeightVisualizerCallback(TrainerCallback):
    def __init__(self, model, save_path="./lora_weight_changes.json", figs_path="./trainer_visualization", seed=42, log_steps=200, lora_targets=None, adapter_names=["lora_A", "lora_B"]):
        self.model = model
        self.save_path = save_path
        self.figs_path = figs_path
        self.log_steps = log_steps
        self.seed = seed
        self.initial_weights = {}
        self.metrics = {}
        self.lora_targets = lora_targets or []

        self.adapter_names = adapter_names

        self.adapter_names = ["transform_up_latent", "transform_down_latent"]

        # Set the seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Capture initial LoRA weights
        self._capture_initial_weights()

    def _capture_initial_weights(self):
        """Capture the initial weights of the LoRA layers."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # LoRA layers are trainable
                self.initial_weights[name] = param.detach().clone()

    def _compute_metrics(self, initial, current):
        """Compute metrics such as cosine similarity, mean magnitude, and variance."""
        flat_initial = initial.view(-1).to(current.device)
        flat_current = current.view(-1)

        cosine_sim = cosine_similarity(flat_initial.unsqueeze(0), flat_current.unsqueeze(0), dim=1).item()
        mean_magnitude_initial = torch.mean(flat_initial).item()
        mean_magnitude_current = torch.mean(flat_current).item()
        variance_initial = torch.var(flat_initial).item()
        variance_current = torch.var(flat_current).item()

        # Distribution metrics
        initial_numpy = flat_initial.cpu().numpy()
        current_numpy = flat_current.cpu().numpy()
        
        wasserstein_dist = wasserstein_distance(initial_numpy, current_numpy)

        return {
            "cosine_similarity": cosine_sim,
            "mean_magnitude_initial": mean_magnitude_initial,
            "mean_magnitude_current": mean_magnitude_current,
            "variance_initial": variance_initial,
            "variance_current": variance_current,
            "wasserstein_distance": wasserstein_dist
        }

    def _save_heatmap(self, initial, current, layer_name, step):
        """Save a heatmap of the differences between initial and current weights."""
        differences = (current - initial.to(current.device)).cpu().numpy()

        # Compute mean and variance differences for rows and columns
        row_means = np.mean(differences, axis=1)
        col_means = np.mean(differences, axis=0)
        row_variances = np.var(differences, axis=1)
        col_variances = np.var(differences, axis=0)

        # Create the directory for this step
        step_dir = os.path.join(self.figs_path, f"trainer_visualization_{step}")
        os.makedirs(step_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Heatmap
        im = axes[0, 0].imshow(differences, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=axes[0, 0])
        axes[0, 0].set_title(f"Weight Differences Heatmap: {layer_name} (Step {step})")
        axes[0, 0].set_xlabel("Columns")
        axes[0, 0].set_ylabel("Rows")

        # Row means
        axes[0, 1].plot(row_means)
        axes[0, 1].set_title("Mean Differences per Row")
        axes[0, 1].set_xlabel("Row Index")
        axes[0, 1].set_ylabel("Mean Difference")

        # Column means
        axes[1, 0].plot(col_means)
        axes[1, 0].set_title("Mean Differences per Column")
        axes[1, 0].set_xlabel("Column Index")
        axes[1, 0].set_ylabel("Mean Difference")

        # Row variances
        axes[1, 1].plot(row_variances)
        axes[1, 1].set_title("Variance Differences per Row")
        axes[1, 1].set_xlabel("Row Index")
        axes[1, 1].set_ylabel("Variance")

        # Column variances
        axes[2, 0].plot(col_variances)
        axes[2, 0].set_title("Variance Differences per Column")
        axes[2, 0].set_xlabel("Column Index")
        axes[2, 0].set_ylabel("Variance")

        # Empty subplot
        axes[2, 1].axis('off')

        # Save the figure
        file_path = os.path.join(step_dir, f"{layer_name.replace('.', '_')}_heatmap.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def _save_metrics_figure(self, step):
        """Save a figure showing metrics for all layers at this step."""
        step_metrics = self.metrics[step]

        layers = list(step_metrics.keys())
        cosine_similarities = [step_metrics[layer]["cosine_similarity"] for layer in layers]
        mean_magnitudes_current = [step_metrics[layer]["mean_magnitude_current"] for layer in layers]
        var_currents = [step_metrics[layer]["variance_current"] for layer in layers]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(layers, cosine_similarities, label="Cosine Similarity", marker='o')
        ax.plot(layers, mean_magnitudes_current, label="Mean Magnitude (Current)", marker='o')
        ax.plot(layers, var_currents, label="Variance (Current)", marker='o')

        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_title(f"Metrics Overview (Step {step})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Metric Value")
        ax.legend()

        step_dir = os.path.join(self.figs_path, f"trainer_visualization_{step}")
        file_path = os.path.join(step_dir, "metrics_overview.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def _extract_layer_index(self, name):
        """Extract the layer index from the parameter name."""
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.isdigit():
                return int(part)
        return -1  # Default to -1 if no layer index is found

    def _save_metrics_by_layer(self, step):
        """Save figures tracking metrics across decoder layers for specific LoRA targets."""
        step_metrics = self.metrics[step]
        layers_data = {}

        # Organize metrics by target type and layer index
        for name, metrics in step_metrics.items():
            for target in self.lora_targets:
                if target in name:
                    layer_index = self._extract_layer_index(name)
                    if target not in layers_data:
                        layers_data[target] = {self.adapter_names[0]: [], self.adapter_names[1]: []}
                    if self.adapter_names[0] in name:
                        layers_data[target][self.adapter_names[0]].append((layer_index, metrics))
                    elif self.adapter_names[1] in name:
                        layers_data[target][self.adapter_names[1]].append((layer_index, metrics))

        # Create plots for each target type
        for target, matrices in layers_data.items():
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            for i, (matrix_type, data) in enumerate(matrices.items()):
                data.sort(key=lambda x: x[0])  # Sort by layer index
                layer_indices = [d[0] for d in data]
                cosine_similarities = [d[1]["cosine_similarity"] for d in data]
                mean_magnitudes = [d[1]["mean_magnitude_current"] for d in data]
                var_currents = [d[1]["variance_current"] for d in data]

                axes[i].plot(layer_indices, cosine_similarities, label="Cosine Similarity", marker='o')
                axes[i].plot(layer_indices, mean_magnitudes, label="Mean Magnitude", marker='o')
                axes[i].plot(layer_indices, var_currents, label="Variance", marker='o')

                axes[i].set_title(f"{target} - {matrix_type} Metrics")
                axes[i].set_xlabel("Layer Index")
                axes[i].set_ylabel("Metric Value")
                axes[i].legend()
                # Set x-axis ticks to display every layer index
                axes[i].set_xticks(layer_indices)

            plt.tight_layout()

            # Save the figure
            step_dir = os.path.join(self.figs_path, f"trainer_visualization_{step}")
            os.makedirs(step_dir, exist_ok=True)
            file_path = os.path.join(step_dir, f"metrics_{target}.png")
            plt.savefig(file_path)
            plt.close()

    def on_step_end(self, args, state, control, **kwargs):
        """Log changes in weights at specified intervals."""
        if state.global_step % self.log_steps == 0:
            step_metrics = {}

            print("Computing difference and visualization...")

            for name, param in self.model.named_parameters():
                if param.requires_grad:  # LoRA layers are trainable
                    initial = self.initial_weights[name]
                    current = param.detach()

                    # Compute metrics
                    step_metrics[name] = self._compute_metrics(initial, current)

                    # Save heatmap of weight differences
                    if len(initial.shape) == 2:  # Only for matrix-like parameters
                        self._save_heatmap(initial, current, name, state.global_step)

            self.metrics[state.global_step] = step_metrics

            # Save metrics JSON file
            step_dir = os.path.join(self.figs_path, f"trainer_visualization_{state.global_step}")
            os.makedirs(step_dir, exist_ok=True)
            metrics_path = os.path.join(step_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(step_metrics, f, indent=4)

            # Save metrics overview figures by target
            self._save_metrics_by_layer(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Save metrics to a JSON file at the end of training."""
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
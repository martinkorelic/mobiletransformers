import math

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

from mobiletransformers.peft.ablation.config import AblationConfig, AblationVariant


class AblationLayer(BaseTunerLayer):
    adapter_layer_names = ()

    def __init__(
        self,
        base_layer: nn.Module,
        ablation_config: AblationConfig,
        ablation_variant: AblationVariant,
        **kwargs,
    ) -> None:
        self.base_layer = base_layer

        self.up_project = nn.ParameterDict({})
        self.down_project = nn.ParameterDict({})

        self.ablation_variant = ablation_variant

        if self.ablation_variant == AblationVariant.VARIANT_0:
            pass
        elif self.ablation_variant == AblationVariant.VARIANT_A:
            self.intermediate = nn.ParameterDict({})
        elif self.ablation_variant == AblationVariant.VARIANT_B:
            self.input_vector = nn.ParameterDict({})
        elif self.ablation_variant == AblationVariant.VARIANT_C:
            self.intermediate = nn.ParameterDict({})
        elif self.ablation_variant == AblationVariant.VARIANT_D:
            self.intermediate = nn.ParameterDict({})

    def update_layer(
        self,
        original_weights,
        adapter_name,
        ablation_config: AblationConfig,
        ablation_variant: AblationVariant,
    ):

        self.ablation_variant = ablation_variant
        self.alpha = ablation_config.alpha / ablation_config.r

        init_weight = getattr(ablation_config, "init_weight", "kaiming")

        # Initialize A matrix
        self.down_project[adapter_name] = nn.Parameter(
            torch.empty(original_weights.in_features, ablation_config.r), requires_grad=True
        )

        if init_weight == "kaiming":
            torch.nn.init.kaiming_uniform_(self.down_project[adapter_name], a=math.sqrt(5))
        elif init_weight == "gaussian":
            torch.nn.init.normal_(self.down_project[adapter_name], mean=0.0, std=1.0 / ablation_config.r)
        else:
            raise ValueError(f"Unknown init_weight: {init_weight}. Use 'kaiming' or 'gaussian'")

        # Initialize B matrix (up_project) with zeros - this will be trained
        self.up_project[adapter_name] = nn.Parameter(
            torch.zeros(ablation_config.r, original_weights.out_features), requires_grad=True
        )

        if self.ablation_variant == AblationVariant.VARIANT_0:
            self.adapter_layer_names = ("down_project", "up_project")

        elif self.ablation_variant == AblationVariant.VARIANT_A:
            self.adapter_layer_names = ("down_project", "intermediate", "up_project")
            self.intermediate[adapter_name] = nn.Parameter(
                torch.empty(ablation_config.r, ablation_config.r), requires_grad=True
            )
            if init_weight == "kaiming":
                torch.nn.init.kaiming_uniform_(self.intermediate[adapter_name], a=math.sqrt(5))
            elif init_weight == "gaussian":
                torch.nn.init.normal_(self.intermediate[adapter_name], mean=0.0, std=1.0 / ablation_config.r)
            else:
                raise ValueError(f"Unknown init_weight: {init_weight}. Use 'kaiming' or 'gaussian'")

        elif self.ablation_variant == AblationVariant.VARIANT_B:
            self.adapter_layer_names = ("input_vector", "up_project")
            # Initialize input vector with random normal distribution
            self.input_vector[adapter_name] = nn.Parameter(
                torch.randn(original_weights.in_features), requires_grad=True
            )
            # Cannot compute kaiming for a single vector
            if init_weight == "kaiming":
                torch.nn.init.normal_(self.input_vector[adapter_name], mean=0.0, std=1.0 / ablation_config.r)
            elif init_weight == "gaussian":
                torch.nn.init.normal_(self.input_vector[adapter_name], mean=0.0, std=1.0 / ablation_config.r)
            else:
                raise ValueError(f"Unknown init_weight: {init_weight}. Use 'kaiming' or 'gaussian'")

            self.down_project[adapter_name].requires_grad = False
        elif self.ablation_variant == AblationVariant.VARIANT_C:
            self.adapter_layer_names = ("intermediate", "up_project")
            self.intermediate[adapter_name] = nn.Parameter(
                torch.empty(ablation_config.r, ablation_config.r), requires_grad=True
            )
            if init_weight == "kaiming":
                torch.nn.init.kaiming_uniform_(self.intermediate[adapter_name], a=math.sqrt(5))
            elif init_weight == "gaussian":
                torch.nn.init.normal_(self.intermediate[adapter_name], mean=0.0, std=1.0 / ablation_config.r)
            else:
                raise ValueError(f"Unknown init_weight: {init_weight}. Use 'kaiming' or 'gaussian'")
            self.down_project[adapter_name].requires_grad = False

        elif self.ablation_variant == AblationVariant.VARIANT_D:
            self.adapter_layer_names = ("intermediate", "up_project")
            self.intermediate[adapter_name] = nn.Parameter(
                torch.empty(ablation_config.r, ablation_config.r), requires_grad=True
            )
            if init_weight == "kaiming":
                torch.nn.init.kaiming_uniform_(self.intermediate[adapter_name], a=math.sqrt(5))
            elif init_weight == "gaussian":
                torch.nn.init.normal_(self.intermediate[adapter_name], mean=0.0, std=1.0 / ablation_config.r)
            else:
                raise ValueError(f"Unknown init_weight: {init_weight}. Use 'kaiming' or 'gaussian'")
            self.down_project[adapter_name].requires_grad = False

        # Variant E and F
        elif (
            self.ablation_variant == AblationVariant.VARIANT_E
            or self.ablation_variant == AblationVariant.VARIANT_F
        ):
            self.adapter_layer_names = ("down_project", "up_project")
        elif self.ablation_variant == AblationVariant.VARIANT_G:
            self.adapter_layer_names = ("down_project", "up_project")
            # Int8 quantized backbone (original weights)
            self.base_layer = ManualQuantizedLinear(original_weights, bits=8)
        elif self.ablation_variant == AblationVariant.VARIANT_H:
            self.adapter_layer_names = ("down_project", "up_project")
            # Int4 quantized backbone (original weights)
            self.base_layer = ManualQuantizedLinear(original_weights, bits=4)

        self.adapter_name = adapter_name

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)


class ManualQuantizedLinear(nn.Module):
    def __init__(self, original_linear, bits=8, symmetric=True, per_channel=True):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        self._quantize_weights(original_linear.weight.data)

        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data)
        else:
            self.bias = None

    def _quantize_weights(self, weight):
        """Quantize weights with scale and zero point"""
        if self.bits == 8:
            if self.symmetric:
                qmin, qmax = -127, 127  # Reserve -128 for symmetric
            else:
                qmin, qmax = -128, 127
            dtype = torch.int8
        elif self.bits == 4:
            if self.symmetric:
                qmin, qmax = -7, 7  # Reserve -8 for symmetric
            else:
                qmin, qmax = -8, 7
            dtype = torch.int8  # Store in int8, but only use 4 bits
        else:
            raise ValueError("Only 4 and 8 bits supported")

        if self.per_channel:
            # Per-channel quantization (per output channel)
            axis = 0  # Quantize along output dimension
            weight_reshaped = weight.view(weight.shape[0], -1)

            if self.symmetric:
                # Symmetric quantization: zero_point = 0
                max_vals = weight_reshaped.abs().max(dim=1, keepdim=True)[0]
                scales = max_vals / qmax
                scales = torch.clamp(scales, min=1e-8)
                zero_points = torch.zeros_like(scales, dtype=torch.int32)
            else:
                # Asymmetric quantization: calculate optimal zero_point
                min_vals = weight_reshaped.min(dim=1, keepdim=True)[0]
                max_vals = weight_reshaped.max(dim=1, keepdim=True)[0]

                scales = (max_vals - min_vals) / (qmax - qmin)
                scales = torch.clamp(scales, min=1e-8)

                zero_points = qmin - torch.round(min_vals / scales)
                zero_points = torch.clamp(zero_points, qmin, qmax)

            # Broadcast scales and zero_points back to weight shape
            scales = scales.view(-1, 1).expand_as(weight)
            zero_points = zero_points.view(-1, 1).expand_as(weight)
        else:
            # Per-tensor quantization
            if self.symmetric:
                max_val = weight.abs().max()
                scales = max_val / qmax
                scales = torch.clamp(scales, min=1e-8)
                zero_points = torch.zeros_like(scales, dtype=torch.int32)
            else:
                min_val = weight.min()
                max_val = weight.max()

                scales = (max_val - min_val) / (qmax - qmin)
                scales = torch.clamp(scales, min=1e-8)

                zero_points = qmin - torch.round(min_val / scales)
                zero_points = torch.clamp(zero_points, qmin, qmax)

        # Quantize: q = round(x/scale + zero_point)
        quantized = torch.round(weight / scales + zero_points)
        quantized = torch.clamp(quantized, qmin, qmax)

        # Store quantized weights and parameters
        self.register_buffer("quantized_weight", quantized.to(dtype))

        if self.per_channel:
            # Store per-channel scales and zero_points
            self.register_buffer("scales", scales[:, 0])  # Take first column since all are same
            self.register_buffer("zero_points", zero_points[:, 0].to(torch.int32))
        else:
            # Store per-tensor scales and zero_points
            self.register_buffer("scales", scales)
            self.register_buffer("zero_points", zero_points.to(torch.int32))

    def dequantize_weights(self):
        """Dequantize weights: x = scale * (q - zero_point)"""
        if self.per_channel:
            # Expand scales and zero_points for broadcasting
            scales = self.scales.view(-1, 1).expand_as(self.quantized_weight)
            zero_points = self.zero_points.view(-1, 1).expand_as(self.quantized_weight)
        else:
            scales = self.scales
            zero_points = self.zero_points

        # Dequantize: x = scale * (q - zero_point)
        dequantized = scales * (self.quantized_weight.float() - zero_points.float())
        return dequantized

    def forward(self, x):
        # Dequantize weights during forward pass
        dequantized_weight = self.dequantize_weights()
        return torch.nn.functional.linear(x, dequantized_weight, self.bias)

    def get_quantization_info(self):
        """Return quantization parameters for inspection"""
        return {
            "scales": self.scales,
            "zero_points": self.zero_points,
            "bits": self.bits,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "quantized_weight_shape": self.quantized_weight.shape,
            "quantized_weight_dtype": self.quantized_weight.dtype,
        }


class Linear(nn.Module, AblationLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        ablation_variant: AblationVariant,
        ablation_config: AblationConfig,
        **kwargs,
    ) -> None:

        super().__init__()
        AblationLayer.__init__(self, base_layer, ablation_config, ablation_variant, **kwargs)

        self._active_adapter = adapter_name
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.update_layer(base_layer, adapter_name, ablation_config, ablation_variant)

        # Metric tracking setup
        self.metric_tracking = (
            ablation_config.metric_tracking if hasattr(ablation_config, "metric_tracking") else False
        )
        self.track_n = ablation_config.track_n if hasattr(ablation_config, "track_n") else 100

        if self.metric_tracking:
            self.step_counter = 0

            # Calibration-style storage
            self.in_activation_stats = []
            self.out_activation_stats = []
            self.weight_magnitudes = []
            self.gradient_norms = []

            # Long-term storage
            self.stored_metrics = {}

    def _track_layer_metrics_calibration(
        self, layer_name, input_tensor, output_tensor, param_tensor, active_adapter
    ):
        """Calibration-style tracking for a single layer"""
        if not self.metric_tracking:
            return

        with torch.no_grad():
            # Per-channel metrics
            if input_tensor.dim() == 3:  # (batch, seq, features)
                in_magnitude = torch.mean(torch.abs(input_tensor), dim=(0, 1))
                out_magnitude = torch.mean(torch.abs(output_tensor), dim=(0, 1))
            else:  # (batch, features)
                in_magnitude = torch.mean(torch.abs(input_tensor), dim=0)
                out_magnitude = torch.mean(torch.abs(output_tensor), dim=0)

            # Store activation stats
            self.in_activation_stats.append(
                {"layer_name": layer_name, "magnitude": in_magnitude.cpu().detach().numpy()}
            )
            self.out_activation_stats.append(
                {"layer_name": layer_name, "magnitude": out_magnitude.cpu().detach().numpy()}
            )

            # Compute weight magnitudes
            if "down_project" in layer_name:
                weight_norm = torch.norm(param_tensor, p=1, dim=1)  # Shape: (input_channels,)
            elif "up_project" in layer_name:
                weight_norm = torch.norm(param_tensor, p=1, dim=0)  # Shape: (output_channels,)
            elif "intermediate" in layer_name:
                weight_norm = torch.norm(param_tensor, p=1, dim=1)
            elif "input_vector" in layer_name:
                weight_norm = torch.abs(param_tensor)
            else:
                weight_norm = torch.norm(param_tensor, p=1, dim=-1)

            # Store weight magnitudes
            self.weight_magnitudes.append(
                {"layer_name": layer_name, "weight_norm": weight_norm.cpu().detach().numpy()}
            )

            # Store gradient norm if gradients exist
            if param_tensor.grad is not None:
                grad_norm = param_tensor.grad.detach().norm(p=2).item()
            else:
                grad_norm = 0.0

            self.gradient_norms.append({"layer_name": layer_name, "grad_norm": grad_norm})

    def _store_channel_metrics(self, active_adapter):
        """Store aggregated channel metrics - called every track_n steps"""
        if not self.metric_tracking:
            return

        # Get recent entries for this window
        window_start = max(0, len(self.in_activation_stats) - self.track_n)

        # Aggregate metrics for this window
        window_metrics = {"step_range": (self.step_counter - self.track_n, self.step_counter), "layers": {}}

        # Group by layer name
        layer_groups = {}
        for i in range(window_start, len(self.in_activation_stats)):
            layer_name = self.in_activation_stats[i]["layer_name"]
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {
                    "in_magnitudes": [],
                    "out_magnitudes": [],
                    "weight_norms": [],
                    "grad_norms": [],
                }

            layer_groups[layer_name]["in_magnitudes"].append(self.in_activation_stats[i]["magnitude"])
            layer_groups[layer_name]["out_magnitudes"].append(self.out_activation_stats[i]["magnitude"])
            layer_groups[layer_name]["weight_norms"].append(self.weight_magnitudes[i]["weight_norm"])
            layer_groups[layer_name]["grad_norms"].append(self.gradient_norms[i]["grad_norm"])

        # Compute aggregated statistics for each layer
        for layer_name, layer_data in layer_groups.items():
            if layer_data["in_magnitudes"]:
                import numpy as np

                in_mags = np.array(layer_data["in_magnitudes"])
                out_mags = np.array(layer_data["out_magnitudes"])
                weight_norms = np.array(layer_data["weight_norms"])

                window_metrics["layers"][layer_name] = {
                    "in_magnitude_mean": np.mean(in_mags, axis=0),
                    "in_magnitude_std": np.std(in_mags, axis=0),
                    "out_magnitude_mean": np.mean(out_mags, axis=0),
                    "out_magnitude_std": np.std(out_mags, axis=0),
                    "weight_norm_mean": np.mean(weight_norms, axis=0),
                    "weight_norm_std": np.std(weight_norms, axis=0),
                }

                if layer_data["grad_norms"]:
                    grad_norms = np.array(layer_data["grad_norms"])
                    window_metrics["layers"][layer_name]["grad_norm_mean"] = np.mean(grad_norms)
                    window_metrics["layers"][layer_name]["grad_norm_std"] = np.std(grad_norms)

        # Store in long-term storage
        if "windows" not in self.stored_metrics:
            self.stored_metrics["windows"] = []
        self.stored_metrics["windows"].append(window_metrics)

        # Clear buffers to prevent memory growth
        self.in_activation_stats.clear()
        self.out_activation_stats.clear()
        self.weight_magnitudes.clear()
        self.gradient_norms.clear()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            # Update counter
            if self.metric_tracking:
                self.step_counter += 1

            for active_adapter in self.active_adapters:
                if active_adapter not in self.up_project.keys():
                    continue

                if self.ablation_variant == AblationVariant.VARIANT_0:
                    # Normal LoRA: x -> down_project -> up_project
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = adapter_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            adapter_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_A:
                    # LoRA + intermediate: x -> down_project -> intermediate -> up_project
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    intermediate_output = adapter_output @ self.intermediate[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "intermediate",
                            adapter_output,
                            intermediate_output,
                            self.intermediate[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = intermediate_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            intermediate_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_B:
                    # Input vector + frozen down + up: (x * input_vector) -> down_project -> up_project
                    x_modified = x * self.input_vector[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "input_vector", x, x_modified, self.input_vector[active_adapter], active_adapter
                        )

                    adapter_output = x_modified @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x_modified,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = adapter_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            adapter_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_C:
                    # Frozen down + intermediate + up: x -> down_project -> intermediate -> up_project
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    intermediate_output = adapter_output @ self.intermediate[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "intermediate",
                            adapter_output,
                            intermediate_output,
                            self.intermediate[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = intermediate_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            intermediate_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_D:
                    # Shared frozen down + intermediate + up: x -> down_project -> intermediate -> up_project
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    intermediate_output = adapter_output @ self.intermediate[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "intermediate",
                            adapter_output,
                            intermediate_output,
                            self.intermediate[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = intermediate_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            intermediate_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_E:
                    # Mid training random rank pruning (assume already pruned)
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = adapter_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            adapter_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_F:
                    # Mid training least L1 dimensions rank pruning (assume already pruned)
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = adapter_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            adapter_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_G:
                    # Int8 quantized backbone - only adapter computation (base already computed above)
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = adapter_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            adapter_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

                elif self.ablation_variant == AblationVariant.VARIANT_H:
                    # Int4 quantized backbone - only adapter computation (base already computed above)
                    adapter_output = x @ self.down_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "down_project",
                            x,
                            adapter_output,
                            self.down_project[active_adapter],
                            active_adapter,
                        )

                    adapter_output_final = adapter_output @ self.up_project[active_adapter]

                    if self.metric_tracking:
                        self._track_layer_metrics_calibration(
                            "up_project",
                            adapter_output,
                            adapter_output_final,
                            self.up_project[active_adapter],
                            active_adapter,
                        )

                    result += adapter_output_final * self.alpha

        # Store metrics every track_n steps
        if self.metric_tracking and self.step_counter % self.track_n == 0:
            self._store_channel_metrics(active_adapter)

        result = result.to(previous_dtype)
        return result

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # This function is introduced in newer PEFT versions. we modify this function instead of modifying
        # the merge function (as we did previously for version 0.4.0 of PEFT).
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # TODO: Implement merging (zip merging to a certain latent space)
        pass

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "abl." + rep

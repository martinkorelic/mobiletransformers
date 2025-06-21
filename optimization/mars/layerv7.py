from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

from research.experiments import create_orthogonal_matrices

class QuantizedBaseLayer(nn.Module):
    """
    Manual dynamic quantization for frozen backbone layers.
    Stores weights as uint8 and dequantizes during forward pass.
    Activations remain in full precision.
    """
    
    def __init__(self, base_layer):
        super().__init__()
        
        if not isinstance(base_layer, nn.Linear):
            raise ValueError("Currently only supports nn.Linear layers")
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Quantize and store weights as uint8
        weight_quantized, weight_scale, weight_zero_point = self._quantize_tensor(
            base_layer.weight.data
        )
        
        # Register as buffers (no gradients needed)
        self.register_buffer('weight_quantized', weight_quantized)
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('weight_zero_point', weight_zero_point)
        
        # Handle bias (keep in full precision since it's small)
        if base_layer.bias is not None:
            self.register_buffer('bias', base_layer.bias.data.clone())
        else:
            self.register_buffer('bias', None)
    
    def _quantize_tensor(self, tensor):
        """
        Quantize tensor to uint8 with per-tensor quantization
        Returns: quantized_tensor (uint8), scale (float), zero_point (uint8)
        """
        # Find min and max values
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        # Calculate scale and zero_point for uint8 (0-255)
        qmin, qmax = 0, 255
        
        # Handle edge case where min == max
        if tensor_min == tensor_max:
            scale = 1.0
            zero_point = 0
        else:
            scale = (tensor_max - tensor_min) / (qmax - qmin)
            zero_point = qmin - tensor_min / scale
            zero_point = max(qmin, min(qmax, int(round(zero_point))))
        
        # Quantize: (float_val / scale) + zero_point
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.uint8)
        
        return quantized, torch.tensor(scale), torch.tensor(zero_point, dtype=torch.uint8)
    
    def _dequantize_weights(self):
        """
        Dequantize weights back to float32 for computation
        """
        # Dequantize: (quantized_val - zero_point) * scale
        return (self.weight_quantized.float() - self.weight_zero_point.float()) * self.weight_scale
    
    def forward(self, x):
        """
        Forward pass: dequantize weights on-the-fly and compute linear transformation
        """
        # Dequantize weights
        weight_fp32 = self._dequantize_weights()
        
        # Standard linear operation with dequantized weights
        return torch.nn.functional.linear(x, weight_fp32, self.bias)

class MarsLayer(BaseTunerLayer):
    adapter_layer_names = ("up_project",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.up_project = nn.ModuleDict({})
        self.down_project = nn.ModuleDict({})

        # Check if quantization is requested
        self.quantize_base = kwargs.get("quantize_base", False)
        self.preserve_errors = kwargs.get("preserve_errors", False)
        self.trainable_down = kwargs.get("trainable_down", True)
        self.onnx_export = kwargs.get("onnx_export", False)
        
        # Apply dynamic quantization to base layer if requested
        if self.quantize_base and isinstance(base_layer, nn.Linear):
            self.base_layer = QuantizedBaseLayer(
                base_layer
            )
        else:
            self.base_layer = base_layer
        
        self.manual_seed = True

    def update_layer(self, original_weights, adapter_name, rank, alpha, projection_type, **kwargs):
        self.rank = rank
        self.in_features = original_weights.in_features
        self.out_features = original_weights.out_features
        self.alpha = alpha / self.rank
        self.projection_type = projection_type

        # Check if error preservation is requested
        self.shared_rank = kwargs.get("shared_rank", rank)
        self.is_standalone = kwargs.get("is_standalone", True)

        # If base layer quantized and preserve errors (either standalone or shared)
        if self.quantize_base and self.preserve_errors:
            self._compute_error_residuals_and_svd(original_weights.weight.data.clone(), adapter_name, rank, self.shared_rank)

        # If base layer is not quantized and standalone adapter
        elif self.is_standalone and not self.preserve_errors:
                
                self.adapter_layer_names = ("up_project","down_project")

                # Down-projection
                self.down_project[adapter_name] = nn.Linear(self.in_features, rank, bias=False)
                torch.nn.init.normal_(self.down_project[adapter_name].weight, mean=0, std=1 / rank)

                # If frozen down add scale to matrix
                if not self.trainable_down:
                    self.down_project[adapter_name].weight.data *= alpha / rank
                self.down_project[adapter_name].requires_grad_(self.trainable_down)

                # Up-projection
                self.up_project[adapter_name] = nn.Linear(self.rank, self.out_features, bias=False)
                self.up_project[adapter_name].weight.data = torch.nn.init.zeros_(self.up_project[adapter_name].weight.data)
                self.up_project[adapter_name].weight.requires_grad = True
        
        # If base layer is not quantized and not standalone
        else:
            # Up-projection
            self.up_project[adapter_name] = nn.Linear(self.rank, self.out_features, bias=False)
            self.up_project[adapter_name].weight.data = torch.nn.init.zeros_(self.up_project[adapter_name].weight.data)
            self.up_project[adapter_name].weight.requires_grad = True

        # If we have to get it ready for ONNX export by replacing the quantized weights with original weights
        # ONNX dynamic quantization will take care of quantizing the base weights
        if self.onnx_export:
            self.base_layer = original_weights
    
        self.adapter_name = adapter_name
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)
    
    def _compute_error_residuals_and_svd(self, original_weight, adapter_name, rank, shared_rank):
        """
        Compute error residuals between quantized and original weights,
        then perform SVD to initialize adapter weights with error correction.
        """
        
        # Get dequantized weights from quantized layer
        quantized_weight = self._extract_dequantized_weights()
        
        # Compute error residuals: E = Q(W) - W
        error_residuals = quantized_weight - original_weight
        
        # Perform SVD on error residuals
        U, S, Vt = torch.linalg.svd(error_residuals.T, full_matrices=False)

        max_rank = min(rank, S.shape[0], Vt.shape[0])

        if self.is_standalone:

            U_truncated = U[:, :max_rank]
            S = torch.diag(S)
            S_truncated = S[:max_rank, :max_rank]
            Vt_truncated = Vt[:max_rank, :]

            # Down-projection
            self.down_project[adapter_name] = nn.Linear(self.in_features, rank, bias=False)
            with torch.no_grad():
                self.down_project[adapter_name].weight.copy_((U_truncated @ S_truncated).T)

            # If frozen down add scale to matrix
            if not self.trainable_down:
                self.down_project[adapter_name].weight.data *= self.alpha / self.rank
            self.down_project[adapter_name].requires_grad_(self.trainable_down)

            # Replace up projection with Vt
            self.up_project[adapter_name] = nn.Linear(rank, self.out_features, bias=False)
            with torch.no_grad():
                self.up_project[adapter_name].weight.copy_(Vt_truncated.T.clone())
            self.up_project[adapter_name].weight.requires_grad = True

            return
        
        # Truncate to specified shared rank
        max_shared_rank = min(shared_rank, U.shape[1], S.shape[0])
        
        U_truncated = U[:, :max_shared_rank]
        S = torch.diag(S)
        S_truncated = S[:max_shared_rank, :max_rank]
        Vt_truncated = Vt[:max_rank, :]
        
        # Store U and Sigma separately for potential future use
        self._store_svd_components(U_truncated, S_truncated)
        
        # Replace up projection with Vt
        self.up_project[adapter_name] = nn.Linear(rank, self.out_features, bias=False)
        with torch.no_grad():
            self.up_project[adapter_name].weight.copy_(Vt_truncated.T.clone())
        self.up_project[adapter_name].weight.requires_grad = True

    def _extract_dequantized_weights(self):
        """Extract dequantized weights from quantized base layer"""
        
        # Check if it's our custom QuantizedBaseLayer
        if isinstance(self.base_layer, QuantizedBaseLayer):
            # Use the built-in dequantization method
            return self.base_layer._dequantize_weights()
        
        # For PyTorch's dynamically quantized layers
        elif hasattr(self.base_layer, '_packed_params'):
            # For quantized linear layers
            packed_params = self.base_layer._packed_params
            if hasattr(packed_params, 'unpack'):
                weight, bias = packed_params.unpack()
                return weight.dequantize()
        
        # Direct dequantization if available
        elif hasattr(self.base_layer, 'weight') and hasattr(self.base_layer.weight, 'dequantize'):
            return self.base_layer.weight.dequantize()
        
        # Fallback for regular non-quantized layers
        elif hasattr(self.base_layer, 'weight'):
            return self.base_layer.weight.data
        
        else:
            raise ValueError("Cannot extract dequantized weights from base layer")
    
    def _store_svd_components(self, U, S):
        """Store U and Sigma components separately"""
        if not hasattr(self, 'svd_U'):
            self.svd_U = None
        if not hasattr(self, 'svd_S'):
            self.svd_S = None
        
        self.svd_U = U.clone()
        self.svd_S = S.clone()

    def clear_svd_components(self):
        """Clear stored U and Sigma matrices to free memory"""
        
        # Clear all SVD components
        if hasattr(self, 'svd_U'):
            del self.svd_U
            self.svd_U = None
        if hasattr(self, 'svd_S'):
            del self.svd_S
            self.svd_S = None
        
class SharedMLPAdapter(nn.Module):

    def __init__(self, hidden_size, rank, shared_rank, alpha, **kwargs):
        super().__init__()

        self.rank = rank
        self.shared_rank = shared_rank
        self.alpha = alpha
        self.orth_init = kwargs.get('orth_init', False)
        self.no_mixture = kwargs.get('no_mixture', False)
        self.trainable_down = kwargs.get("trainable_down", True)

        # Shared frozen down-projection
        self.mars_down_mlp = nn.Linear(hidden_size, shared_rank, bias=False)
        torch.nn.init.normal_(self.mars_down_mlp.weight, mean=0, std=1 / rank)

        if not self.trainable_down:
            self.mars_down_mlp.weight.data *= alpha / rank
        self.mars_down_mlp.requires_grad_(self.trainable_down)
        
        # Shared trainable transform as nn.Linear (with 'mars' key)
        if not self.no_mixture:
            self.mars = nn.Linear(shared_rank, 2 * rank, bias=False)

        self._update_layer()

    def _update_layer(self, u_proj=None, sigma=None, res_projection_type=None):

        # Replace down projection matrix or initialize
        if u_proj is not None:
            with torch.no_grad():
                self.mars_down_mlp.weight.copy_(u_proj.T)
        else:
            torch.nn.init.normal_(self.mars_down_mlp.weight, mean=0, std=1 / self.rank)

        if self.no_mixture:
            return

        # Update intermediate matrices
        if self.orth_init:
            orthogonal_matrices = create_orthogonal_matrices(self.rank, num_matrices=2, mean=0, std= 1 / self.rank, initial_matrix=sigma, device='cuda')

            with torch.no_grad():
                self.mars.weight.copy_(orthogonal_matrices.T)
        elif sigma is not None and res_projection_type is not None:

            with torch.no_grad():
                if res_projection_type == 'gate':
                    self.mars.weight[:self.rank, :].copy_(sigma.T)
                elif res_projection_type == 'up':
                    self.mars.weight[self.rank:, :].copy_(sigma.T)
        else:
            torch.nn.init.normal_(self.mars.weight, mean=0, std=1 / self.rank)

    def forward(self, x):
        # Shared down projection
        shared_out = self.mars_down_mlp(x)  # [batch, seq_len, 2r]

        # Trainable transform (use nn.Linear module)
        transformed = self.mars(shared_out)  # [batch, seq_len, 2r]

        # Split into gate and up projections
        gate_out, up_out = transformed.chunk(2, dim=-1)

        return gate_out, up_out

class SharedAttentionAdapter(nn.Module):

    def __init__(self, hidden_size, rank, shared_rank, alpha, enabled: list[str] = ['q', 'k', 'v'], **kwargs):
        super().__init__()

        self.rank = rank
        self.shared_rank = shared_rank
        self.alpha = alpha
        self.enabled = enabled
        self.enabled_idx = [i for i, name in enumerate(['q', 'k', 'v']) if name in enabled]
        self.num_enabled = len(self.enabled)

        self.orth_init = kwargs.get('orth_init', False)
        self.no_mixture = kwargs.get('no_mixture', False)
        self.trainable_down = kwargs.get("trainable_down", True)

        # Shared frozen down-projection
        self.mars_down_qkv = nn.Linear(hidden_size, shared_rank, bias=False)
        
        if not self.trainable_down:
            self.mars_down_qkv.weight.data *= alpha / rank
        self.mars_down_qkv.requires_grad_(self.trainable_down)

        # Trainable transform
        if not self.no_mixture:
            self.mars = nn.Linear(shared_rank, self.num_enabled * rank, bias=False)

        self._update_layer()

    def _update_layer(self, u_proj=None, sigma=None, res_projection_type=None):

        # Replace down projection matrix or initialize
        if u_proj is not None:
            with torch.no_grad():
                self.mars_down_qkv.weight.copy_(u_proj.T)
        else:
            torch.nn.init.normal_(self.mars_down_qkv.weight, mean=0, std=1 / self.rank)

        if self.no_mixture:
            return

        if self.orth_init:

            orthogonal_matrices = create_orthogonal_matrices(self.rank, num_matrices=self.num_enabled, mean=0, std= 1 / self.rank, initial_matrix=sigma, device='cuda')

            with torch.no_grad():
                self.mars.weight.copy_(orthogonal_matrices.T)
        elif sigma is not None and res_projection_type is not None:
            try:
                idx = self.enabled.index(res_projection_type)
            except ValueError:
                raise ValueError(f"{res_projection_type} is not enabled in {self.enabled}")
            
            start = idx * self.rank
            end = (idx + 1) * self.rank

            with torch.no_grad():
                self.mars.weight[start:end, :].copy_(sigma.T)
        else:
            torch.nn.init.normal_(self.mars.weight, mean=0, std=1 / self.rank)
        
    def forward(self, x):
        # Shared projection + transform
        shared_out = self.mars_down_qkv(x)
        transformed = self.mars(shared_out)

        # Split into parts
        parts = transformed.chunk(self.num_enabled, dim=-1)

        return parts

class Linear(nn.Module, MarsLayer):
    def __init__(self, base_layer, adapter_name, r, alpha, projection_type, **kwargs):
        super().__init__()
        MarsLayer.__init__(self, base_layer, **kwargs)

        self.ada_name = kwargs.get("target_name", None)
        self.fan_in_fan_out = kwargs.get("fan_in_fan_out", False)
        self._active_adapter = adapter_name
        
        self.update_layer(base_layer, adapter_name, r, alpha, projection_type, **kwargs)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:

        if not self.is_standalone and len(args) > 0:
        # First arg is shared_output, second is original input
            shared_out = args[0]
            x = args[1] if len(args) > 1 else None
            remaining_args = args[2:] if len(args) > 2 else ()
        else:
            # Standalone adapter - normal args
            shared_out = None
            x = args[0] if len(args) > 0 else None
            remaining_args = args[1:] if len(args) > 1 else ()

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            # Pass original args to base layer (without shared_out)
            base_args = (x,) + remaining_args if x is not None else remaining_args
            return self.base_layer(*base_args, **kwargs)

        # Get base model output
        base_args = (x,) + remaining_args if x is not None else remaining_args
        result = self.base_layer(*base_args)
        
        # Apply up-projection if needed
        for active_adapter in self.active_adapters:
            if active_adapter not in self.up_project:
                continue

            if self.is_standalone:
                adapter_result = self.up_project[active_adapter](self.down_project[active_adapter](x))
            elif shared_out is not None:
                adapter_result = self.up_project[active_adapter](shared_out)
            
            if self.trainable_down:
                result += adapter_result * self.alpha
            else:
                result += adapter_result
                
        return result
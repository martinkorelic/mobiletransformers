from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn
import numpy as np

class MarsLayer(BaseTunerLayer):

    adapter_layer_names = ("up_project", "base_scaling", "adapter_scaling")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.up_project = nn.ParameterDict({})
        #self.down_project = nn.ParameterDict({})
        self.adapter_scaling = nn.ParameterDict({})
        self.base_scaling = nn.ParameterDict({})
        self.latent_space = nn.ParameterDict({})

        # Analysis
        self.count = 0
        self.window_size = kwargs.get("window_size", 5)
        self.current_window = 0

        self.calibration_phase = True
        self.svd_decompose = False
        self.sparse_channels = False
        self.scale_adapters = False
        self.manual_seed = True

        self.channel_stats = {
            "in_magnitudes": [],  
            "out_magnitudes": [],
            "l1_weight_down_project": [],
            "l1_weight_up_project": [],
            "base_scaling": [],
            "adapter_scaling": [],
            "gradient_norm": []
        }

        # Rolling buffers for the current window
        self.in_activation_stats = []
        self.out_activation_stats = []
        self.base_scaling_factors = []
        self.adapter_scaling_factors = []
        self.weight_magnitudes = []

    def update_layer(self, original_weights, adapter_name, ranks, lora_alphas, cumulative_ranks):
        
        self.ranks = ranks

        self.inner_rank = 4

        self.in_features = original_weights.in_features
        self.out_features = original_weights.out_features

        self.down_project = nn.ParameterDict({})

        # Decompose into U, S, Vt
        if self.svd_decompose:
            U, S, Vt = torch.linalg.svd(original_weights.weight.T, full_matrices=False)
            
            # Truncate to the desired rank
            U_truncated = U[:, :self.inner_rank]
            S_truncated = S[:self.inner_rank]
            Vt_truncated = Vt[:self.inner_rank, :]

            # Compute down_project = U * Sigma (truncated)
            down_project = U_truncated @ torch.diag(S_truncated)

            up_project = Vt_truncated

            self.down_project[adapter_name] = nn.Parameter(down_project)
            self.up_project[adapter_name] = nn.Parameter(up_project)
        elif self.sparse_channels:

            gen = None
            if self.manual_seed:
                gen = torch.Generator()
                gen.manual_seed(42)

            self.input_channels = int(self.in_features)
            self.output_channels = int(self.out_features * 0.75)

            self.down_project = nn.Parameter(torch.nn.init.normal_(torch.empty((self.input_channels, self.inner_rank)), mean=0, std=0.1, generator=gen), requires_grad=False)
            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.inner_rank, self.output_channels))))

            input_ordering = torch.randint(size=(self.input_channels,), high=self.in_features)
            self.register_buffer("input_ordering", input_ordering)
            output_ordering = torch.randint(size=(self.output_channels,), high=self.out_features)
            self.register_buffer("output_ordering", output_ordering)

            self.in_features = self.input_channels
            self.out_features = self.output_channels
        else:

            gen = None
            if self.manual_seed:
                gen = torch.Generator()
                gen.manual_seed(42)

            self.down_project = nn.Parameter(torch.nn.init.normal_(torch.empty((original_weights.in_features, self.inner_rank)), mean=0, std=0.1, generator=gen), requires_grad=False)
            #self.latent_space[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.inner_rank, self.inner_rank)), mean=0, std=0.01))
            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.inner_rank, original_weights.out_features))))

        self.alpha = 100.0

        logit_one = -torch.log(1/torch.tensor(0.5) - 1)
        self.adapter_scaling[adapter_name] = nn.Parameter(logit_one * torch.ones(self.out_features))

        logit_one = -torch.log(1/torch.tensor(1) - 1)
        self.base_scaling[adapter_name] = nn.Parameter(logit_one * torch.ones(self.out_features))

        self.scaler = nn.Sigmoid()
        #self.scaler = nn.Softmax(dim=0)

        self.adapter_name = adapter_name

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

class Linear(nn.Module, MarsLayer):

    adapter_layer_names = ("up_project", "base_scaling", "adapter_scaling")

    def __init__(self,
                 base_layer,
                 adapter_name,
                 ranks,
                 lora_alphas,
                 cumulative_ranks,
                 fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 **kwargs) -> None:
        
        super().__init__()
        MarsLayer.__init__(self, base_layer, **kwargs)

        
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            base_layer, adapter_name, ranks, lora_alphas, cumulative_ranks
        )

    def _store_channel_metrics(self, active_adapter):
        """
        Store the per-channel metrics for the current window.
        """
        # Average collected statistics over the window
        avg_in_magnitudes = np.mean([stat['magnitude'] for stat in self.in_activation_stats], axis=0)
        avg_out_magnitudes = np.mean([stat['magnitude'] for stat in self.out_activation_stats], axis=0)
        avg_weight_magnitudes = {
            "down_norm": np.mean([wm["down_norm"] for wm in self.weight_magnitudes], axis=0),
            "up_norm": np.mean([wm["up_norm"] for wm in self.weight_magnitudes], axis=0)
        }
        #avg_base_scaling = np.mean(self.base_scaling_factors, axis=0)
        

        # Store the averaged statistics
        self.channel_stats["in_magnitudes"].append(avg_in_magnitudes)
        self.channel_stats["out_magnitudes"].append(avg_out_magnitudes)
        self.channel_stats["l1_weight_down_project"].append(avg_weight_magnitudes["down_norm"])
        self.channel_stats["l1_weight_up_project"].append(avg_weight_magnitudes["up_norm"])
        #self.channel_stats["base_scaling"].append(avg_base_scaling)

        if self.scale_adapters:
            avg_adapter_scaling = np.mean(self.adapter_scaling_factors, axis=0)
            self.channel_stats["adapter_scaling"].append(avg_adapter_scaling)

        norm = self.up_project[active_adapter].grad.detach().norm(p=2).item()

        # Store up_projection gradient norms
        self.channel_stats["gradient_norm"].append(
            norm
        )

        # Reset rolling buffers
        self.in_activation_stats.clear()
        self.out_activation_stats.clear()
        self.weight_magnitudes.clear()
        self.adapter_scaling_factors.clear()
        self.base_scaling_factors.clear()
        self.current_window += 1

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
            batch_size, seq_len, hidden_size = x.shape    

            for active_adapter in self.active_adapters:
                if active_adapter not in self.up_project.keys():
                    continue

                #base_scaler = self.scaler(self.base_scaling[active_adapter])
                #result = base_scaler * result

                adapter_scaler = None

                if self.sparse_channels:
                    x_sliced = torch.index_select(x, dim=2, index=self.input_ordering)

                    out = x_sliced @ self.down_project @ self.up_project[active_adapter]

                    if self.scale_adapters:
                        adapter_scaler = self.scaler(self.alpha * self.adapter_scaling[active_adapter])
                        out = adapter_scaler * out
                    
                    result.scatter_add_(dim=2, index=self.output_ordering.expand(batch_size, seq_len, -1), src=out)
                else:
                    out = x @ self.down_project @ self.up_project[active_adapter]
                    
                    if self.scale_adapters:
                        adapter_scaler = self.scaler(self.alpha * self.adapter_scaling[active_adapter])
                        out = adapter_scaler * out

                    result += out

                # Compute per-channel magnitudes and variances
                #with torch.no_grad():
                if self.calibration_phase:

                    # Per-channel metrics
                    in_magnitude = torch.mean(torch.abs(x), dim=(0, 1))
                    out_magnitude = torch.mean(torch.abs(out), dim=(0, 1))

                    self.in_activation_stats.append({
                        'magnitude': in_magnitude.cpu().detach().numpy()
                    })
                    self.out_activation_stats.append({
                        'magnitude': out_magnitude.cpu().detach().numpy()
                    })

                    # Compute weight magnitudes for each matrix
                    down_norm = torch.norm(self.down_project, p=1, dim=1)  # Shape: (input_channels,)
                    up_norm = torch.norm(self.up_project[active_adapter], p=1, dim=0)      # Shape: (output_channels,)

                    # Store the individual norms if needed
                    self.weight_magnitudes.append({
                        "down_norm": down_norm.cpu().detach().numpy(),
                        "up_norm": up_norm.cpu().detach().numpy()
                    })

                    #self.base_scaling_factors.append(base_scaler.cpu().detach().numpy())
                    if self.scale_adapters:
                        self.adapter_scaling_factors.append(adapter_scaler.cpu().detach().numpy())

                    self.count += 1
                    # Store metrics if the window size is reached
                    if self.count % self.window_size == 0:
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
        # TODO: Implement merging
        pass

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mars." + rep
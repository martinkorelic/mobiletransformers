from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn
import numpy as np

class MarsLayer(BaseTunerLayer):

    adapter_layer_names = ("transform_down_latent", "transform_up_latent")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.up_project = nn.ParameterDict({})
        self.down_project = nn.ParameterDict({})

        #self.lora_down_stacked = nn.ParameterDict({})
        #self.lora_up_stacked = nn.ParameterDict({})
        #self.lora_latent_stacked = nn.ParameterDict({})
        
        #self.transform_matrices = nn.ParameterDict({})
        #self.transform_up = nn.ParameterDict({})
        self.transform_up_latent = nn.ParameterDict({})
        self.transform_down_latent = nn.ParameterDict({})
        self.group_importance = nn.ParameterDict({})
        self.in_group_pool = nn.ParameterDict({})
        
        #self.mixture_projects = nn.ModuleDict()

        # Analysis
        self.in_magnitudes = None
        self.in_variances = None
        self.out_magnitudes = None
        self.out_variances = None
        self.count = 0
        self.calibration_phase = False

    def update_layer(self, original_weights, adapter_name, ranks, lora_alphas, cumulative_ranks):
        #U, S, Vt = torch.linalg.svd(original_weights.weight.T, full_matrices=False)
        
        self.ranks = ranks

        self.inner_rank = 4

        self.in_features = original_weights.in_features
        self.out_features = original_weights.out_features

        self.input_channels = int(self.in_features * 0.75)
        self.output_channels = int(self.out_features * 0.75)

        self.cumulative_ranks = cumulative_ranks
        # TODO: Move some parameters to register buffers (could be inferred from cumulative_ranks?)
        self.sum_rank = sum(ranks)
        
        # What if we only use one matrix without inner ranks? (less input and output channels)
        self.transform_down_latent[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.input_channels, self.inner_rank)), mean=0, std=0.1))
        #self.transform_latent = nn.Parameter(torch.nn.init.normal_(torch.empty((self.inner_rank, self.inner_rank)), mean=0, std=0.00001))
        self.transform_up_latent[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.inner_rank, self.output_channels))))
        
        #nn.Parameter(torch.nn.init.normal_(torch.empty((self.inner_rank, self.output_channels)), mean=0, std=0.00001))

        self.num_subspaces = len(ranks)
        self.adapter_name = adapter_name

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

        input_ordering = torch.randint(size=(self.input_channels,), high=self.in_features)
        self.register_buffer("input_ordering", input_ordering)
        output_ordering = torch.randint(size=(self.output_channels,), high=self.out_features)
        self.register_buffer("output_ordering", output_ordering)

    def identify_trainable_channels(self):
        """
        Identifies most important activation channels and creates permutation so it they are ordered at the front.
        """
        
        # Identify top-k channels for magnitudes and variances
        in_top_magnitude_indices = np.argsort(-self.in_magnitudes)[:self.input_channels]

        # TODO: Only top variances
        in_top_variance_indices = np.argsort(-self.in_variances)[:self.input_channels]
        self.input_ordering = torch.from_numpy(in_top_variance_indices)

        out_top_variance_indices = np.argsort(-self.in_variances)[:self.output_channels]
        self.output_ordering = torch.from_numpy(out_top_variance_indices)

class Linear(nn.Module, MarsLayer):

    adapter_layer_names = ("transform_down_latent", "transform_up_latent")

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
                if active_adapter not in self.transform_up_latent.keys():
                    continue

                # Compute per-channel magnitudes and variances
                if self.calibration_phase:
                    in_feature_magnitudes = torch.mean(torch.abs(x), dim=(0,1))
                    in_variances = torch.var(x, dim=1).mean(dim=0)

                    out_feature_magnitudes = torch.mean(torch.abs(result), dim=(0,1))
                    out_variances = torch.var(result, dim=1).mean(dim=0)

                    if self.in_magnitudes is None:
                        # Direct initialization
                        self.in_magnitudes = in_feature_magnitudes.cpu().numpy()
                        self.in_variances = in_variances.cpu().numpy()
                        self.out_magnitudes = out_feature_magnitudes.cpu().numpy()
                        self.out_variances = out_variances.cpu().numpy()
                    else:
                        # Update rolling mean
                        self.in_magnitudes += (in_feature_magnitudes.cpu().numpy() - self.in_magnitudes) / self.count
                        self.in_variances += (in_variances.cpu().numpy() - self.in_variances) / self.count
                        self.out_magnitudes += (out_feature_magnitudes.cpu().numpy() - self.out_magnitudes) / self.count
                        self.out_variances += (out_variances.cpu().numpy() - self.out_variances) / self.count

                    # Increment count
                    self.count += 1
                else:
                    #x_sliced = x[:, :, self.input_ordering]

                    x_sliced = torch.index_select(x, dim=2, index=self.input_ordering)

                    out = x_sliced @ self.transform_down_latent[active_adapter] @ self.transform_up_latent[active_adapter]

                    #lora_out = torch.zeros(batch_size, seq_len, self.out_features, device=x.device)

                    #lora_out[:, :, self.output_ordering] = out
                    
                    #result += lora_out

                    result.scatter_add_(dim=2, index=self.output_ordering.expand(batch_size, seq_len, -1), src=out)

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
        return "mars." + rep
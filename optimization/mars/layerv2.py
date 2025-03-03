from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn

class MarsLayer(BaseTunerLayer):

    adapter_layer_names = ("latent_up_proj", "latent_up_proj", "mixture_projects")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.up_project = nn.ParameterDict({})
        self.down_project = nn.ParameterDict({})

        self.latent_up_proj = nn.ParameterDict({})
        self.latent_down_proj = nn.ParameterDict({})
        self.mixture_projects = nn.ModuleDict()

    def update_layer(self, original_weights, adapter_name, ranks, lora_alphas, cumulative_ranks):
        U, S, Vt = torch.linalg.svd(original_weights.weight.T, full_matrices=False)
        
        self.ranks = ranks

        # TODO: Hardcoded
        self.io_rank = 8

        self.cumulative_ranks = cumulative_ranks
        # TODO: Move some parameters to register buffers (could be inferred from cumulative_ranks?)
        self.sum_rank = sum(ranks)

        self.down_project[adapter_name] = nn.Parameter((U[:, :self.io_rank] @ torch.diag(S[:self.io_rank])).T, requires_grad=False)

        self.up_project[adapter_name] = nn.Parameter(Vt[:self.io_rank, :].T, requires_grad=False)

        self.latent_up_proj[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.io_rank * len(ranks), self.sum_rank)), mean=0, std=0.00001))
        self.latent_down_proj[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.sum_rank, self.io_rank * len(ranks))), mean=0, std=0.00001))

        self.mixture_projects[adapter_name] = nn.ParameterDict()

        for n, r in enumerate(ranks):

            mixture_key = f"mixture_{n}"

            if n == len(ranks) - 1:
                continue
            else:
                self.mixture_projects[adapter_name][mixture_key] = nn.Parameter(torch.nn.init.normal_(torch.empty((ranks[n+1],r)), mean=0, std=0.00001))
        
        self.num_subspaces = len(ranks)
        self.adapter_name = adapter_name

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

class Linear(nn.Module, MarsLayer):

    adapter_layer_names = ("latent_up_proj", "latent_up_proj", "mixture_projects")

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

        #self.adaptive_scaling = True
        #self.latent_scalar_weights = torch.nn.Parameter(torch.ones(self.num_subspaces))

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
            
            for active_adapter in self.active_adapters:
                if active_adapter not in self.latent_up_proj.keys():
                    continue
                
                down_projection = x @ self.down_project[active_adapter].T

                batch_size, seq_len, hidden_dim = down_projection.shape
                new_hidden_dim = hidden_dim * self.num_subspaces

                #down_proj_in = torch.tile(down_projection, (1, 1, self.num_subspaces))
                down_proj_in = (
                    down_projection.unsqueeze(-1)
                    .expand(batch_size, seq_len, hidden_dim, self.num_subspaces)
                    .reshape(batch_size, seq_len, new_hidden_dim)
                )

                proj_in = down_proj_in @ self.latent_up_proj[active_adapter]

                # Second project out because we canot modify gradient in place
                proj_out = torch.clone(proj_in)

                # Step 3: Iteratively update proj
                for ns in range(self.num_subspaces - 1):
                    start_idx = self.cumulative_ranks[ns]
                    end_idx = self.cumulative_ranks[ns + 1]

                    # Extract chunk from the current tensor
                    chunk = proj_in[..., start_idx:end_idx]

                    # Compute and add the result to the next segment
                    next_start_idx = end_idx
                    next_end_idx = self.cumulative_ranks[ns + 2]  # Use the next rank's end index
                    
                    # TODO: Mars Alphas (beta)
                    proj_out[..., next_start_idx:next_end_idx] += chunk @ self.mixture_projects[active_adapter][f"mixture_{ns}"].T

                out_projection = proj_out @ self.latent_down_proj[active_adapter]

                #up_proj_out = torch.tile(self.up_project[active_adapter], (self.num_subspaces, 1))

                # Expand self.up_project[active_adapter] along the first dimension to match the required size
                up_proj_out = self.up_project[active_adapter].T.expand(self.num_subspaces, -1, -1)

                up_proj_out = up_proj_out.reshape(-1, self.up_project[active_adapter].T.shape[-1])

                # TODO: LoRA Alpha
                result += out_projection @ up_proj_out
            

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
from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn

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

    def update_layer(self, original_weights, adapter_name, ranks, lora_alphas, cumulative_ranks):
        U, S, Vt = torch.linalg.svd(original_weights.weight.T, full_matrices=False)
        
        self.ranks = ranks

        # TODO: Hardcoded
        self.group_rank = 16

        self.num_groups = 32

        self.in_group_size = original_weights.in_features // self.num_groups
        self.out_group_size = original_weights.out_features // self.num_groups

        #self.base_layer_1 = nn.Parameter(U[:, :self.in_group_size] @ torch.diag(S[:self.in_group_size]), requires_grad=False)
        #self.base_layer_latent = nn.Parameter(torch.ones(self.in_group_size))
        #self.base_layer_2 = nn.Parameter(Vt[:self.out_group_size, :], requires_grad=False)

        print(f"In group size: {self.in_group_size}")
        print(f"Out group size: {self.out_group_size}")

        self.group_importance[adapter_name] = nn.Parameter(torch.ones(self.num_groups))

        self.cumulative_ranks = cumulative_ranks
        # TODO: Move some parameters to register buffers (could be inferred from cumulative_ranks?)
        self.sum_rank = sum(ranks)

       # LoRA parameters as ModuleDict
        #lora_down_dict = {
        #    f"group_{i}": nn.Parameter(torch.nn.init.normal_(torch.empty((self.group_size, self.group_rank)), mean=0, std=0.00001))
        #    for i in range(self.num_groups)
        #}
        #lora_up_dict = {
        #    f"group_{i}": nn.Parameter(torch.nn.init.normal_(torch.empty((self.group_rank, self.group_size)), mean=0, std=0.00001))
        #    for i in range(self.num_groups)
        #}

        #lora_latent_dict = {
        #    f"group_{i}": nn.Parameter(torch.nn.init.normal_(torch.empty((self.group_rank, self.group_rank)), mean=0, std=0.00001))
        #    for i in range(self.num_groups)
        #}

        # Pre-stack LoRA parameters into tensors for grouped computation
        #self.lora_down_stacked[adapter_name] = nn.Parameter(
        #    torch.stack([param for param in lora_down_dict.values()]), requires_grad=True
        #)  # Shape: (num_groups, group_size, rank)
        #self.lora_up_stacked[adapter_name] = nn.Parameter(
        #    torch.stack([param for param in lora_up_dict.values()]), requires_grad=True
        #)  # Shape: (num_groups, rank, group_size)

        #self.lora_latent_stacked[adapter_name] = nn.Parameter(
        #    torch.stack([param for param in lora_latent_dict.values()]), requires_grad=True
        #)

        
        #self.transform_matrices[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.num_groups, self.group_size)), mean=0, std=0.00001))
        self.transform_down_latent[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.in_group_size, self.group_rank)), mean=0, std=0.00001))
        self.transform_up_latent[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.group_rank, self.out_group_size)), mean=0, std=0.00001))

        self.in_group_pool[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.in_group_size, self.group_rank)), mean=0, std=0.00001))

        #self.transform_up[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.group_rank, original_weights.out_features)), mean=0, std=0.00001))

        self.num_subspaces = len(ranks)
        self.adapter_name = adapter_name

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

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

            batch_size, seq_len, hidden_dim = x.shape
            x_grouped = x.view(batch_size, seq_len, self.num_groups, self.in_group_size)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.transform_up_latent.keys():
                    continue

                x_pooled = x_grouped.mean(dim=2)


                x_grouped_up = x_grouped @ self.transform_down_latent[active_adapter]

                #print(x_grouped.shape)

                # TODO: Could it be transformed with element wise?
                x_pooled_transform = x_pooled @ self.in_group_pool[active_adapter]

                x_grouped_up = x_grouped_up + x_pooled_transform.unsqueeze(2)

                x_grouped_down = x_grouped_up @ self.transform_up_latent[active_adapter]

                # Group-wise importance
                x_grouped_down = x_grouped_down * self.group_importance[active_adapter].view(1, 1, -1, 1)

                lora_out = x_grouped_down.view(batch_size, seq_len, hidden_dim)
                
                # Concatenation sum and prod idea
                #x_grouped = x.view(batch_size, seq_len, self.num_groups, self.group_size)
                #x_reshaped = x_grouped.permute(2, 0, 1, 3)  # [num_groups, batch_size, seq_len, group_size]
                #transform_matrices = self.transform_matrices[active_adapter].unsqueeze(0).unsqueeze(0).unsqueeze(3)  # Shape: [1, 1, 16, 128]
                #transform_matrices = transform_matrices.expand(batch_size, seq_len, self.num_groups, 1, self.group_size)
                #concate = torch.cat([x_grouped.unsqueeze(3), transform_matrices], dim=3).prod(dim=3)
                #x_concate = concate.view(batch_size, seq_len, hidden_dim)
                
                # Initial idea of grouping
                # Reshape input for grouped computation
                #x_grouped = x.view(batch_size, seq_len, self.num_groups, self.group_size)
                #x_grouped = x_grouped.sum(dim=-2)

                #transformed = x_grouped @ self.transform_up_latent[active_adapter]
                #lora_out = transformed @ self.transform_up[active_adapter]

                #transformed = x_grouped * self.transform_matrices[active_adapter]
                #lora_out = transformed.sum(dim=2) @ self.transform_up[active_adapter]
    
                # Apply grouped LoRA (pre-stacked parameters)
                #lora_down = torch.einsum(
                #    "bsgd,gdr->bsgr", x_grouped, self.lora_down_stacked[active_adapter]
                #)  # Shape: (batch_size, seq_len, num_groups, rank)

                #lora_out = torch.einsum(
                #    "bsgr,grd->bsgd", lora_latent, self.lora_up_stacked[active_adapter]
                #).reshape(batch_size, seq_len, -1)  # Reshape to (batch_size, seq_len, input_dim)

                result += lora_out
            

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
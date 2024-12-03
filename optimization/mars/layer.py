
from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn


import onnx

from optimization.mars.study import reshape_to_higher_order, tensor_train_contract, tensor_train_decomposition, tt_tensor_elements

class MarsLayer(BaseTunerLayer):

    adapter_layer_names = ("latent_spaces", "mixture_projects")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        # SVD-decomposed components
        self.up_project = nn.ParameterDict({})
        self.down_project = nn.ParameterDict({})

        self.latent_spaces = nn.ModuleDict()
        self.mixture_projects = nn.ModuleDict()

    def update_layer(self, original_weights, adapter_name, ranks, lora_alphas):
        U, S, Vt = torch.linalg.svd(original_weights.weight.T, full_matrices=False)

        self.in_features = original_weights.in_features
        self.max_rank = max(ranks)
        self.sum_rank = sum(ranks)
        self.out_features = original_weights.out_features

        self.down_project[adapter_name] = nn.Parameter(U[:, :self.max_rank] @ torch.diag(S[:self.max_rank]), requires_grad=False)
        self.up_project[adapter_name] = nn.Parameter(Vt[:self.max_rank, :], requires_grad=False)

        self.latent_spaces[adapter_name] = nn.ParameterDict()
        self.mixture_projects[adapter_name] = nn.ParameterDict()

        for n, r in enumerate(ranks):

            subspace_key = f"subspace_{n}"
            mixture_key = f"mixture_{n}"
            self.latent_spaces[adapter_name][subspace_key] = nn.Parameter(torch.nn.init.normal_(torch.empty((r,r)), mean=0, std=0.00001))

            if n == len(ranks) - 1:
                continue
            else:
                self.mixture_projects[adapter_name][mixture_key] = nn.Parameter(torch.nn.init.normal_(torch.empty((ranks[n+1],r)), mean=0, std=0.00001))
        
        self.num_subspaces = len(ranks)
        self.adapter_name = adapter_name

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

class Linear(nn.Module, MarsLayer):

    adapter_layer_names = ("latent_spaces", "mixture_projects")

    def __init__(self,
                 base_layer,
                 adapter_name, ranks,
                 lora_alphas,
                 fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 **kwargs) -> None:
        
        super().__init__()

        MarsLayer.__init__(self, base_layer, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            base_layer, adapter_name, ranks, lora_alphas
        )

        self.adaptive_scaling = True
        self.latent_scalar_weights = torch.nn.Parameter(torch.ones(self.num_subspaces))

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
                if active_adapter not in self.latent_spaces.keys():
                    continue

                prev_state = torch.zeros(*x.shape[:-1], self.latent_spaces[active_adapter]["subspace_0"].shape[0], device=x.device)
                hidden_state_sum = torch.zeros_like(result, device=x.device)

                out_project = torch.empty(*x.shape[:-1], self.sum_rank, device=x.device)

                down_projection = x @ self.down_project[active_adapter]

                mid_rank = 0

                for ns in range(self.num_subspaces):
                    
                    rank = self.latent_spaces[active_adapter][f"subspace_{ns}"].shape[0]
                    
                    #truncated_U = self.U[active_adapter][:, :rank]
                    #truncated_Sigma_row = self.Sigma[active_adapter][:rank]

                    # Avoid using torch.diag for onnx.export
                    #truncated_Sigma_matrix = torch.eye(truncated_Sigma_row.shape[0], device=x.device) * truncated_Sigma_row
                    hidden_state = prev_state + down_projection[..., :rank]
                    hidden_state = hidden_state @ self.latent_spaces[active_adapter][f"subspace_{ns}"]
                    #hidden_state += down_projection
                    #hidden_state = F.layer_norm(hidden_state, normalized_shape=(hidden_state.shape[-1],), eps=1e-6)

                    hidden_state = (prev_state + down_projection[..., :rank]) @ self.latent_spaces[active_adapter][f"subspace_{ns}"]

                    if ns + 1 != self.num_subspaces:
                        prev_state = hidden_state @ self.mixture_projects[active_adapter][f"mixture_{ns}"].T
                        #prev_state = F.layer_norm(prev_state, normalized_shape=(prev_state.shape[-1],), eps=1e-6)

                    out_project[..., mid_rank:(mid_rank + rank)] = hidden_state
                    #hidden_state_sum += hidden_state @ self.up_project[active_adapter][:rank, :]#self.Vt[active_adapter][:rank, :]
                    mid_rank += rank
                
                
                k = [self.up_project[active_adapter][:self.latent_spaces[active_adapter][f"subspace_{i}"].shape[0], :] for i in range(self.num_subspaces)]
                vt_out = torch.cat(k, dim=0)
                
                hidden_state_sum = out_project @ vt_out
                # TODO: Alphas
                result = hidden_state_sum + result
            

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
    

def test_mars_lora_layer():
    # Example setup for testing the MarsLoraLayer
    # Set the dimensions of the input matrix and rank configurations for the layers
    original_matrix = torch.randn(1024, 1024)  # A mock matrix for decomposition
    adapter_name = "adapter1"
    ranks = [64, 32, 16]  # Different ranks for each adapter subspace
    lora_alphas = [1.0, 1.0, 1.0]  # Placeholder values for alpha scaling

    # Instantiate the MarsLoraLayer
    mars_lora_layer = Linear()

    mars_lora_layer.update_layer(original_matrix, adapter_name, ranks, lora_alphas)


    #mars_lora_layer.train()
    # Create a random tensor to simulate input data
    batch_size = 8
    sequence_length = 10
    hidden_dim = 1024  # Example hidden dimension
    x = torch.randn(batch_size, sequence_length, hidden_dim)

    # Pass through the layer
    output = mars_lora_layer(x)

    # Print the output to check if the forward pass works without errors
    print(f"Output shape: {output.shape}")

    # Check some intermediate matrices (U, Sigma, Vt) from SVD decomposition
    for adapter_name in mars_lora_layer.U:
        U_matrix = mars_lora_layer.U[adapter_name].detach().numpy()
        Sigma_vector = mars_lora_layer.Sigma[adapter_name].detach().numpy()
        Vt_matrix = mars_lora_layer.Vt[adapter_name].detach().numpy()

        print(f"Adapter: {adapter_name}")
        print(f"U matrix shape: {U_matrix.shape}")
        print(f"Sigma vector shape: {Sigma_vector.shape}")
        print(f"Vt matrix shape: {Vt_matrix.shape}")
        print("-" * 50)

    torch.onnx.export(
        mars_lora_layer,  # Model to export
        x,  # Dummy input for tracing
        "test.onnx",  # Output file name
        export_params=True,  # Store the trained parameters
        opset_version=20,  # ONNX opset version (make sure it's compatible with the model)
        do_constant_folding=False,  # Optimize constants
        input_names=['input'],  # Names for the model input
        output_names=['output'],  # Names for the model output
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Allow dynamic batch size
        #verbose=True  # Print information during the export process
    )

def check_initializers():
    # Load the ONNX model
    onnx_model = onnx.load('test.onnx')  # Replace with your ONNX model file path

    # Inspect the initializers (weights)
    initializers = onnx_model.graph.initializer

    # Print details about the initializers
    print(f"Total initializers in the model: {len(initializers)}")
    for initializer in initializers:
        print(f"Initializer Name: {initializer.name}")
        print(f"Shape: {initializer.dims}")
        print(f"Data Type: {initializer.data_type}")
        print("------------------------------------------------")

if __name__ == "__main__":
    # Run the test
    test_mars_lora_layer()
    check_initializers()





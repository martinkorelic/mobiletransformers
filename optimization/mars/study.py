from itertools import combinations
import torch
import tensorly as tl
from tensorly.decomposition import tensor_train
import tensorly.backend as T

# Make sure to set Tensorly to use PyTorch as backend
tl.set_backend('pytorch')

def tensor_train_decomposition(weight_matrix, ranks):
    """
    Perform Tensor Train decomposition on the input weight matrix.
    Args:
        weight_matrix (torch.Tensor): The original weight matrix to decompose.
        ranks (list): List of ranks for the decomposition (including the boundary ranks).
    Returns:
        list of tensors: The tensor train cores.
    """
    # Perform Tensor Train decomposition using Tensorly's MPS method (matrix product state)
    tensor_train_cores = tensor_train(weight_matrix, rank=ranks)
    return tensor_train_cores

def tensor_train_contract(cores):
    """
    Contract the Tensor Train decomposition back into the original matrix.
    Args:
        cores (list of torch.Tensor): The cores of the Tensor Train decomposition.
    Returns:
        torch.Tensor: The reconstructed matrix from the Tensor Train cores.
    """
    # Perform the contraction of the tensor train back to a matrix
    contracted_matrix = tl.tt_to_tensor(cores)
    return contracted_matrix

def factorize(n):
    """Finds all factors of a number."""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def find_best_shape(num_elements, target_order):
    """
    Finds the best shape for a higher-order tensor with a given target order.
    Args:
        num_elements (int): Total number of elements in the tensor.
        target_order (int): Desired order of the tensor.
    Returns:
        tuple: Best shape for the tensor.
    """
    factors = factorize(num_elements)
    best_shape = None
    min_diff = float('inf')

    # Iterate through all combinations of factors for the target order
    for dims in combinations(factors, target_order):
        if torch.prod(torch.tensor(dims)) == num_elements:  # Ensure the product matches the total elements
            diff = max(dims) - min(dims)  # Minimize the difference between dimensions
            if diff < min_diff:
                min_diff = diff
                best_shape = dims

    if best_shape is None:
        raise ValueError(f"Cannot reshape tensor into order-{target_order} with the given constraints.")
    return best_shape

def reshape_to_higher_order(tensor, target_order):
    """
    Reshapes a 2D tensor into a higher-order tensor.
    Args:
        tensor (torch.Tensor): Input 2D tensor.
        target_order (int): Desired order of the tensor.
    Returns:
        torch.Tensor: Reshaped tensor.
    """
    num_elements = tensor.numel()
    best_shape = find_best_shape(num_elements, target_order)
    return tensor.view(*best_shape)

def tt_tensor_elements(tt_cores):
    """
    Calculate the total number of elements in a TT decomposition.
    
    Args:
        tt_cores (list of torch.Tensor): List of TT cores, where each core is a 3D tensor 
                                         with shape (r_prev, n, r_next).
    Returns:
        int: Total number of elements in the TT decomposition.
    """
    total_elements = 0
    for core in tt_cores:
        print(core.shape)
        total_elements += core.numel()
    return total_elements

#self.og_shape = base_layer.weight.shape
#ranks = [self.og_shape[1], 256, 128, 64, 1]
#og_el = base_layer.weight.numel()

#bl = reshape_to_higher_order(base_layer.weight, target_order=4)

#tensor_train_cores = tensor_train_decomposition(bl, ranks)
#print(tensor_train_cores)
#tt_el = tt_tensor_elements(tensor_train_cores)

#print("OG elements:")
#print(og_el)
#print("TT elements:")
#print(tt_el)
#print(og_el / tt_el)

#torch.reshape(tensor_train_contract(tensor_train_cores), shape=og_shape)
#self.tt_cores = nn.ParameterList([nn.Parameter(core, requires_grad=False) for core in tensor_train_cores])
# Step 2: Contract the tensor train back to get the approximated matrix
#base_layer.weight = nn.Parameter(, requires_grad=False)

"""
# Normalize input to stabilize projections
x_normalized = F.layer_norm(x, normalized_shape=(x.shape[-1],))

# Collect hidden states from different subspaces
hidden_states = []
prev_state = torch.zeros(*x.shape[:-1], 
                    self.latent_spaces[active_adapter]["subspace_0"].shape[0], 
                    device=x.device)

for ns in range(self.num_subspaces):
rank = self.latent_spaces[active_adapter][f"subspace_{ns}"].shape[0]

# Stable projection
projected_state = self._stable_projection(
    x_normalized, 
    self.U[active_adapter], 
    self.Sigma[active_adapter], 
    rank, 
    x.device
)

# Inter-subspace mixing with controlled projection
if ns > 0:
    mixture_proj = self.mixture_projects[active_adapter][f"mixture_{ns-1}"] * 0.1
    prev_state = F.linear(prev_state, mixture_proj)

# Combine previous state and current projection
combined_state = prev_state + projected_state

# Project to current subspace
subspace_proj = self.latent_spaces[active_adapter][f"subspace_{ns}"]
hidden_state = combined_state @ subspace_proj

hidden_states.append(hidden_state)
prev_state = combined_state

# Adaptive scaling of subspace contributions
scaled_hidden_states = self._adaptive_subspace_scaling(hidden_states)

# Final summation with controlled scaling
hidden_state_sum = torch.zeros_like(result, device=x.device)
for hs, V in zip(scaled_hidden_states, 
            [self.Vt[active_adapter][:rank, :] for rank in 
            [self.latent_spaces[active_adapter][f"subspace_{ns}"].shape[0] 
            for ns in range(self.num_subspaces)]]):
# Soft-clamped final projection
projected = torch.clamp(hs @ V, min=-2.0, max=2.0)
hidden_state_sum += projected

# Final addition with small scaling
result = result + hidden_state_sum

"""
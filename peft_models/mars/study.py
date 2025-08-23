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


def sequential_svd(matrix, ranks, steps):
    """
    Perform sequential SVD decomposition on a given matrix with rank truncation at each step.
    Afterward, reconstruct the matrix and calculate the error accuracy drop-off.
    
    Args:
        matrix (np.ndarray): The input matrix to decompose.
        ranks (list[int]): A list of ranks to truncate to at each step.
        steps (int): The number of SVD steps to perform.
    
    Returns:
        float: The reconstruction error as a fraction of the original matrix norm.
    """
    assert len(ranks) == steps, "Number of ranks must match the number of steps."
    original_matrix = matrix.clone()
    intermediates = []  # Store intermediate results
    
    for step in range(steps):
        # Perform SVD decomposition
        U, Sigma, Vt = torch.linalg.svd(matrix, full_matrices=False)
        
        # Truncate based on the rank for this step
        rank = ranks[step]
        U_truncated = U[:, :rank]
        Sigma_truncated = torch.diag(Sigma[:rank]) 


        print(Sigma_truncated)
        Vt_truncated = Vt[:rank, :]

        # Store the truncated components
        intermediates.append(U_truncated)
        
        # Multiply Sigma and Vt to create the next matrix for SVD
        matrix = Sigma_truncated @ Vt_truncated
        #intermediates.append(matrix[:rank, :rank])

    intermediates.append(matrix)
    
    # Reconstruct the final matrix from all truncated components
    reconstructed_matrix = intermediates[0]
    for m in intermediates[1:]:
        reconstructed_matrix = reconstructed_matrix @ m
    
    # Compute reconstruction error
    error = torch.linalg.norm(original_matrix - reconstructed_matrix)
    original_norm = torch.linalg.norm(original_matrix)
    reconstruction_error = error / original_norm  # Fraction of the original norm
    
    return reconstruction_error

# Example usage:
if __name__ == "__main__":
    # Generate a random large matrix
    torch.manual_seed(42)
    matrix = torch.rand(1024, 1024)
    # Define ranks for each step and number of steps
    #ranks = [1, 32, 32, 1]  # Ranks at each step
    
    #tt_c = tensor_train_decomposition(weight_matrix=matrix, ranks=ranks)
    #tt_d = tensor_train_contract(tt_c)
    #error = torch.linalg.norm(matrix - tt_d)
    #original_norm = torch.linalg.norm(matrix)
    #reconstruction_error = error / original_norm  # Fraction of the original norm
    #print(f"Reconstruction error (TT): {reconstruction_error:.6f}")
    # Perform sequential SVD and get reconstruction error

    ranks = [128]
    steps = len(ranks)
 
    error = sequential_svd(matrix, ranks, steps=steps)
    print(f"Reconstruction error (SVD): {error:.6f}")
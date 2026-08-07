"""Orthogonal-matrix construction for MARS shared adapters.

VENDORED from ``research/pytorch_experiments/matrix_experiments.py`` (Migration Map S3).

``research/`` is deliberately kept OUT of the package — it is exploratory code with heavy, unpinned
dependencies (sklearn, scipy, a CCA module). ``mars/layer.py`` needed exactly one self-contained
function from it, so that function is vendored here rather than dragging the research tree into the
wheel. Keep the two in sync only deliberately; this copy is the one that ships.
"""

from __future__ import annotations

import torch


def create_orthogonal_matrices(
    m, num_matrices, mean=0.0, std=1.0, initial_matrix=None, device="cpu", generator=None
):
    """
    Create mutually orthogonal matrices using null space construction.

    This approach is based on the fact that if B is in the null space of A^T,
    then A^T @ B = 0, which means the matrices are orthogonal.

    Args:
        m (int): Dimension of each square matrix (m x m)
        num_matrices (int): Number of matrices to generate
        mean (float): Mean for random initialization
        std (float): Standard deviation for random initialization
        initial_matrix (torch.Tensor, optional): Initial matrix of size (m, m)
        device (str): Device to create tensors on
        generator (torch.Generator, optional): Random number generator

    Returns:
        torch.Tensor: Stacked matrices of size (m, m * num_matrices)
    """

    if initial_matrix is not None:
        if initial_matrix.shape != (m, m):
            raise ValueError(f"Initial matrix must be of size ({m}, {m})")
        A = initial_matrix.to(device)
        remaining = num_matrices - 1
    else:
        # Generate first matrix with controlled rank
        # We need null_dim >= remaining matrices
        # So rank should be <= m - (num_matrices - 1)
        remaining = num_matrices - 1
        max_rank = m - remaining

        if max_rank <= 0:
            raise ValueError(f"Cannot fit {num_matrices} matrices of size ({m}, {m}). Maximum possible: {m}")

        # Create a rank-deficient matrix by construction: A = U @ V^T
        # where U is (m, max_rank) and V is (m, max_rank)
        U = torch.nn.init.normal_(
            torch.empty((m, max_rank), device=device), mean=mean, std=std, generator=generator
        )
        V = torch.nn.init.normal_(
            torch.empty((m, max_rank), device=device), mean=mean, std=std, generator=generator
        )
        A = U @ V.T  # This has rank <= max_rank

    if remaining == 0:
        return A

    # Compute QR decomposition to get null space basis
    Q, _ = torch.linalg.qr(A, mode="complete")

    # The rank determines how much null space we have
    rank_A = torch.linalg.matrix_rank(A).item()
    null_dim = m - rank_A

    if null_dim == 0:
        raise ValueError(
            f"Matrix A is full rank (rank={rank_A}). No null space available for orthogonal matrices."
        )

    # Check if we have enough null space dimensions
    if remaining > null_dim:
        raise ValueError(
            f"Cannot generate {remaining} additional matrices. Null space dimension is only {null_dim}."
        )

    # Extract null space basis (last null_dim columns of Q)
    null_basis = Q[:, rank_A:]  # Shape: (m, null_dim)

    # Generate remaining matrices in the null space
    matrices = [A]

    for i in range(remaining):
        # Generate random coefficients for the null space basis
        coeffs = torch.nn.init.normal_(
            torch.empty((null_dim, m), device=device), mean=mean, std=std, generator=generator
        )

        # Create matrix in null space: null_basis @ coeffs
        B = null_basis @ coeffs  # Shape: (m, m)
        matrices.append(B)

    # Stack all matrices horizontally
    result = torch.cat(matrices, dim=1)
    return result

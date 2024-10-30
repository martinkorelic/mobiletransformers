import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD

class LoRA_XS_Linear(nn.Module):
    def __init__(self, weight, r=4, alpha=1.0):
        """
        LoRA-augmented linear layer.
        Args:
            input_size: The input size of the linear layer.
            output_size: The output size of the linear layer.
            r: The rank of the low-rank decomposition.
            alpha: Scaling factor for the low-rank adaptation.
        """
        """
        LoRA-augmented linear layer using SVD for initialization.
        
        Args:
            weight: The weight matrix of the original linear layer (frozen).
            r: The rank of the low-rank decomposition.
        """
        super(LoRA_XS_Linear, self).__init__()
        w = weight.T.cpu().detach().numpy()

        # Perform truncated SVD on the original weight matrix
        svd = TruncatedSVD(n_components=r, n_iter=1)
        svd.fit(w.T)
        reduced_matrix = svd.transform(w.T)

        # Create frozen low-rank matrices A and B
        self.A = nn.Parameter(torch.tensor(reduced_matrix.T, dtype=torch.float32), requires_grad=False)  # A = U_r * Σ_r
        self.B = nn.Parameter(torch.tensor(svd.components_.T, dtype=torch.float32), requires_grad=False) # B = V_r^T
        
        # Trainable low-rank matrix T_r of size r x r
        self.T_r = nn.Parameter(torch.randn(r, r) * 0.01)
        
        # Store the input and output size of the original weight matrix
        self.input_size = weight.shape[1]
        self.output_size = weight.shape[0]
        self.rank = r
        self.alpha = alpha

    def forward(self, x):


        return result

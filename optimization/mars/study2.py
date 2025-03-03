import torch
import torch.nn as nn

class SparseLoRALayer(nn.Module):
    def __init__(self, input_features, output_features, inner_rank, input_mask, output_mask):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.inner_rank = inner_rank

        # Create dense matrices for initialization
        A_dense = torch.randn(input_features, inner_rank)
        B_dense = torch.randn(inner_rank, output_features)

        # Apply input mask to zero out unimportant columns in A
        A_dense *= input_mask.unsqueeze(1)

        # Apply output mask to zero out unimportant rows in B
        B_dense *= output_mask.unsqueeze(0)

        a = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float64)
        self.A = a.to_sparse_csr()

        b = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float64)
        self.B = b.to_sparse_csr()

    def forward(self, x):
        x_flat = x.view(-1, x.shape[-1])
        x_down = torch.sparse.mm(self.A, x_flat.T).T
        lora_out = torch.sparse.mm(self.B, x_down.T).T
        return lora_out.view(x.shape[0], x.shape[1], -1)

input_features = 4
output_features = 4
inner_rank = 4

# Binary masks for important input/output channels
input_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(input_features)], dtype=torch.float32)
output_mask = torch.tensor([1 if i % 3 == 0 else 0 for i in range(output_features)], dtype=torch.float32)

# Instantiate the layer
sparse_lora_layer = SparseLoRALayer(input_features, output_features, inner_rank, input_mask, output_mask)

# Test input
batch_size, seq_len = 2, 5
x = torch.randn(batch_size, seq_len, input_features, dtype=torch.float64)

# Forward pass
output = sparse_lora_layer(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)

# TODO: Does not support Sparse CSR
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(sparse_lora_layer, x)
onnx_program.save("model.onnx")

def sparse_mul():

    x = torch.rand((1, 12, 4), dtype=torch.float64)

    a = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float64)

    b = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float64)

    x = x.view(-1, x.shape[-1])
    print(x.shape)
    a = a.to_sparse_csr()
    b = b.to_sparse_csr()

    mid = torch.sparse.mm(a, x.T).T
    out = torch.sparse.mm(b, mid.T).T

    print(out.shape)
    print(out)
    """
    a = a.to_sparse_csr()
    torch.Size([12, 4])
    tensor([[7.5658, 0.0000, 7.5658, 7.5658],
            [4.8801, 0.0000, 4.8801, 4.8801],
            [7.4587, 0.0000, 7.4587, 7.4587],
            [6.6181, 0.0000, 6.6181, 6.6181],
            [5.8133, 0.0000, 5.8133, 5.8133],
            [7.0007, 0.0000, 7.0007, 7.0007],
            [8.0715, 0.0000, 8.0715, 8.0715],
            [4.4390, 0.0000, 4.4390, 4.4390],
            [7.7564, 0.0000, 7.7564, 7.7564],
            [6.1653, 0.0000, 6.1653, 6.1653],
            [4.5225, 0.0000, 4.5225, 4.5225],
            [6.6311, 0.0000, 6.6311, 6.6311]], dtype=torch.float64)
    """
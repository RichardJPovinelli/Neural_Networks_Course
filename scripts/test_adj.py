import torch

A = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=torch.float32)
h0 = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
A_hat = A + torch.eye(A.size(0))
deg = A_hat.sum(dim=1, keepdim=True)
out = (A_hat @ h0) / deg
print("out:", out)

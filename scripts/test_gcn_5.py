import torch

num_nodes = 5
A = torch.tensor(
    [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]], dtype=torch.float32
)

I = torch.eye(num_nodes)
A_tilde = A + I

deg = A_tilde.sum(dim=1)
D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))

H0 = torch.tensor([[1.0, 0.5], [2.0, -1.0], [0.0, 1.0], [3.0, 2.0], [4.0, 0.0]])
W = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

H_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt @ H0
H1 = H_hat @ W
print("H1:", H1)

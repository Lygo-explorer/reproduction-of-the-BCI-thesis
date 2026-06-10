import torch

def laplacian_norm_batch(A, eps=1e-6):
    """
    Graph Laplacian normalization:
        A_norm = D^{-1/2} A D^{-1/2}

    A: (B, N, N)
    return: (B, N, N)
    """

    # 1. degree matrix (sum over last dim)
    # shape: (B, N)
    deg = torch.sum(A, dim=-1)

    # 2. D^{-1/2}
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)

    # 3. reshape for broadcasting
    # (B, N, 1) and (B, 1, N)
    deg_inv_sqrt_row = deg_inv_sqrt.unsqueeze(-1)
    deg_inv_sqrt_col = deg_inv_sqrt.unsqueeze(-2)

    # 4. normalization
    A_norm = deg_inv_sqrt_row * A * deg_inv_sqrt_col

    return A_norm
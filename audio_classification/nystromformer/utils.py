import torch
from continual.logging import getLogger

# OVERFLOW_VALUE = 88.
# UNDERFLOW_VALUE = -103.
OVERFLOW_VALUE = 80.
UNDERFLOW_VALUE = -95.

logger_once = getLogger(__name__, log_once=True)

# Generic qk_product
def qk_product(q, k, stable_exp=None):
    matrix = torch.bmm(q, torch.transpose(k, 1, 2))
    if stable_exp is not None:
        # Based on https://github.com/pytorch/pytorch/blob/34bce27f0d12bf7226b37dfe365660aad456701a/aten/src/ATen/native/SoftMax.cpp#L234
        matrix -= stable_exp

    matrix = torch.exp(matrix)
    if torch.any(torch.isinf(matrix)):
        logger_once.warn("qk product produces overflow infinite after exponential")
    elif torch.all(matrix == 0):
        logger_once.warn("qk product produces underflow zeros after exponential")
    return matrix

# Makes a continual step removing the first column and row from M and adding a new column and row defined as:
# [M a]
# [b c]
def continual_matrix_concat(M, a, b):
    return add_continual_vector(
        add_continual_vector(M, a, dim=2),
        b,
        dim=1,
    )

# Computes the pseudo-inverse of a matrix with the iterative method. See Nyströmformer paper for more details
def iterative_inv(mat, n_iter=6):
    I = torch.eye(mat.size(-1), device=mat.device)
    K = mat
    V = 1 / (torch.max(torch.sum(torch.abs(K), dim=-2)) * torch.max(torch.sum(torch.abs(K), dim=-1))) * K.transpose(-1,
                                                                                                                    -2)
    for _ in range(n_iter):
        KV = torch.matmul(K, V)
        V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
    return V

# Performs the row-wise multiplication between the inverse of a diagonal vector d_M and a matrix M
# and returns the result
def odot(d_M, M):
    M = M / d_M
    # Replace all zero-divisions by zero
    return torch.nan_to_num(M, posinf=0.0, neginf=0.0)

def add_continual_vector(matrix, new_vector, dim=1):
    matrix = torch.roll(matrix, -1, dims=dim)
    if dim == 1:
        matrix[:, -1:] = new_vector
    elif dim == 2:
        matrix[:, :, -1:] = new_vector
    return matrix

import torch
import warnings

# OVERFLOW_VALUE = 88.
# UNDERFLOW_VALUE = -103.
OVERFLOW_VALUE = 80.
UNDERFLOW_VALUE = -95.

def fix_overflow(tensor: torch.Tensor) -> torch.Tensor:
    diff = torch.max(tensor) - OVERFLOW_VALUE
    return tensor - diff

def fix_underflow(tensor: torch.Tensor) -> torch.Tensor:
    diff = torch.min(tensor) + UNDERFLOW_VALUE
    return tensor + diff

# Generic qk_product
def qk_product(q, k, stable_exp=False):
    matrix = torch.bmm(q, torch.transpose(k, 1, 2))
    if stable_exp:
        max_matrix = torch.max(matrix, dim=2).values
        min_matrix = torch.min(matrix, dim=2).values

        overflow = max_matrix > OVERFLOW_VALUE
        underflow = min_matrix < UNDERFLOW_VALUE

        if overflow.any() or underflow.any():
            precision_error = overflow & underflow

            if precision_error.any():
                warnings.warn('Both overflow and underflow are happening. Consider normalizing the data')

            overflow = overflow.unsqueeze(-1).repeat(1, 1, matrix.size()[2])
            underflow = underflow.unsqueeze(-1).repeat(1, 1, matrix.size()[2])

            matrix = torch.where(overflow, fix_overflow(matrix), matrix)
            matrix = torch.where(underflow & ~overflow, fix_underflow(matrix), matrix)

    return torch.exp(matrix)

# Makes a continual step removing the first column and row from M and adding a new column and row defined as:
# [M a]
# [b c]
def continual_matrix_concat(M, a, b, c):
    return torch.cat((
            torch.cat((
                M,
                a
                ),
                dim=2
            ),
            torch.cat((
                b,
                c
                ),
                dim=2
            )
        ),
        dim=1
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

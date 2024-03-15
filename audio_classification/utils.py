import torch

# Makes a continual step removing the first column and row from A and adding a new column and row defined as:
# [A b]
# [c d]
def continual_matrix_concat(A, b, c, d):
    return torch.cat((
        torch.cat((
            A[:, :, 1:, 1:],
            b
            ),
            dim=3
        ),
        torch.cat((
            c,
            d
        ),
            dim=3
        )
        ),
        dim=2
    )

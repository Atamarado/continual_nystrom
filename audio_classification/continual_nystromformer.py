import torch.nn as nn
import torch
import math
import continual as co
from functools import partial

from typing import Optional, Tuple
from torch import Tensor

from continual.logging import getLogger

from utils import continual_matrix_concat, qk_product, iterative_inv, odot

logger = getLogger(__name__)
logger_once = getLogger(__name__, log_once=True)

State = Tuple[
    # Landmarks
    Tensor,  # Q_tilde (B, m, d)
    Tensor,  # K_tilde (B, m, d)

    Tensor,  # Q (B, n, d) # Used to compute new landmarks
    Tensor,  # K (B, n, d) # Used to retrieve k_old and to compute new landmarks
    Tensor,  # V (B, n, d) # Used for the last multiplication when updating landmarks

    # Values used for updates without landmark updates
    Tensor,  # BetaD_GammaD_mem (B, n-1, m)
    Tensor,  # Gamma_D (B, m, m)
    Tensor,  # d_Delta_prev (B, m, 1)
    Tensor,  # DeltaV_prev (B, m, d)

    # Additional values just used for updates with landmark updates
    Tensor,  # d_Beta_prev (B, n-1, 1) # We only need to store the m-1 values for when we update the landmarks
    Tensor,  # d_Gamma_prev (B, m-1, 1)
    Tensor,  # Beta_mem (B, n-1, m-1)
    Tensor,  # Gamma_mem (B, m-1, m-1)

    Tensor,  # state_index
    int,  # iteration
]

def _scaled_dot_product_attention_default_state(
    batch_size: int,
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    num_landmarks: int,
    init_fn=torch.zeros,
    dtype=None,
    device=None,
) -> State:
    init_fn = partial(init_fn, dtype=dtype, device=device)
    d = embed_dim // num_heads
    B = batch_size * num_heads
    n = sequence_len
    m = num_landmarks

    default_state = (
        init_fn((B, m, d)),
        init_fn((B, m, d)),

        init_fn((B, n, d)),
        init_fn((B, n, d)),
        init_fn((B, n, d)),

        init_fn((B, n-1, m)),
        init_fn((B, m, m)),
        init_fn((B, m, 1)),
        init_fn((B, m, d)),

        init_fn(B, n-1, 1),
        init_fn(B, m-1, 1),
        init_fn(B, n-1, m-1),
        init_fn(B, m-1, m-1),

        init_fn(0),
        0
    )
    return default_state


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    m: int = 10,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    use_conv: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention as in Nyströmformer on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and state for continual inference.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        m: int. Number of landmarks used for the Nyström method. Default=10
        use_conv: Indicates whether to apply a convolution layer over the value input or not. Default=False

    Shape:
        - q: :math:`(B, N, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, N, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, N, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, N, N)` or a 2D tensor of
            shape :math:`(N, N)`.

        - Output: attention values have shape :math:`(B, N, E)`; attention weights
            are a Shape tuple
    """
    if attn_mask is not None:  # pragma: no cover
        logger_once.warning("attn_mask is not supported yet and will be skipped")
    if dropout_p != 0.0:  # pragma: no cover
        logger_once.warning("dropout_p is not supported yet and will be skipped")
    if use_conv:
        logger_once.warning("use_conv is not supported yet and will be skipped")

    B, N, E = q.shape

    if N % m != 0:
        raise ValueError("N must be divisible by m to apply Nyströmformers")


    device = q.device

    Q = q / math.sqrt(math.sqrt(E))
    K = k / math.sqrt(math.sqrt(E))
    V = v

    # Landmark selection
    Q_tilde = Q.reshape(B, m, N // m, E).mean(dim=-2)
    K_tilde = K.reshape(B, m, N // m, E).mean(dim=-2)

    Beta = qk_product(Q, K_tilde)  # Note that the first row will be old at the next iteration
    Gamma = qk_product(Q_tilde, K_tilde)
    Delta = qk_product(Q_tilde, K)

    # The first set of diagonals are computed in the same way as in the original paper
    d_Beta = torch.bmm(Beta, torch.ones((B, m, 1), device=device))
    d_Gamma = torch.bmm(Gamma, torch.ones((B, m, 1), device=device))
    d_Delta = torch.bmm(Delta, torch.ones((B, N, 1), device=device))

    Beta_D = odot(d_Beta, Beta)
    Gamma_D = odot(d_Gamma, Gamma)
    Gamma_D = iterative_inv(Gamma_D)  # TODO: Improved formulation coming

    BetaD_GammaD = torch.bmm(Beta_D, Gamma_D)

    Delta_V = torch.bmm(Delta, V)
    Delta_DV = odot(d_Delta, Delta_V)

    output = torch.bmm(BetaD_GammaD, Delta_DV)

    prev_state = (
        Q_tilde,
        K_tilde,

        Q,
        K,
        V,

        BetaD_GammaD[:, 1:],
        Gamma_D,
        d_Delta,
        Delta_V,

        d_Beta[:, 1:],
        d_Gamma[:, 1:],
        Beta[:, 1:, 1:],
        Gamma[:, 1:, 1:],

        torch.empty(), # TODO: Learn what to do with it
        0
    )

    return output, prev_state


# Adapted from https://github.com/LukasHedegaard/continual-inference/blob/b75acad64abf26ffd5ae693bf6eecff7536468cf/continual/multihead_attention/retroactive_mha.py#L47
def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    last_iter: int,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    use_conv: bool = False
) -> Tuple[Tensor, State, Tuple]: # TODO: Change back to [Tensor, State]
    """
    Computes the Continual Retroactive Scaled Nyströmformer Dot-Product Attention on query, key and value tensors.
    Returns attended values and updated states.

    Args:
        q_step, k_step, v_step: query, key and value tensors for a step. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        use_conv: Indicates whether to apply a convolution layer over the value input or not. Default=False

    Shape:
        - q_step: :math:`(B, E)` where B is batch size and E is embedding dimension.
        - k_step: :math:`(B, E)` where B is batch size and E is embedding dimension.
        - v_step: :math:`(B, E)` where B is batch size and E is embedding dimension.

        - Output: attention values have shape :math:`(B, n, E)`; new state
    """
    if attn_mask is not None:  # pragma: no cover
        logger_once.warning("attn_mask is not supported yet and will be skipped")
    if dropout_p != 0.0:  # pragma: no cover
        logger_once.warning("dropout_p is not supported yet and will be skipped")
    if use_conv:
        logger_once.warning("use_conv is not supported yet and will be skipped")

    (
        Q_tilde,
        K_tilde,

        Q,
        K,
        V,

        BetaD_GammaD_mem,
        Gamma_D,
        d_Delta_prev,
        DeltaV_prev,

        d_Beta_prev,
        d_Gamma_prev,
        Beta_mem,
        Gamma_mem,

        state_index,  # TODO: Check if it's necessary
        iteration
    ) = prev_state

    iteration += 1

    device = q_step.device

    B, E = q_step.shape
    _, n, m = BetaD_GammaD_mem.shape
    n += 1
    tokens_per_landmark = n // m
    d = E

    q_new = torch.reshape(q_step, (B, 1, d))
    k_new = torch.reshape(k_step, (B, 1, d))
    v_new = torch.reshape(v_step, (B, 1, d))

    q_new = torch.div(q_new, math.sqrt(math.sqrt(d)))
    k_new = torch.div(k_new, math.sqrt(math.sqrt(d)))

    k_old = K[:, 0].unsqueeze(-2)
    v_old = V[:, 0].unsqueeze(-2)

    Q = torch.cat((
        Q[:, 1:],
        q_new
        ),
        dim=1
    )
    K = torch.cat((
        K[:, 1:],
        k_new
        ),
        dim=1
    )
    V = torch.cat((
        V[:, 1:],
        v_new
        ),
        dim=1
    )

    Beta_D_new = None # TODO: Remove later

    if iteration % tokens_per_landmark == tokens_per_landmark - 1:
        # Landmark changes
        # New landmarks
        q_tilde_new = Q[:, -tokens_per_landmark:].mean(dim=-2).unsqueeze(-2)
        k_tilde_new = K[:, -tokens_per_landmark:].mean(dim=-2).unsqueeze(-2)

        k_tilde_old = K_tilde[:, 0].unsqueeze(dim=-2)

        Q_mem = Q[:, :-1]
        K_mem = K[:, :-1]

        Q_tilde_mem = Q_tilde[:, 1:]
        K_tilde_mem = K_tilde[:, 1:]

        # Update Q_tilde, K_tilde
        Q_tilde = torch.cat((
            Q_tilde_mem,
            q_tilde_new
            ),
            dim=1
        )

        K_tilde = torch.cat((
            K_tilde_mem,
            k_tilde_new
            ),
            dim=1
        )

        # Beta update
        Beta_A = qk_product(Q_mem, k_tilde_new)
        Beta_B = qk_product(q_new, K_tilde_mem)
        Beta_C = qk_product(q_new, k_tilde_new)
        Beta = continual_matrix_concat(Beta_mem, Beta_A, Beta_B, Beta_C)
        Beta_mem = Beta[:, 1:, 1:]

        # Gamma update
        Gamma_A = qk_product(Q_tilde_mem, k_tilde_new)
        Gamma_B = qk_product(q_tilde_new, K_tilde_mem)
        Gamma_C = qk_product(q_tilde_new, k_tilde_new)
        Gamma = continual_matrix_concat(Gamma_mem, Gamma_A, Gamma_B, Gamma_C)
        Gamma_mem = Gamma[:, 1:, 1:]

        # d_Beta update
        d_Beta = d_Beta_prev - qk_product(Q_mem, k_tilde_old) + qk_product(Q_mem, k_tilde_new)
        d_Beta_new = torch.cat((
            qk_product(q_new, K_tilde_mem),
            qk_product(q_new, k_tilde_new)
            ),
            dim=2
        )
        d_Beta_new = torch.bmm(d_Beta_new, torch.ones((B, m, 1), device=device))
        d_Beta = torch.cat((
            d_Beta,
            d_Beta_new
            ),
            dim=1
        )
        d_Beta_mem = d_Beta[:, 1:]

        # Next: d_Gamma update
        d_Gamma = d_Gamma_prev - qk_product(Q_tilde_mem, k_tilde_old) + qk_product(Q_tilde_mem, k_tilde_new)
        d_Gamma_new = torch.cat((
            qk_product(q_tilde_new, K_tilde_mem),
            qk_product(q_tilde_new, k_tilde_new)
        ),
            dim=2
        )
        d_Gamma_new = torch.bmm(d_Gamma_new, torch.ones((B, m, 1), device=device))
        d_Gamma = torch.cat((
            d_Gamma,
            d_Gamma_new
            ),
            dim=1
        )
        d_Gamma_mem = d_Gamma[:, 1:]

        # Next: d_Delta update
        Delta_old = qk_product(Q_tilde_mem, k_old)
        Delta_new = qk_product(Q_tilde_mem, k_new)

        d_Delta = d_Delta_prev[:, 1:] - Delta_old + Delta_new
        d_Delta_new = torch.cat((
            qk_product(q_tilde_new, K_mem),
            qk_product(q_tilde_new, k_tilde_new)
            ),
            dim=2
        )
        d_Delta_new = torch.bmm(d_Delta_new, torch.ones((B, n, 1), device=device))
        d_Delta = torch.cat((
            d_Delta,
            d_Delta_new
            ),
            dim=1
        )

        # Delta^D V
        DeltaV_prev = DeltaV_prev[:, 1:]
        DeltaV_prev = DeltaV_prev - torch.bmm(Delta_old, v_old) + torch.bmm(Delta_new, v_new)

        DeltaV_new_row = torch.bmm(qk_product(q_tilde_new, K), V)

        Delta_V = torch.cat((
            DeltaV_prev,
            DeltaV_new_row
            ),
            dim=1
        )
        # Delta^D odot
        DeltaD_V = odot(d_Delta, Delta_V)

        # Vector matrix multiplications
        Beta_D = odot(d_Beta, Beta)
        Gamma_D = iterative_inv(odot(d_Gamma, Gamma))

        BetaD_GammaD = torch.bmm(Beta_D, Gamma_D)
        BetaD_GammaD_mem = BetaD_GammaD[:, 1:]

    else:
        # Same landmarks
        d_Gamma_mem = d_Gamma_prev

        # Beta^D * Gamma^D computation
        Beta_new = qk_product(q_new, K_tilde)
        d_Beta_new = torch.bmm(Beta_new, torch.ones((B, m, 1), device=device))
        Beta_D_new = odot(d_Beta_new, Beta_new)

        BetaD_GammaD = torch.cat((
            BetaD_GammaD_mem,
            torch.bmm(Beta_D_new, Gamma_D)
            ),
            dim=1
        )
        BetaD_GammaD_mem = BetaD_GammaD[:, 1:]

        # Delta^D * V computation
        Delta_old = qk_product(Q_tilde, k_old)
        Delta_new = qk_product(Q_tilde, k_new)

        Delta_V = DeltaV_prev - torch.bmm(Delta_old, v_old) + torch.bmm(Delta_new, v_new)
        d_Delta = d_Delta_prev - Delta_old + Delta_new

        # Delta^D odot
        DeltaD_V = odot(d_Delta, Delta_V)

        # Update Beta_mem and d_Beta_mem
        Beta_mem = torch.cat((
            Beta_mem[:, 1:],
            Beta_new[:, :, 1:]
            ),
            dim=1
        )
        d_Beta_mem = torch.cat((
            d_Beta_prev[:, 1:],
            d_Beta_new
            ),
            dim=1
        )

    # Operations common to both branches
    output = torch.bmm(BetaD_GammaD, DeltaD_V)

    new_states = (
        Q_tilde,
        K_tilde,

        Q,
        K,
        V,

        BetaD_GammaD_mem,
        Gamma_D,
        d_Delta,
        Delta_V,

        d_Beta_mem,
        d_Gamma_mem,
        Beta_mem,
        Gamma_mem,

        state_index,
        iteration
    )

    if iteration == last_iter:
        return output, new_states, (Beta_D_new, Gamma_D)
    return output, new_states, ()

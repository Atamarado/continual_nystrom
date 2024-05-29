import torch.nn as nn
import torch
import math
import continual as co
from functools import partial

from typing import Optional, Tuple
from torch import Tensor

from continual.logging import getLogger
from continual.module import CoModule
from continual.module import _callmode

from .utils import continual_matrix_concat, qk_product, iterative_inv, odot, add_continual_vector
from .nystromformer import NystromMultiheadAttention

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
    Tensor,  # BetaD_GammaD_prev (B, n, m)
    Tensor,  # Gamma_D (B, m, m)
    Tensor,  # d_Delta_prev (B, m, 1)
    Tensor,  # DeltaV_prev (B, m, d)

    # Additional values just used for updates with landmark updates
    Tensor,  # d_Beta_prev (B, n, 1)
    Tensor,  # d_Gamma_prev (B, m, 1)
    Tensor,  # Beta_prev (B, n, m)
    Tensor,  # Gamma_prev (B, m, m)

    Tensor,  # q_tilde_new (B, 1, d)
    Tensor,  # k_tilde_new (B, 1, d)

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

        init_fn((B, n, m)),
        init_fn((B, m, m)),
        torch.full((B, m, 1), n, dtype=torch.float32),  # init_fn((B, m, 1))
        init_fn((B, m, d)),

        init_fn(B, n, 1),
        init_fn(B, m, 1),
        torch.full((B, n, m), 1., dtype=torch.float32),
        init_fn(B, m, m),

        init_fn(B, 1, d),
        init_fn(B, 1, d),

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

    Q = torch.div(q, math.sqrt(math.sqrt(E)))
    K = torch.div(k, math.sqrt(math.sqrt(E)))
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

        BetaD_GammaD,
        Gamma_D,
        d_Delta,
        Delta_V,

        d_Beta,
        d_Gamma,
        Beta,
        Gamma,

        torch.zeros((B, 1, E), device=device),
        torch.zeros((B, 1, E), device=device),

        0
    )

    return output, prev_state


# Adapted from https://github.com/LukasHedegaard/continual-inference/blob/b75acad64abf26ffd5ae693bf6eecff7536468cf/continual/multihead_attention/retroactive_mha.py#L47
def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    use_conv: bool = False,
    fixed_landmarks: bool = False,
    single_output: bool = False,
    stable_exp: float = None,
    return_kernels: bool = False,
    compute_inverse: bool = True,
) -> Tuple[Tensor, State]:
    """
    Computes the Continual Retroactive Scaled Nyströmformer Dot-Product Attention on query, key and value tensors.
    Returns attended values and updated states.

    Args:
        q_step, k_step, v_step: query, key and value tensors for a step. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        use_conv: Indicates whether to apply a convolution layer over the value input or not. Default=False
        fixed_landmarks: Whether to update landmarks or not. Default=False
        single_output: Indicates which mode the attention step is done. Default=False

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

        BetaD_GammaD_prev,
        Gamma_D,
        d_Delta_prev,
        DeltaV_prev,

        d_Beta_prev,
        d_Gamma_prev,
        Beta_prev,
        Gamma_prev,

        q_tilde_new,
        k_tilde_new,

        iteration
    ) = prev_state

    iteration += 1

    device = q_step.device

    B, E = q_step.shape
    _, n, m = BetaD_GammaD_prev.shape
    tokens_per_landmark = n // m

    if (iteration % n) < (n % m):
        tokens_per_landmark += 1  # If the token we are replacing corresponds to one of the longest landmarks, then we add one more

    d = E

    q_new = torch.reshape(q_step, (B, 1, d))
    k_new = torch.reshape(k_step, (B, 1, d))
    v_new = torch.reshape(v_step, (B, 1, d))

    q_new = torch.div(q_new, math.sqrt(math.sqrt(d)))
    k_new = torch.div(k_new, math.sqrt(math.sqrt(d)))

    k_old = K[:, 0].unsqueeze(-2)
    v_old = V[:, 0].unsqueeze(-2)

    Q = add_continual_vector(Q, q_new, dim=1)
    K = add_continual_vector(K, k_new, dim=1)
    V = add_continual_vector(V, v_new, dim=1)

    if not fixed_landmarks:
        # Add the contribution of q_new, k_new to the landmarks
        q_tilde_new += torch.div(q_new, tokens_per_landmark)
        k_tilde_new += torch.div(k_new, tokens_per_landmark)

    if not fixed_landmarks and (iteration % tokens_per_landmark == 0):
        # Landmark changes

        k_tilde_old = K_tilde[:, 0].unsqueeze(dim=-2)
        Q_tilde_prev = Q_tilde

        # Update Q_tilde, K_tilde
        Q_tilde = add_continual_vector(Q_tilde, q_tilde_new, dim=1)
        K_tilde = add_continual_vector(K_tilde, k_tilde_new, dim=1)

        # Gamma update
        Gamma_A = qk_product(Q_tilde_prev, k_tilde_new, stable_exp=stable_exp)
        Gamma_B = qk_product(q_tilde_new, K_tilde, stable_exp=stable_exp)
        Gamma = continual_matrix_concat(Gamma_prev, Gamma_A, Gamma_B)
        Gamma_prev = Gamma

        # Next: d_Gamma update
        d_Gamma_new = Gamma_B # qk_product(q_tilde_new, K_tilde, stable_exp=stable_exp)
        d_Gamma_new = torch.bmm(d_Gamma_new, torch.ones((B, m, 1), device=device))

        d_Gamma = d_Gamma_prev - qk_product(Q_tilde_prev, k_tilde_old, stable_exp=stable_exp) + qk_product(Q_tilde_prev, k_tilde_new, stable_exp=stable_exp)
        d_Gamma = add_continual_vector(d_Gamma, d_Gamma_new, dim=1)
        d_Gamma_prev = d_Gamma

        Gamma_D = iterative_inv(odot(d_Gamma, Gamma)) if compute_inverse else 1./odot(d_Gamma, Gamma)

        # Beta, d_Beta update
        Beta_B = qk_product(q_new, K_tilde, stable_exp=stable_exp)
        d_Beta_new = torch.bmm(Beta_B, torch.ones((B, m, 1), device=device))

        if single_output:
            Beta_D_new = odot(d_Beta_new, Beta_B)
            Beta_D_Gamma_D_new = torch.bmm(Beta_D_new, Gamma_D)
        else:
            Beta_A = qk_product(Q, k_tilde_new, stable_exp=stable_exp)
            Beta = continual_matrix_concat(Beta_prev, Beta_A, Beta_B)
            Beta_prev = Beta

            # d_Beta update
            d_Beta = d_Beta_prev - qk_product(Q, k_tilde_old, stable_exp=stable_exp) + Beta_A
            d_Beta = add_continual_vector(d_Beta, d_Beta_new, dim=1)
            d_Beta_prev = d_Beta

            # Vector matrix multiplications
            Beta_D = odot(d_Beta, Beta)

            BetaD_GammaD = torch.bmm(Beta_D, Gamma_D)

        # Next: d_Delta update
        Delta_old = qk_product(Q_tilde_prev, k_old, stable_exp=stable_exp)
        Delta_new = qk_product(Q_tilde_prev, k_new, stable_exp=stable_exp)

        d_Delta = d_Delta_prev - Delta_old + Delta_new

        q_tilde_new_K = qk_product(q_tilde_new, K, stable_exp=stable_exp)
        d_Delta_new = q_tilde_new_K

        d_Delta_new = torch.bmm(d_Delta_new, torch.ones((B, n, 1), device=device))
        d_Delta = add_continual_vector(d_Delta, d_Delta_new, dim=1)

        # Delta^D V
        DeltaV_prev = DeltaV_prev - torch.bmm(Delta_old, v_old) + torch.bmm(Delta_new, v_new)
        DeltaV_new_row = torch.bmm(q_tilde_new_K, V)
        Delta_V = add_continual_vector(DeltaV_prev, DeltaV_new_row, dim=1)

        # Delta^D odot
        DeltaD_V = odot(d_Delta, Delta_V)

        # Reset new landmark memory
        q_tilde_new = torch.zeros((B, 1, d), device=device)
        k_tilde_new = torch.zeros((B, 1, d), device=device)

    else:
        # Same landmarks
        # Beta^D * Gamma^D computation
        Beta_new = qk_product(q_new, K_tilde, stable_exp=stable_exp)
        d_Beta_new = torch.bmm(Beta_new, torch.ones((B, m, 1), device=device))
        Beta_D_new = odot(d_Beta_new, Beta_new)

        Beta_D_Gamma_D_new = torch.bmm(Beta_D_new, Gamma_D)

        if not single_output:
            BetaD_GammaD = add_continual_vector(BetaD_GammaD_prev, Beta_D_Gamma_D_new, dim=1)

        # Delta^D * V computation
        Delta_old = qk_product(Q_tilde, k_old , stable_exp=stable_exp)
        Delta_new = qk_product(Q_tilde, k_new , stable_exp=stable_exp)

        Delta_V = DeltaV_prev - torch.bmm(Delta_old, v_old) + torch.bmm(Delta_new, v_new)
        d_Delta = d_Delta_prev - Delta_old + Delta_new

        # Delta^D odot
        DeltaD_V = odot(d_Delta, Delta_V)

        # Update Beta_prev and d_Beta_prev
        if not single_output:
            Beta_prev = add_continual_vector(Beta_prev, Beta_new, dim=1)
            d_Beta_prev = add_continual_vector(d_Beta_prev, d_Beta_new)

    # Operations common to both branches
    if single_output:
        output = torch.bmm(Beta_D_Gamma_D_new, DeltaD_V)
        # Update dummy state
        BetaD_GammaD = BetaD_GammaD_prev
    else:
        output = torch.bmm(BetaD_GammaD, DeltaD_V)

    new_states = (
        Q_tilde,
        K_tilde,

        Q,
        K,
        V,

        BetaD_GammaD,  # TODO: not used for single_output
        Gamma_D,
        d_Delta,
        Delta_V,

        d_Beta_prev, # TODO: not used for single_output
        d_Gamma_prev,
        Beta_prev, # TODO: not used for single_output
        Gamma_prev,

        q_tilde_new,
        k_tilde_new,

        iteration
    )

    if return_kernels:
        Beta = qk_product(Q, K_tilde, stable_exp=stable_exp)
        d_Beta = torch.bmm(Beta, torch.ones((B, m, 1), device=device))
        Beta_D = odot(d_Beta, Beta)

        Delta = qk_product(Q_tilde, K, stable_exp=stable_exp)
        d_Delta = torch.bmm(Delta, torch.ones((B, n, 1), device=device))
        Delta_D = odot(d_Delta, Delta)

        # output = torch.bmm(torch.bmm(Beta_D, Gamma_D), torch.bmm(Delta_D, V))
        return output, new_states, Beta_D, Gamma_D, Delta_D
    return output, new_states


class ContinualNystromMultiheadAttention(NystromMultiheadAttention):
    """
    MultiHeadAttention with retroactively or single output updated attention outputs during continual inference.

    Continual MHAs were proposed by Hedegaard et al. in
    "Continual Transformers: Redundancy-Free Attention for Online Inference"
    https://arxiv.org/abs/2201.06268 (paper) https://www.youtube.com/watch?v=gy802Tlp-eQ (video).

    This module augments the MultiHeadAttention in PyTorch with
    `forward_step` / `forward_steps` functions, in which one / more
    query, key, and value tokens are passed to yield the multihead attentions, and
    updated outputs are computed for each token input.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        num_landmarks: Number of landmarks used for the Nyström approximation.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        device: torch device to initialize layer on. Defaults to None.
        dtype: datatype of layer parameters. Defaults to None.
        sequence_len: Length of token sequence.
        forward_returns_attn_mask: Whether forward should return attention mask.
        embed_dim_second: Whether the embed dimension should be second.

    .. note::
        If :attr:`kdim` and :attr:`vdim` are None, they will be set
        to :attr:`embed_dim` such that query, key, and value have the same
        number of features.

    Examples::

        mha = co.RetroactiveMultiheadAttention(
            embed_dim=512,
            num_heads=8,
            sequence_len=32,
            dropout=0.0,
            batch_first=True,
            embed_dim_second=True,
        )
        x = torch.rand(10, 512, 32)

        out, attn_mask = mha.forward(x)

        # continual inference API
        firsts = mha.forward_steps(x[:,:,:-1])
        last = mha.forward_step(x[:,:,-1])

        assert firsts is None  # The module first needs to observe ``sequence_len`` values
        assert torch.allclose(out, last, atol=1e-6)
    """

    _state_shape = 15
    _dynamic_state_inds = [True]*14 + [False]

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_landmarks,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        sequence_len=None,
        forward_returns_attn_mask=True,
        embed_dim_second=False,
        init_mem=True,
        batch_size=32,
        single_output_mode=False,
        single_output_forward=False,
        query_index=None,
        fixed_landmarks=False,
        compute_inverse=True,
        forward_mode="forward",
    ) -> None:
        assert single_output_mode >= single_output_forward # single_output_forward can only be True when single_output_mode is

        NystromMultiheadAttention.__init__(
            self,
            sequence_len,
            embed_dim,
            num_heads,
            num_landmarks,
            batch_size,
            dropout,
            device,
            dtype,
            forward_returns_attn_mask,
            single_output_forward,
            fixed_landmarks,
        )

        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.num_landmarks = num_landmarks
        self.embed_dim_second = embed_dim_second
        self.single_output_mode = single_output_mode
        self.single_output_forward = single_output_forward
        self.query_index = query_index
        self.compute_inverse = compute_inverse
        self.forward_mode = forward_mode

        if init_mem:
            torch.set_default_device(device=device)
            self.state = _scaled_dot_product_attention_default_state(batch_size, sequence_len, embed_dim, num_heads, num_landmarks)
            torch.set_default_device(device="cpu")

    def fix_landmarks(self, q_data, k_data=None, alg="kmeans", kmeans_attempts=10, seed=0):
        q_tilde, k_tilde, Gamma_D, Gamma_D_inv = (
            NystromMultiheadAttention.fix_landmarks(self, q_data, k_data, alg=alg, kmeans_attempts=kmeans_attempts, seed=seed))

        state = list(self.state)
        state[0] = q_tilde
        state[1] = k_tilde
        state[6] = Gamma_D
        state[12] = Gamma_D_inv
        self.state = tuple(state)

    def forward(self, query, key=None, value=None):
        match self.forward_mode:
            case "forward_steps":
                return self.forward_steps(query, key, value)
            case "forward_step":
                return self.forward_step(query, key, value)

        if not self.single_output_forward or not self.single_output_mode:
            return NystromMultiheadAttention.forward(
                self, query, key, value
            )

        if key is None:
            key = query
        if value is None:
            value = query

        o = NystromMultiheadAttention.forward(
            self, query, key, value, single_output_forward=True
        )

        return o

    def forward_step(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        update_state=True,
        *args,
        **kwargs,
    ) -> Optional[Tensor]:
        """
        Args:
            query, key, value: step_inputs for mapping a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.

        Shapes for inputs:
            - query: :math:`(N, E)` N is the batch size, E is the embedding dimension.
            - key: :math:`(N, E)`, where N is the batch size, E is the embedding dimension.
            - value: :math:`(N, E)` where N is the batch size, E is the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
              :math:`(N, E, L)` if ``batch_first`` and ``embed_dim_second ``True``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        query = torch.squeeze(self.prepare_input(query.unsqueeze(-1), self.W_q))
        key = torch.squeeze(self.prepare_input(key.unsqueeze(-1), self.W_k))
        value = torch.squeeze(self.prepare_input(value.unsqueeze(-1), self.W_v))

        o, new_state = _scaled_dot_product_attention_step(
            self.state, query, key, value,
            single_output=self.single_output_mode,
            fixed_landmarks=self.fixed_landmarks,
            compute_inverse=self.compute_inverse,
        )

        if update_state:
            self.state = new_state

        if isinstance(o, Tensor) and self.embed_dim_second:
            o = o.transpose(1, 2)

        return torch.flatten(o, 1, 2)

    def forward_steps(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        update_state=True,
        *args,
        **kwargs,
    ) -> Optional[Tensor]:
        bs, d, n = query.shape

        if bs != self.batch_size:
            torch.set_default_device(device=self.device)
            self.state = _scaled_dot_product_attention_default_state(bs, self.sequence_len, self.embed_dim, self.num_heads, self.num_landmarks)
            torch.set_default_device("cpu")
            self.batch_size = bs

        for i in range(query.size()[2]):
            query_step = query[:, :, i]
            if key:
                key_step = key[:, :, i]
            else:
                key_step = query[:, :, i]
            if value:
                value_step = value[:, :, i]
            else:
                value_step = query[:, :, i]

            o = self.forward_step(query_step, key_step, value_step, update_state, *args, **kwargs)

            # if isinstance(o, Tensor):
            #     outs.append(o)

        if self.single_output_forward:
            n = 1
        o = o.reshape((bs, n, d))
        o = self.ff(o).permute(0, 2, 1)

        return o

    def flops(self):
        d = self.embed_dim
        m = self.num_landmarks
        n = self.sequence_len

        ratio_updates = m/n
        ratio_fixed = 1 - ratio_updates

        if self.call_mode == "forward":
            return super().flops()
        elif self.single_output_mode:
            fixed_cost = 8*d*m + m**2 + 6*m
            if self.fixed_landmarks:
                f = fixed_cost
            else:
                continual_cost = n*d*m + 3*n*d + n + 14*d*m + 24*(m**3) + 23*(m**2) + 20*m
                f = (ratio_updates*continual_cost) + (ratio_fixed*fixed_cost)
        else:
            fixed_cost = n*d*m + 8*d*m + m**2 + 8*m
            if self.fixed_landmarks:
                f = fixed_cost
            else:
                continual_cost = n*d*m + 6*n*d + 9*n + n*(m**2) + n*m + 13*d*m + 24*(m**3) + 22*m**2 + 17*m
                f = (ratio_updates*continual_cost) + (ratio_fixed*fixed_cost)

        return f*self.num_head

import torch
import torch.nn as nn
import math

from abc import abstractmethod

from continual.module import CoModule
import continual as co
from continual import RecyclingPositionalEncoding
from continual.logging import getLogger
from continual.module import CoModule

from models import LearnedPositionalEncoding
#from continual_nystromformer import CoNystromAttention
from utils import iterative_inv

from typing import Any, Callable, List, Optional, Tuple, Union
from torch import Tensor

logger = getLogger(__name__)
logger_once = getLogger(__name__, log_once=True)


class NystromMultiheadAttention(CoModule):
    def __init__(
            self,
            sequence_len,
            embed_dim,
            num_heads,
            num_landmarks,
            dropout=0.0,
            device=None,  # TODO: Implement
            dtype=torch.float32,
            forward_returns_attn_mask=False

    ):
        super().__init__()

        self.sequence_len = sequence_len
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_landmarks = num_landmarks

        self.dropout = dropout

        self.W_q = nn.Linear(self.sequence_len, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.sequence_len, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.sequence_len, self.num_head * self.head_dim)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.sequence_len)

    def forward(self, input):
        query, key, value = input

        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        attn_out = _scaled_dot_product_attention(query, key, value, self.num_landmarks, dropout_p=self.dropout)
        output = self.ff(attn_out)

        return output

    def split_heads(self, X):
        return X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)



def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    m: int,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    use_conv: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention as in Nyströmformer on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        m: int. Number of landmarks used for the Nyström method. Default=10
        use_conv: Indicates whether to apply a convolution layer over the value input or not. Default=False

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """

    if attn_mask is not None:  # pragma: no cover
        logger_once.warning("attn_mask is not supported yet and will be skipped")
    if dropout_p != 0.0:  # pragma: no cover
        logger_once.warning("dropout_p is not supported yet and will be skipped")
    if use_conv:
        logger_once.warning("use_conv is not supported yet and will be skipped")

    B, Nt, E = q.shape

    assert Nt % m == 0, "The sequence length must be divisible bt the number of landmarks"

    q = torch.div(q, math.sqrt(math.sqrt(E)))
    k = torch.div(k, math.sqrt(math.sqrt(E)))

    if m >= Nt:
        # Apply base attention, as the number of samples is greater than the sequence length
        attn = torch.nn.functional.softmax(torch.bmm(q, k.transpose(-1, -2)), dim=-1)  # - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
        output = torch.bmm(attn, v)
    else:
        q_landmarks = q.reshape(-1, m, Nt // m, E).mean(dim=-2)
        k_landmarks = k.reshape(-1, m, Nt // m, E).mean(dim=-2)

        kernel_1 = torch.nn.functional.softmax(torch.bmm(q, k_landmarks.transpose(-1, -2)), dim=-1)
        kernel_2 = torch.nn.functional.softmax(torch.bmm(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1)
        kernel_3 = torch.nn.functional.softmax(torch.bmm(q_landmarks, k.transpose(-1, -2)), dim=-1)  # - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
        output = torch.bmm(torch.bmm(kernel_1, iterative_inv(kernel_2)), torch.bmm(kernel_3, v))

    return output, torch.empty()  # TODO: See whether is necessary to return the weights or not
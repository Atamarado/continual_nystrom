import torch
import torch.nn as nn
import math
import continual as co
from continual import RecyclingPositionalEncoding

from models import LearnedPositionalEncoding
#from continual_nystromformer import CoNystromAttention
from utils import iterative_inv

from continual.logging import getLogger
from typing import Any, Callable, List, Optional, Tuple, Union
from torch import Tensor

logger = getLogger(__name__)
logger_once = getLogger(__name__, log_once=True)

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    m: int = 10,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    use_conv: bool = False
) -> Tuple[Tensor, Tuple]: # TODO: Change back to [Tensor, Tensor]
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

    return output, (kernel_1, kernel_2, kernel_3)  # TODO: See whether is necessary to return the weights or not

def Nystromformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1,
    continual=False
):
    assert embedding_dim % num_heads == 0

    linear_encoding = co.Linear(input_dim, embedding_dim, channel_dim=1)

    if continual:
        position_encoding = RecyclingPositionalEncoding(
            embedding_dim,
            int(embedding_dim * 1.0),  # Change num pos enc to cycle between
            forward_update_index_steps=1,
        )
    else:
        position_encoding = LearnedPositionalEncoding(
            embedding_dim,
            embedding_dim,
            sequence_len
        )

    pe_dropout = nn.Dropout(p=dropout_rate)

    layers = []
    for _ in range(num_layers):
        encoder = NystromformerEncoder(
            embedding_dim,
            num_heads,
            attn_ff_hidden_dim,
            dropout_rate,
            sequence_len,
            continual=continual
        )
        layers.append(encoder)

    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    return co.Sequential(
        linear_encoding,
        position_encoding,
        pe_dropout,
        *layers,
        pre_head_ln,
        mlp_head,
    )


class NystromformerEncoder(nn.Module):

    def __init__(self, embed_dim, heads, mlp_dim, dropout_rate, sequence_len, activation=nn.GELU(), layer_norm_eps=1e-5,
                 single_output_forward=True, continual=False):
        super().__init__()
        attention_module = CoNystromAttention if continual else NystromAttention

        self.attention = attention_module(
            embed_dim=embed_dim,
            num_head=heads,
            num_landmarks=10,  # TODO: Specify number of landmarks
            seq_len=sequence_len,
            #conv_kernel_size=, TODO: Study it's implementation
        )

        self.delay = co.transformer.SelectOrDelay(delay=0) if single_output_forward else nn.Identity()

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.linear1 = co.Linear(embed_dim, mlp_dim, channel_dim=2)
        self.activation = activation
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = co.Linear(mlp_dim, embed_dim, channel_dim=2)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, X):
        X = self.attention(X)
        X = self.norm1(X)
        X = self.linear1(X)
        X = self.activation(X)
        X = self.dropout1(X)
        X = self.linear2(X)
        X = self.dropout2(X)
        X = self.norm2(X)

        X = torch.transpose(X, 2, 1)
        X = self.delay(X)
        return X

# From https://github.com/mlpen/Nystromformer/blob/56893131bf3fa99b5a2d3ab452a591dca722529a/reorganized_code/encoders/backbones/efficient_attentions/attention_nystrom.py#L6
class NystromAttention(nn.Module):
    def __init__(self, embed_dim, num_head, num_landmarks, seq_len, conv_kernel_size=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.num_landmarks = num_landmarks
        self.seq_len = seq_len

        self.use_conv = conv_kernel_size
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (conv_kernel_size, 1), padding = (conv_kernel_size // 2, 0),
                bias = False,
                groups = self.num_head)

        self.q_linear = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_head)])
        self.k_linear = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_head)])
        self.v_linear = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_head)])

    def forward_QKV(self, Q, K, V):#, mask):
        # Multi-head split and linear layer
        Q_head = torch.empty((self.num_head, Q.size()[0], self.seq_len, self.head_dim)).to(Q.device)
        K_head = torch.empty((self.num_head, K.size()[0], self.seq_len, self.head_dim)).to(K.device)
        V_head = torch.empty((self.num_head, V.size()[0], self.seq_len, self.head_dim)).to(V.device)

        for i in range(self.num_head):
            Q_head[i] = self.q_linear[i](Q)
            K_head[i] = self.k_linear[i](K)
            V_head[i] = self.v_linear[i](V)

        Q = torch.permute(Q_head, (1, 0, 2, 3))
        K = torch.permute(K_head, (1, 0, 2, 3))
        V = torch.permute(V_head, (1, 0, 2, 3))

        Q = Q / math.sqrt(math.sqrt(self.head_dim))  # * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K / math.sqrt(math.sqrt(self.head_dim))  # * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)), dim = -1)# - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim = -1)# - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        if self.use_conv:
            X += self.conv(V)# * mask[:, None, :, None])

        # Merge multi-heads
        X = torch.permute(X, (0, 2, 1, 3))
        X = torch.flatten(X, start_dim=2, end_dim=3)
        return X

    def forward(self, X, swap_axes=True):
        if swap_axes:
            X = torch.transpose(X, 1, 2)
        return self.forward_QKV(X, X, X)

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'

def ContinualNystromformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1
):
    return Nystromformer(
        sequence_len,
        input_dim,
        embedding_dim,
        attn_ff_hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        dropout_rate=dropout_rate,
        continual=True
    )
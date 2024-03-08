import torch
import torch.nn as nn
import math
import continual as co

from models import LearnedPositionalEncoding

def Nystromformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1,
):
    assert embedding_dim % num_heads == 0

    linear_encoding = co.Linear(input_dim, embedding_dim, channel_dim=1)
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
        )
        layers.append(encoder)

    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    return nn.Sequential(
        linear_encoding,
        position_encoding,
        pe_dropout,
        *layers,
        pre_head_ln,
        mlp_head,
    )


class NystromformerEncoder(nn.Module):

    def __init__(self, embed_dim, heads, mlp_dim, dropout_rate, sequence_len, activation=nn.GELU(), layer_norm_eps=1e-5, single_output_forward=True):
        super().__init__()

        self.attention = NystromAttention(
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
            K_head[i] = self.q_linear[i](K)
            V_head[i] = self.q_linear[i](V)

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
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

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

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim = -2)) * torch.max(torch.sum(torch.abs(K), dim = -1))) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'
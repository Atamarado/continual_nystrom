import torch
import torch.nn as nn
import math
import gc

from torch.nn.modules.activation import MultiheadAttention
from sklearn.cluster import KMeans

from abc import abstractmethod

from continual.module import CoModule
import continual as co
from continual import RecyclingPositionalEncoding
from continual.logging import getLogger
from continual.module import CoModule

from .utils import iterative_inv

from typing import Any, Callable, List, Optional, Tuple, Union
from torch import Tensor

logger = getLogger(__name__)
logger_once = getLogger(__name__, log_once=True)


class NystromMultiheadAttention(CoModule, MultiheadAttention):
    def __init__(
            self,
            sequence_len,
            embed_dim,
            num_heads,
            num_landmarks,
            batch_size,
            dropout=0.0,
            device=None,  # TODO: Implement
            dtype=torch.float32,
            forward_returns_attn_mask=False,
            single_output_forward=False,
            fixed_landmarks=False,
            compute_inverse=True
    ):
        super().__init__(embed_dim, num_heads)

        self.device = device
        self.sequence_len = sequence_len
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_landmarks = num_landmarks
        self.batch_size = batch_size
        self.single_output_forward = single_output_forward

        self.dropout = dropout

        self.W_q = [nn.Linear(self.embed_dim, self.head_dim).to(device) for _ in range(self.num_head)]
        self.W_k = [nn.Linear(self.embed_dim, self.head_dim).to(device) for _ in range(self.num_head)]
        self.W_v = [nn.Linear(self.embed_dim, self.head_dim).to(device) for _ in range(self.num_head)]

        self.ff = nn.Linear(self.embed_dim, self.embed_dim)

        self.fixed_landmarks = fixed_landmarks
        self.compute_inverse = compute_inverse

    def fix_landmarks(self, q_data, k_data=None, alg="base", kmeans_attempts=3, seed=0, num_points=0):
        self.fixed_landmarks = True
        if k_data is None:
            k_data = q_data

        q_tilde = []
        k_tilde = []
        if alg == "kmeans":
            for i in range(self.num_head):
                # Clear caches, as this operation may require the storage of big amounts of data
                torch.cuda.empty_cache()
                gc.collect()

                q_head_data = []
                k_head_data = []
                for j in range(0, q_data.size()[0], self.batch_size):
                    q_head_data.append(self.prepare_input(q_data[j:min(j+self.batch_size, q_data.size()[0])], self.W_q, index=i).flatten(0, 1).detach().cpu())
                    k_head_data.append(self.prepare_input(k_data[j:min(j+self.batch_size, q_data.size()[0])], self.W_q, index=i).flatten(0, 1).detach().cpu())

                q_head_data = torch.cat(q_head_data, dim=0)
                k_head_data = torch.cat(k_head_data, dim=0)

                if num_points > 0:
                    index = torch.randperm(q_head_data.size()[0], generator=torch.Generator().manual_seed(seed))[:num_points]
                    q_head_data = q_head_data[index]
                    index = torch.randperm(q_head_data.size()[0], generator=torch.Generator().manual_seed(seed+1))[:num_points]
                    k_head_data = k_head_data[index]

                q_clusters = KMeans(n_clusters=self.num_landmarks,
                                    n_init=kmeans_attempts,
                                    random_state=seed).fit(q_head_data).cluster_centers_
                k_clusters = KMeans(n_clusters=self.num_landmarks,
                                    n_init=kmeans_attempts,
                                    random_state=seed).fit(k_head_data).cluster_centers_

                q_tilde.append(torch.tensor(q_clusters))
                k_tilde.append(torch.tensor(k_clusters))
        else:
            raise NotImplementedError("Only kmeans is implemented")

        q_tilde = torch.stack(q_tilde, dim=0).type(torch.float32).to(self.device)
        k_tilde = torch.stack(k_tilde, dim=0).type(torch.float32).to(self.device)

        # Repeat for all samples in the batch size
        q_tilde = q_tilde.repeat(self.batch_size, 1, 1)
        k_tilde = k_tilde.repeat(self.batch_size, 1, 1)

        Gamma_D = torch.nn.functional.softmax(torch.bmm(q_tilde, k_tilde.transpose(-1, -2)), dim=-1)
        Gamma_D_inv = iterative_inv(Gamma_D)

        self.q_tilde = q_tilde
        self.k_tilde = k_tilde
        self.kernel_2 = Gamma_D_inv

        return q_tilde, k_tilde, Gamma_D, Gamma_D_inv

    def prepare_input(self, query, linear, index=None, join_nhead=True):
        query = query.to(self.device)
        # (bs, d, n) -> (bs, n, d)
        query = query.permute(0, 2, 1)
        return query
        # if index is None:
        #     lin_queries = []
        #     for w_q in linear:
        #         lin_queries.append(w_q(query))
        #     # (bs, num_head, n, head_dim)
        #     lin_queries = torch.stack(lin_queries, dim=1).to(query.device)
        #
        #     # (bs, num_head, n, head_dim) -> (bs*num_head, n, head_dim)
        #     if join_nhead:
        #         lin_queries = lin_queries.flatten(0, 1)
        #
        #     return lin_queries
        # else:
        #     return linear[index](query)

    def forward(self, query, key=None, value=None, single_output_forward=False):
        bs, d, n = query.shape

        if bs > self.batch_size:
            output = []
            for index in range(0, bs, self.batch_size):
                start = index
                end = min(index+self.batch_size, bs)
                output.append(self.forward(query[start:end],
                             key=key[start:end] if key is not None else None,
                             value=value[start:end] if value is not None else None))
            return torch.cat(output, dim=0)

        if key is None:
            key = query
        if value is None:
            value = query

        query = self.prepare_input(query, self.W_q)
        key = self.prepare_input(key, self.W_k)
        value = self.prepare_input(value, self.W_v)

        if self.fixed_landmarks:
            attn_out = _scaled_dot_product_attention(query, key, value,
                                                     self.num_landmarks,
                                                     dropout_p=self.dropout,
                                                     single_output_forward=single_output_forward,
                                                     q_landmarks=self.q_tilde,
                                                     k_landmarks=self.k_tilde,
                                                     kernel_2=self.kernel_2,
                                                     compute_inverse=self.compute_inverse,
                                                     )
        else:
            attn_out = _scaled_dot_product_attention(query, key, value,
                                                     self.num_landmarks,
                                                     dropout_p=self.dropout,
                                                     single_output_forward=single_output_forward,
                                                     compute_inverse=self.compute_inverse,
                                                     )

        if single_output_forward:
            n = 1
        output = attn_out.reshape((bs, n, d)).permute(0, 2, 1)
        return output

    def split_heads(self, X):
        return X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)

    def flops(self):
        d = self.embed_dim
        m = self.num_landmarks
        n = self.sequence_len

        f = 4*n*d*m + 2*n*d + n*(m**2) + 2*n*m + d*(m**2) + 24*(m**3) + 23*(m**2)
        return f*self.num_head


def get_landmarks(matrix, m):
    B, n, E = matrix.shape

    mod = n % m
    if mod == 0:
        # Basic case, all landmarks contain the same number of points
        return matrix.reshape(-1, m, n // m, E).mean(dim=-2)

    # The first n%m landmarks contain one extra point
    min_points_per_landmark = n // m

    landmark_split_point = (min_points_per_landmark + 1)*mod
    num_landmarks_first = mod
    num_landmarks_last = m - mod

    first_landmarks = matrix[:, :landmark_split_point].reshape(B, num_landmarks_first, landmark_split_point // num_landmarks_first, E).mean(dim=-2)
    last_landmarks = matrix[:, landmark_split_point:].reshape(B, num_landmarks_last, (n - landmark_split_point) // num_landmarks_last, E).mean(dim=-2)

    return torch.cat(
        (
            first_landmarks,
            last_landmarks
        ),
        dim=1
    )



def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    m: int,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    use_conv: bool = False,
    return_kernels=False,
    single_output_forward=False,
    q_landmarks=None,
    k_landmarks=None,
    kernel_2=None,
    compute_inverse: bool = True,
) -> Tuple[Tensor]:
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
        kernel_2: Pre-computed landmark matrix from q_landmarks, k_landmarks. Default=None
        q_landmarks, k_landmarks: Landmarks to use for the Nyström approximation. If the landmarks are not provided,
            get_landmarks method is summoned to compute them. Default=None

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - q_landmarks, k_landmarks: :math:`(B, m, E)` where m is the number of landmarks
        - kernel_2: :math:`(B, m, m)`

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
        if q_landmarks is None:
            q_landmarks = get_landmarks(q, m)
        else:
            q_landmarks = q_landmarks[:B]
        if k_landmarks is None:
            k_landmarks = get_landmarks(k, m)
        else:
            k_landmarks = k_landmarks[:B]
        if kernel_2 is None:
            kernel_2 = torch.nn.functional.softmax(torch.bmm(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1)
        else:
            kernel_2 = kernel_2[:B]

        if single_output_forward:
            q = q[:, -1:]

        kernel_1 = torch.nn.functional.softmax(torch.bmm(q, k_landmarks.transpose(-1, -2)), dim=-1)
        kernel_3 = torch.nn.functional.softmax(torch.bmm(q_landmarks, k.transpose(-1, -2)), dim=-1)  # - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
        output = torch.bmm(torch.bmm(kernel_1, kernel_2), torch.bmm(kernel_3, v))

        if return_kernels:
            return output, kernel_1, kernel_2, kernel_3
    return output

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        position_embeddings = torch.permute(position_embeddings, (0, 2, 1))
        return x + position_embeddings
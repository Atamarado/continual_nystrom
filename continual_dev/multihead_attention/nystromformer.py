import torch
import torch.nn as nn
import math
import gc
import random

from sklearn.cluster import KMeans

from ..logging import getLogger

from .nystrom_utils import iterative_inv

from typing import Optional, Tuple
from torch import Tensor

logger = getLogger(__name__)
logger_once = getLogger(__name__, log_once=True)

def split_number_in_list(number_to_split, elements_in_list):
    min_element_in_list = number_to_split//elements_in_list

    fraction_extra_elements = number_to_split/elements_in_list - min_element_in_list
    n_extra_elements = math.floor(fraction_extra_elements*elements_in_list)

    split_list_1 = [min_element_in_list]*(elements_in_list - n_extra_elements)
    split_list_2 = [min_element_in_list+1]*n_extra_elements

    split_list = split_list_1 + split_list_2
    random.shuffle(split_list)
    return split_list

class NystromMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,

        # NystromMultiheadAttention additional parameters
        sequence_len=None,
        num_landmarks=10,
        batch_size=1,
        single_output_forward=False,
        fixed_landmarks=False,
        compute_inverse=True
    ):
        nn.MultiheadAttention.__init__(
            self,
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )

        self.device = device
        self.sequence_len = sequence_len

        self.num_landmarks = num_landmarks
        self.batch_size = batch_size
        self.single_output_forward = single_output_forward

        self.W_q = nn.Linear(self.embed_dim, self.head_dim*self.num_heads).to(device)
        self.W_k = nn.Linear(self.embed_dim, self.head_dim*self.num_heads).to(device)
        self.W_v = nn.Linear(self.embed_dim, self.head_dim*self.num_heads).to(device)

        self.fixed_landmarks = fixed_landmarks
        self.compute_inverse = compute_inverse

    def _fix_landmarks(self, q_data, kmeans_attempts, num_points, seed=0):
        q_points = []
        q_tilde = []

        for i in range(0, q_data.size()[0], self.batch_size):
            q_batch = self.prepare_input(q_data[i:min(i + self.batch_size, q_data.size()[0])], self.W_q, join_nhead=False).flatten(0, 1)
            # q_batch have shape # (bs*n, num_heads, head_dim)
            q_points.append(q_batch)

        q_points = torch.cat(q_points, dim=0) # (points, num_heads, head_dim)

        # We select randomly the number of points
        if num_points > 0:
            index = torch.randperm(q_points.size()[0], generator=torch.Generator().manual_seed(seed))[:num_points]
            q_points = q_points[index]

        # Perform clustering for every head
        for i in range(self.num_heads):
            q_head_data = q_points[:, i].detach().cpu()

            # Clear caches, as the following operations are memory heavy
            torch.cuda.empty_cache()
            gc.collect()

            q_clusters = KMeans(n_clusters=self.num_landmarks,
                                n_init=kmeans_attempts,
                                random_state=seed).fit(q_head_data).cluster_centers_

            q_tilde.append(torch.tensor(q_clusters))

        del q_points

        q_tilde = torch.stack(q_tilde, dim=0).type(torch.float32).to(self.device)
        q_tilde = q_tilde.repeat(self.batch_size, 1, 1)
        return q_tilde

    def fix_landmarks(self, q_data, k_data=None, alg="kmeans", kmeans_attempts=3, seed=0, num_points=0):
        if alg != "kmeans":
            raise NotImplementedError("Only kmeans is implemented")

        self.fixed_landmarks = True

        q_tilde = self._fix_landmarks(q_data, kmeans_attempts, num_points, seed)

        if k_data is None:
            k_data = q_data
        k_tilde = self._fix_landmarks(k_data, kmeans_attempts, num_points, seed)

        Gamma_D = torch.nn.functional.softmax(torch.bmm(q_tilde, k_tilde.transpose(-1, -2)), dim=-1)
        Gamma_D_inv = iterative_inv(Gamma_D)

        self.q_tilde = q_tilde.detach()
        self.k_tilde = k_tilde.detach()
        self.kernel_2 = Gamma_D_inv.detach()

        self.q_tilde.requires_grad = False
        self.k_tilde.requires_grad = False
        self.kernel_2.requires_grad = False

        return q_tilde, k_tilde, Gamma_D, Gamma_D_inv

    def prepare_input(self, query, linear, join_nhead=True):
        query = query.to(self.device)

        if query.ndim < 3:
            query = query.unsqueeze(-1)

        # (bs, d, n) -> (bs, n, d)
        query = query.permute(0, 2, 1)

        query = linear(query)
        query = query.unflatten(-1, (self.num_heads, -1)) # (bs, n, num_heads, head_dim)

        if join_nhead:
            query = query.permute(0, 2, 1, 3)  # (bs, n, num_heads, head_dim) -> (bs, num_heads, n, head_dim)
            query = query.flatten(0, 1) # (bs, num_heads, n, head_dim) -> (bs*num_heads, n, head_dim)

        return query

    def forward(self, query, key=None, value=None, single_output_forward=False):
        bs, d, n = query.shape

        if bs > self.batch_size:
            output = []
            for index in range(0, bs, self.batch_size):
                start = index
                end = min(index+self.batch_size, bs)
                output.append(self.forward(query[start:end],
                             key=None if key is None else key[start:end],
                             value=None if value is None else value[start:end]))
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
        return X.reshape(X.size(0), X.size(1), self.num_heads, self.head_dim)

    def flops(self, call_mode=None):
        d = self.embed_dim
        m = self.num_landmarks
        n = self.sequence_len

        if self.fixed_landmarks:
            f = 4*n*d*m + n*(m**2) + 2*n*m + n + m
        else:
            f = 2*n*d + n + n*(m**2) + 2*n*m + d*(m**2) + 24*(m**3) + 22*(m**2) + 2*m
        return f*self.num_heads

    def mem_costs(self, call_mode=None):
        d = self.embed_dim
        m = self.num_landmarks
        n = self.sequence_len

        if self.fixed_landmarks:
            valley_cost = 3*(n*d-1) + 2*d*m + m**2
            peak_cost = 4*n*d + 2*n*m + 2*d*m + 2*(m**2) + 1
        else:
            valley_cost = 3*(n*d-1)
            peak_cost = 4*(n*d) + 2*n*m + 2*d*m + 1 + 6*(m**2) + m

        return valley_cost*self.num_heads, peak_cost*self.num_heads


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
            if compute_inverse:
                kernel_2 = iterative_inv(kernel_2)
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
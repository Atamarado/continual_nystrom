import torch.nn as nn
import torch
import math

from utils import continual_matrix_concat
from nystromformer import Nystromformer

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

class CoNystromAttention(nn.Module):
    def __init__(self, embed_dim, num_head, num_landmarks, seq_len, conv_kernel_size=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.num_landmarks = num_landmarks
        self.seq_len = seq_len

        self.tokens_per_landmark = self.seq_len // self.num_landmarks == self.seq_len

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

        self.d_q = math.sqrt(self.head_dim)

    def multi_head(self, Q, K, V):
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

        return Q, K, V

    # Generic qk_product
    def qk_product(self, q, k,):
        return torch.exp(torch.matmul(q, torch.transpose(k, 2, 3)) / self.d_q)

    def forward_QKV(self, Q, K, V):
        Q, K, V = self.multi_head(Q, K, V)

        # Q|K|V shape = (batch_size, num_head, num_tokens, head_size)

        # Local variables used to match with formulation
        m = self.num_landmarks
        n = self.seq_len

        # seq_len has to be bigger than the number of tokens
        if Q.size(dim=2) < self.seq_len:
            # TODO: Add solution
            return

        # --------------------------------------------------------------------------------------------------------------
        # Expansion phase (first self.seq_len tokens)

        # Select first set of landmarks
        Q_first = Q[:, :, :self.seq_len]
        K_first = K[:, :, :self.seq_len]
        V_first = V[:, :, :self.seq_len]

        self.Q_tilde = Q_first.reshape(-1, self.num_head, self.seq_len // self.num_landmarks, self.head_dim).mean(dim=-2)
        self.K_tilde = K_first.reshape(-1, self.num_head, self.seq_len // self.num_landmarks, self.head_dim).mean(dim=-2)

        self.Beta_mem = self.qk_product(Q_first, self.K_tilde)  # Note that the first row will be old at the next iteration
        self.Gamma_mem = self.qk_product(self.Q_tilde, self.K_tilde)
        self.Delta_mem = self.qk_product(self.Q_tilde, self.K_tilde)

        # The first set of diagonals are computed in the same way as in the original paper
        self.d_Beta_mem = torch.matmul(self.Beta_mem, torch.ones((m, 1)))
        self.d_Gamma_mem = torch.matmul(self.Gamma_mem, torch.ones((m, 1)))
        self.d_Delta_mem = torch.matmul(self.Delta_mem, torch.ones((n, 1)))

        self.Beta_D_mem = self.Beta_mem / self.d_Beta_mem
        self.Gamma_D_mem = iterative_inv(self.Gamma_mem / self.d_Gamma_mem)  # TODO: Improved formulation coming
        self.Delta_D_mem = self.Delta_mem / self.d_Delta_mem

        self.Beta_Gamma_mem = torch.matmul(self.Beta_D_mem, self.Gamma_D_mem)
        self.Delta_V = torch.matmul(self.Delta_D_mem, V_first)

        attention = [torch.matmul(self.Beta_Gamma, self.Delta_V)]

        # --------------------------------------------------------------------------------------------------------------
        # Continual phase: from seq_len to the end of the input
        for i in range(self.seq_len, Q.size(dim=2)):
            q_new = Q[:, :, i]
            k_new = K[:, :, i]  # TODO: Can we transpose all K directly?

            if i % self.tokens_per_landmark == self.tokens_per_landmark - 1:
                # ------------------------------------------------------------------------------------------------------
                # We need to update the landmarks
                # New landmarks
                q_tilde_new = Q[:, :, i-self.seq_len:i].mean(dim=-2)
                k_tilde_new = K[:, :, i-self.seq_len:i].mean(dim=-2)

                q_tilde_old = Q[:, :, 0]
                k_tilde_old = K[:, :, 0]

                Q_mem = Q[:, :, i-self.seq_len+1: i]
                K_mem = K[:, :, i-self.seq_len+1: i]

                Q_tilde_mem = self.Q_tilde[:, :, 1]
                K_tilde_mem = self.K_tilde[:, :, 1]

                # Beta update
                Beta_B = self.qk_product(Q_mem, k_tilde_new)
                Beta_C = self.qk_product(q_new, K_tilde_mem)
                Beta_D = self.qk_product(q_new, k_tilde_new).squeeze()
                self.Beta_mem = continual_matrix_concat(self.Beta_mem, Beta_B, Beta_C, Beta_D)

                # Gamma update
                Gamma_B = self.qk_product(Q_tilde_mem, k_tilde_new)
                Gamma_C = self.qk_product(q_tilde_new, K_tilde_mem)
                Gamma_D = self.qk_product(q_tilde_new, k_tilde_new).squeeze()
                self.Gamma_mem = continual_matrix_concat(self.Gamma_mem, Gamma_B, Gamma_C, Gamma_D)

                # Delta update
                Delta_B = self.qk_product(Q_tilde_mem, k_new)
                Delta_C = self.qk_product(q_tilde_new, K_mem)
                Delta_D = self.qk_product(q_tilde_new, k_new).squeeze()
                self.Delta_mem = continual_matrix_concat(self.Delta_mem, Delta_B, Delta_C, Delta_D)

                # d_Beta update
                d_Beta_mem = self.d_Beta_mem[:, :, 1:] - self.qk_product(Q_mem, k_tilde_old) + self.qk_product(Q_mem, k_tilde_new)
                d_Beta_new = torch.cat((
                    self.qk_product(q_new, K_tilde_mem),
                    self.qk_product(q_new, k_tilde_new).squeeze()
                    ),
                    dim=2
                )
                d_Beta_new = torch.matmul(d_Beta_new, torch.ones((m, 1)))
                self.d_Beta_mem = torch.cat((
                    d_Beta_mem,
                    d_Beta_new
                    ),
                    dim=2
                )

                # Next: d_Gamma update
                d_Gamma_mem = self.d_Gamma_mem[:, :, 1:] - self.qk_product(Q_tilde_mem, k_tilde_old) + self.qk_product(Q_tilde_mem, k_tilde_new)
                d_Gamma_new = torch.cat((
                    self.qk_product(q_tilde_new, K_tilde_mem),
                    self.qk_product(q_tilde_new, k_tilde_new).squeeze()
                    ),
                    dim=2
                )
                d_Gamma_new = torch.matmul(d_Gamma_new, torch.ones((m, 1)))
                self.d_Gamma_mem = torch.cat((
                    d_Gamma_mem,
                    d_Gamma_new
                    ),
                    dim=2
                )

                # Next: d_Delta update
                d_Delta_mem = self.d_Delta_mem[:, :, 1:] - self.qk_product(Q_tilde_mem, k_old) + self.qk_product(Q_tilde_mem, k_new)
                d_Delta_new = torch.cat((
                    self.qk_product(q_tilde_new, K_mem),
                    self.qk_product(q_tilde_new, k_tilde_new).squeeze()
                    ),
                    dim=2
                )
                d_Delta_new = torch.matmul(d_Delta_new, torch.ones((n, 1)))
                self.d_Delta_mem = torch.cat((
                    d_Delta_mem,
                    d_Delta_new
                    ),
                    dim=2
                )

                # Vector-matrix multipliations
                self.Beta_D_mem = self.Beta_mem / self.d_Beta_mem
                self.Gamma_D_new = iterative_inv(self.Gamma_mem / self.d_Gamma_mem)
                self.Delta_D_mem = self.Delta_mem / self.d_Delta_mem

                # Matrix multiplications
                self.Beta_Gamma_mem = torch.matmul(self.Beta_D_mem, self.Gamma_D_mem)
            else:
                # ------------------------------------------------------------------------------------------------------
                # Fixed landmarks
                # Beta^D * Gamma^D computation
                Beta_new = self.qk_product(q_new, self.K_tilde)
                d_Beta_new = torch.matmul(Beta_new, torch.ones((m, 1))).squeeze()
                Beta_D_new = Beta_new/d_Beta_new

                self.Beta_Gamma_mem = torch.cat((
                    self.Beta_Gamma_mem[:, :, 1:],
                    torch.matmul(Beta_D_new, self.Gamma_D_mem)
                    ),
                    dim=2
                )

                # Update Beta_mem and d_Beta_mem
                self.Beta_nem = torch.cat((
                    self.Beta_mem[:, :, 1:],
                    Beta_new
                    ),
                    dim=2
                )

                self.d_Beta_mem = torch.cat((
                    self.d_Beta_mem[:, :, 1:],
                    d_Beta_new
                    ),
                    dim=2
                )


                # Delta computation
                Delta_new = self.qk_product(self.Q_tilde, k_new)
                self.Delta_mem = torch.cat((
                    self.Delta_mem[:, :, :, 1:],
                    Delta_new
                    ),
                    dim=3
                )

                # self.d_Delta_mem
                k_old = K[:, :, i-self.seq_len]
                d_Delta_old = self.qk_product(self.Q_tilde, k_old)  # TODO: We can cache this operation
                d_Delta_new = Delta_new  # Same value

                self.d_Delta_mem = self.d_Delta_mem - d_Delta_old + d_Delta_new

                # Delta^D odot
                self.Delta_D_mem = self.Delta_mem/self.d_Delta_mem

            # ----------------------------------------------------------------------------------------------------------
            # Finish last computations

            # Delta^D * V multiplication
            self.Delta_V = torch.matmul(self.Delta_D_mem, V[:, :, i-self.seq_len+1:i+1])

            # Append new attention
            attention.append(torch.matmul(self.Beta_Gamma_mem, self.Delta_V))

        return torch.cat(attention).to(attention[0].device)


    def forward(self, X, swap_axes=True):
        if swap_axes:
            X = torch.transpose(X, 1, 2)
        return self.forward_QKV(X, X, X)


def iterative_inv(mat, n_iter=6):
    I = torch.eye(mat.size(-1), device=mat.device)
    K = mat
    V = 1 / (torch.max(torch.sum(torch.abs(K), dim=-2)) * torch.max(torch.sum(torch.abs(K), dim=-1))) * K.transpose(-1,
                                                                                                                    -2)
    for _ in range(n_iter):
        KV = torch.matmul(K, V)
        V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
    return V
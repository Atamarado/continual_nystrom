import torch
import math

from continual_nystromformer import _scaled_dot_product_attention_step, _scaled_dot_product_attention_default_state
from nystromformer import _scaled_dot_product_attention

def compute_diff(pred, target):
    diff = torch.abs(pred - target)
    print("Mean: "+str(torch.mean(diff)))
    print("Max: "+str(torch.max(diff)))

def test_scaled_dot_product_attention_step():
    N = 1000  # sequence length
    E = 5  # embedding dimension
    B = 2  # batch size
    H = 1  # num heads
    m = 10

    query1 = torch.randn((B, N, E))
    key1 = torch.randn((B, N, E))
    value1 = torch.randn((B, N, E))

    # # Manually compute first output
    # q = query1 / math.sqrt(E)
    # # (B, N, E) x (B, E, N) -> (B, N, N)
    # attn_exp = torch.exp(torch.bmm(q, key1.transpose(-2, -1)))
    # attn_sum = attn_exp.sum(dim=-1)  # over key dim
    #
    # # (B, N, N) x (B, N, E) -> (B, N, E)
    # av = torch.bmm(attn_exp, value1)
    # output1 = av / attn_sum.unsqueeze(-1)
    #
    # # Sanity check
    # target1, _ = _scaled_dot_product_attention(query1, key1, value1)
    # assert torch.allclose(target1, output1, atol=1e-6)

    # == Ready for actual test! ==

    # # Shift query, key and value by a time-step
    # query_step = torch.randn((B, E))
    # key_step = torch.randn((B, E))
    # value_step = torch.randn((B, E))
    #
    # query2 = torch.cat((query1[:, 1:], query_step.unsqueeze(1)), dim=1)
    # key2 = torch.cat((key1[:, 1:], key_step.unsqueeze(1)), dim=1)
    # value2 = torch.cat((value1[:, 1:], value_step.unsqueeze(1)), dim=1)
    #
    # # Manually compute first output
    # q = query2 / math.sqrt(E)
    # # (B, N, E) x (B, E, N) -> (B, N, N)
    # # attn_exp2 = torch.exp(torch.bmm(q, key2.transpose(-2, -1)))
    # # av2 = torch.bmm(attn_exp2, value2)
    #
    # target2, _ = _scaled_dot_product_attention(query2, key2, value2)
    #
    # prev_state = (
    #     # attn_sum[:, 1:],
    #     attn_sum[:, 1:].unsqueeze(-1),
    #     av[:, 1:],
    #     # query1 / math.sqrt(E),
    #     query1[:, 1:] / math.sqrt(E),
    #     key1.transpose(-2, -1),
    #     value1,
    #     # 0,
    #     # 0,
    #     # 0,
    # )
    # output2, new_state = _scaled_dot_product_attention_step(
    #     prev_state, query_step, key_step, value_step
    # )
    #
    # for p, n in zip(prev_state, new_state):
    #     assert p.shape == n.shape
    #
    # assert torch.allclose(target2, output2, atol=1e-6)

    target1, kernels = _scaled_dot_product_attention(query1, key1, value1)

    # Now, let's try from zero-init
    state = _scaled_dot_product_attention_default_state(B, N, E, H, m)
    for i in range(N):
        output_step, state, continual_kernels = _scaled_dot_product_attention_step(
            state, query1[:, i], key1[:, i], value1[:, i], last_iter=N
        )

    kernel1, kernel2, kernel3 = kernels
    Beta, Gamma, Delta = continual_kernels

    print("\nDifference Beta (last token): ")
    compute_diff(kernel1[:, N-1], Beta)
    print("\nDifference Gamma: ")
    compute_diff(kernel2, Gamma)
    print("\nDifference Delta: ")
    compute_diff(kernel3, Delta)
    print("\nDifference outputs: ")
    compute_diff(output_step, target1)

    #assert torch.allclose(output_step, target1, atol=1e-6)

    # output_step, state = _scaled_dot_product_attention_step(
    #     state, query_step, key_step, value_step
    # )
    # assert torch.allclose(output_step, target2, atol=1e-6)

if __name__ == '__main__':
    test_scaled_dot_product_attention_step()
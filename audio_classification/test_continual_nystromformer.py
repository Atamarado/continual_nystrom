import torch
import math

from nystromformer.continual_nystromformer import (
    _scaled_dot_product_attention_step,
    _scaled_dot_product_attention_default_state,
    State
)
from nystromformer.nystromformer import _scaled_dot_product_attention, get_landmarks
from nystromformer.utils import qk_product, iterative_inv, odot

def compute_diff(pred, target, mode="l2"):
    assert mode in ["l1", "l2"]
    if mode == "l2":
        diff = torch.sqrt(torch.sum(torch.pow(torch.subtract(pred, target), 2), dim=0))
    else:  # abs
        diff = torch.abs(pred - target)
    print("Mean: "+str(torch.mean(diff)))
    print("Max: "+str(torch.max(torch.abs(torch.subtract(pred, target)))))

def nystromformer_exp(q, k, v, m, stable_exp=False, state_mode=False):
    device = q.device
    B, Nt, E = q.shape

    q = torch.div(q, math.sqrt(math.sqrt(E)))
    k = torch.div(k, math.sqrt(math.sqrt(E)))

    Q_tilde = get_landmarks(q, m)
    K_tilde = get_landmarks(k, m)

    Beta = qk_product(q, K_tilde, stable_exp=stable_exp)
    d_Beta = torch.bmm(Beta, torch.ones((B, m, 1), device=device))
    Beta_D = odot(d_Beta, Beta)

    Gamma = qk_product(Q_tilde, K_tilde, stable_exp=stable_exp)
    d_Gamma = torch.bmm(Gamma, torch.ones((B, m, 1), device=device))
    Gamma_D = odot(d_Gamma, Gamma)
    Gamma_D_inv = iterative_inv(Gamma_D)

    BetaD_GammaD = torch.bmm(Beta_D, Gamma_D_inv)

    Delta = qk_product(Q_tilde, k, stable_exp=stable_exp)
    d_Delta = torch.bmm(Delta, torch.ones((B, Nt, 1), device=device))

    DeltaV = torch.bmm(Delta, v)

    if state_mode:
        state = (
            Q_tilde,
            K_tilde,

            q,
            k,
            v,

            BetaD_GammaD[:, 1:],
            Gamma_D_inv,
            d_Delta,
            DeltaV,

            d_Beta[:, 1:],
            d_Gamma[:, 1:],
            Beta[:, 1:, 1:],
            Gamma[:, 1:, 1:],

            torch.zeros((B, 1, E), device=device),
            torch.zeros((B, 1, E), device=device),

            torch.empty(1),
            0
        )
        return state

    else:
        Delta_D_V = odot(d_Delta, DeltaV)
        Delta_D = odot(d_Delta, Delta)

        output = torch.bmm(BetaD_GammaD, Delta_D_V)

        return output, Beta_D, Gamma_D_inv, Delta_D

def compute_landmarks(state: State, q, k, m):
    device = q.device

    (
        _, # Q_tilde
        _, # K_tilde

        _,
        _,
        V,

        BetaD_GammaD_mem,
        _, # Gamma_D
        d_Delta_prev,
        DeltaV_prev,

        d_Beta_prev,
        _, # d_Gamma_prev
        Beta_mem,
        _, # Gamma_mem

        q_tilde_new,
        k_tilde_new,

        state_index,
        iteration
    ) = state

    B, Nt, E = q.shape
    q = torch.div(q, math.sqrt(math.sqrt(E)))
    k = torch.div(k, math.sqrt(math.sqrt(E)))

    Q_tilde = get_landmarks(q, m)
    K_tilde = get_landmarks(k, m)

    Gamma = qk_product(Q_tilde, K_tilde)
    d_Gamma = torch.bmm(Gamma, torch.ones((B, m, 1), device=device))
    Gamma_D = odot(d_Gamma, Gamma)
    Gamma_D_inv = iterative_inv(Gamma_D)

    state = (
        Q_tilde,
        K_tilde,

        q,
        k,
        V,

        BetaD_GammaD_mem,
        Gamma_D_inv,
        d_Delta_prev,
        DeltaV_prev,

        d_Beta_prev,
        d_Gamma[:, 1:],
        Beta_mem,
        Gamma[:, 1:, 1:],

        q_tilde_new,
        k_tilde_new,

        state_index,
        iteration
    )

    return state

def compare_results(res1, res2):
    out1, Beta1, Gamma1, Delta1 = res1
    out2, Beta2, Gamma2, Delta2 = res2

    print("Difference Beta: ")
    compute_diff(Beta1, Beta2)
    print("Difference Gamma: ")
    compute_diff(Gamma1, Gamma2)
    print("Difference Delta: ")
    compute_diff(Delta1, Delta2)
    print("Difference outputs: ")
    compute_diff(out1, out2)



def test_stable_exp(seed=0):
    N = 1000  # sequence length
    E = 5  # embedding dimension
    B = 2  # batch size
    H = 1  # num heads
    m = 10

    for std in range(1, 20):
        g = torch.Generator()
        g.manual_seed(seed)

        query = torch.empty((B, N, E)).normal_(mean=0, std=std, generator=g)
        key = torch.empty((B, N, E)).normal_(mean=0, std=std, generator=g)
        value = torch.empty((B, N, E)).normal_(mean=0, std=std, generator=g)

        target = _scaled_dot_product_attention(query, key, value, m, return_kernels=True)

        print("\n\n")

        pred = nystromformer_exp(query, key, value, m)
        print("\nBase nystromformer exp. std="+str(std))
        compare_results(target, pred)

        pred = nystromformer_exp(query, key, value, m, stable_exp=True)
        print("\nStable exp nystromformer exp. std="+str(std))
        compare_results(target, pred)



def test_scaled_dot_product_attention_step():
    N = 100  # sequence length
    E = 5  # embedding dimension
    B = 2  # batch size
    H = 1  # num heads
    m = 10

    g = torch.Generator()
    g.manual_seed(0)

    std = 13

    query1 = torch.empty((B, N, E)).normal_(mean=0, std=std, generator=g)
    key1 = torch.empty((B, N, E)).normal_(mean=0, std=std, generator=g)
    value1 = torch.empty((B, N, E)).normal_(mean=0, std=std, generator=g)

    target1, kernel1, kernel2, kernel3 = _scaled_dot_product_attention(query1, key1, value1, m, return_kernels=True)

    # Now, let's try from zero-init
    state = _scaled_dot_product_attention_default_state(B, N, E, H, m)
    state = compute_landmarks(state, query1, key1, m)
    # state = nystromformer_exp(query1, key1, value1, m, stable_exp=True, state_mode=True)
    for i in range(N):
        if i==N-1:
            output_step, state, Beta, Gamma, Delta = _scaled_dot_product_attention_step(
                state, query1[:, i], key1[:, i], value1[:, i], return_kernels=True, update_landmarks=False, stable_exp=True
            )
        else:
            output_step, state = _scaled_dot_product_attention_step(
                state, query1[:, i], key1[:, i], value1[:, i], return_kernels=False, update_landmarks=False, stable_exp=True
            )

    print("\nDifference Beta: ")
    compute_diff(kernel1, Beta)
    print("\nDifference Gamma: ")
    compute_diff(kernel2, Gamma)
    print("\nDifference Delta: ")
    compute_diff(kernel3, Delta)
    print("\nDifference outputs: ")
    compute_diff(output_step, target1)
    pass

if __name__ == '__main__':
    test_stable_exp(seed=0)
    #test_scaled_dot_product_attention_step()

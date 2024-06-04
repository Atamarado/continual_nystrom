import fastf1
import pandas as pd
import numpy as np
import time
import torch
import types
import math
import sys
import os
import gc

from torch.nn.modules.activation import MultiheadAttention
from continual.transformer import SingleOutputMultiheadAttention, RetroactiveMultiheadAttention
from audio_classification.nystromformer.nystromformer import NystromMultiheadAttention
from audio_classification.nystromformer.continual_nystromformer import ContinualNystromMultiheadAttention

from flops import get_flops_and_params

def min_max_normalize(matrix):
    # Get the minimum and maximum values for each column
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)

    # Apply the Min-Max normalization formula
    normalized_matrix = (matrix - col_min) / (col_max - col_min)

    return (normalized_matrix*2)-1

def fetch_f1_data(batch_size=16):
    session = fastf1.get_session(2022, 9, 'R')
    session.load(telemetry=True)

    f1_data = []
    for driver_number in session.drivers:

        car_data = pd.DataFrame(session.car_data[driver_number].add_distance())
        pos_data = pd.DataFrame(session.pos_data[driver_number])

        f1_df = pd.merge_asof(pos_data, car_data, on="Date", direction="nearest")
        f1_df = f1_df.iloc[15000:35000].reset_index(drop=True)
        f1_df.drop(["Status", "Source_x", "SessionTime_x", "Source_y", "SessionTime_y"], axis=1, inplace=True)

        f1_df["Date"] = f1_df["Date"].astype('int64')
        f1_df["Time_x"] = f1_df["Time_x"].astype('int64')
        f1_df["Time_y"] = f1_df["Time_y"].astype('int64')
        f1_df["Brake"] = f1_df["Brake"].astype(float)

        f1_data.append(np.array(f1_df, dtype=np.float32))

    f1_data = np.concatenate(f1_data, axis=1)
    f1_data = min_max_normalize(f1_data)
    f1_data = np.reshape(f1_data, (1, -1, f1_data.shape[-1]))
    return torch.from_numpy(f1_data).to("cuda")

def init_model(model_type, B, N, H, E, M, fixed_landmarks, data=None, device=None):
    match model_type:
        case 'base':
            model = MultiheadAttention(E, H, device=device)
        case 'base_continual_single':
            model = SingleOutputMultiheadAttention(E, H, sequence_len=N, device=device)
        case 'base_continual_retroactive':
            model = RetroactiveMultiheadAttention(E, H, sequence_len=N, device=device)
        case 'nystromformer':
            model = NystromMultiheadAttention(N, E, H, M, B, device=device, single_output_forward=False)
        case 'continual_nystrom_single':
            model = ContinualNystromMultiheadAttention(E, H, M, sequence_len=N, batch_size=B, device=device, single_output_mode=True, single_output_forward=True)
        case 'continual_nystrom_retroactive':
            model = ContinualNystromMultiheadAttention(E, H, M, sequence_len=N, batch_size=B, device=device, single_output_mode=False)

    if fixed_landmarks and model_type in ['nystromformer', 'continual_nystrom_single', 'continual_nystrom_retroactive']:
        if data is None:
            raise Exception("We need data to precompute the landmarks")
        data_split = data[:, :, :data.size()[2]//N * N]
        model.fix_landmarks(torch.reshape(data_split, (-1, E, N)), alg="kmeans")

    if model_type in ['base_continual_single', 'base_continual_retroactive']:
        model.call_mode = "forward_steps"
    elif model_type in ['continual_nystrom_single', 'continual_nystrom_retroactive']:
        model.forward_mode = "forward_steps"

    return model

def compute_flops_and_params(model, model_type, batch_size, input_dim, seq_len):
    config = types.SimpleNamespace()
    config.model = model_type
    flops, params = get_flops_and_params(model, config, batch_size, input_dim, seq_len)
    return flops, params

def evaluate_model(N, E, B, H, data, model_type, M=None, fixed_landmarks=False, iterations=10):
    gc.collect()
    torch.cuda.empty_cache()

    # Model assertions
    assert M is None or M < N
    assert model_type in ['base', 'base_continual_single', 'base_continual_retroactive', 'nystromformer', 'continual_nystrom_single', 'continual_nystrom_retroactive']

    # Data assertions
    b, n, d = data.shape
    assert b >= B
    assert n >= N
    assert d >= E

    data = torch.permute(data, (0, 2, 1))
    all_data = data[:, :E]
    data = data[:B, :E, :2*N]

    data_warmup = data[:, :, :N]
    data_inference = data[:, :, -N:]

    # model = init_model(model_type, layers, 1, N, H, E, M, fixed_landmarks, data, device="cuda").to("cuda")
    # flops, params = compute_flops_and_params(model, model_type, 1, E//H, N)
    flops, params = 0, 0

    # Refresh model
    model = init_model(model_type, B, N, H, E, M, fixed_landmarks, all_data, device="cuda")

    if model_type in ['continual_nystrom_single', 'continual_nystrom_retroactive']:
        # Warmup data. Also sets the initial state for the continual models
        for _ in range(5):
            _ = model(data_warmup)

        start_time = time.time()
        for _ in range(iterations):
            _ = model(data_inference)
        end_time = time.time()
    elif model_type in ['base_continual_single', 'base_continual_retroactive']:
        data_warmup = torch.permute(data_warmup, (0, 2, 1))
        data_inference = torch.permute(data_inference, (0, 2, 1))

        # Warmup data. Also sets the initial state for the continual models
        for _ in range(5):
            _ = model(data_warmup)

        start_time = time.time()
        for _ in range(iterations):
            _ = model(data_inference)
        end_time = time.time()
    elif model_type == 'base':
        data = torch.permute(data, (2, 0, 1))
        data_warmup = torch.permute(data_warmup, (2, 0, 1))
        data_inference = torch.permute(data_inference, (2, 0, 1))

        for _ in range(5):
            _ = model(data_warmup, data_warmup, data_warmup)
        start_time = time.time()
        for _ in range(iterations):
            for data_index in range(data_inference.size()[0]):
                data_window = data[data_index:data_index+N, :, :]
                _ = model(data_window, data_window, data_window)
        end_time = time.time()
    else: # nystromformer
        for _ in range(5):
            _ = model(data_warmup, data_warmup, data_warmup)
        start_time = time.time()
        for _ in range(iterations):
            for data_index in range(data_inference.size()[2]):
                data_window = data[:, :, data_index:data_index+N]
                _ = model(data_window, data_window, data_window)
        end_time = time.time()

    running_time = end_time - start_time

    return flops, params, running_time

N_ITERATIONS = 10

data_path = 'data_runtime.pth'
if __name__ == '__main__':
    if os.path.exists(data_path):
        data = torch.load(data_path)
    else:
        data = fetch_f1_data()
        torch.save(data, data_path)

    print("Transformer:")
    flops, params, running_time = evaluate_model(1000, 50, 1, 1, data, "base",  iterations=20)
    print(running_time)

    print("Continual Transformer (single):")
    flops, params, running_time = evaluate_model(1000, 50, 2, 1, data, "base_continual_single",  iterations=20)
    print(running_time)

    print("Continual Transformer (retroactive):")
    flops, params, running_time = evaluate_model(1000, 50, 2, 1, data, "base_continual_retroactive", iterations=20)
    print(running_time)

    print("Nystromformer:")
    flops, params, running_time = evaluate_model(1000, 50, 1, 1, data, "nystromformer", iterations=20, M=8, fixed_landmarks=True)
    print(running_time)

    print("Continual Nystromformer (single):")
    flops, params, running_time = evaluate_model(1000, 50, 1, 1, data, "continual_nystrom_single", iterations=20, M=8, fixed_landmarks=True)
    print(running_time)

    print("Continual Nystromformer (retroactive):")
    flops, params, running_time = evaluate_model(1000, 50, 1, 1, data, "continual_nystrom_retroactive", iterations=20, M=8, fixed_landmarks=True)
    print(running_time)
    exit(0)

    result_list = []
    for N in range(100, 3001, 100):
        for E in range(100, 201, 100):
            for model in ['base', 'base_continual_single', 'base_continual_retroactive']:
                flops, params, running_time = evaluate_model(N, E, 1, 1, data, model, 1, iterations=N_ITERATIONS)
                results = {'N': N,
                           'E': E,
                           'model': model,
                           'M': 0,
                           'fixed_landmarks': False,
                           'running_time': running_time,
                           }
                print(results)
                sys.stdout.flush()
                result_list.append(results)
            for model in ['nystromformer', 'continual_nystrom_single', 'continual_nystrom_retroactive']:
                M = 64
                for fixed_landmarks in [True, False]:
                    flops, params, running_time = evaluate_model(N, E, 1, 1, data, model,
                                                                 M=M,
                                                                 fixed_landmarks=fixed_landmarks,
                                                                 iterations=N_ITERATIONS)
                    results = {'N': N,
                               'E': E,
                               'model': model,
                               'M': M,
                               'fixed_landmarks': fixed_landmarks,
                               'running_time': running_time,
                               }
                    print(results)
                    sys.stdout.flush()
                    result_list.append(results)

    result_list = []
    N = 1000
    E = 200
    for model in ['nystromformer', 'continual_nystrom_single', 'continual_nystrom_retroactive']:
        for M in [2**i for i in range(1, math.ceil(math.log2(N)))]:
            for fixed_landmarks in [True, False]:
                flops, params, running_time = evaluate_model(N, E, 1, 1, data, model,
                                                             M=M,
                                                             fixed_landmarks=fixed_landmarks,
                                                             iterations=N_ITERATIONS)
                results = {'N': N,
                           'E': E,
                           'model': model,
                           'M': M,
                           'fixed_landmarks': fixed_landmarks,
                           'running_time': running_time,
                           }
                print(results)
                sys.stdout.flush()
                result_list.append(results)


    N = 1000
    for E in range(100, 501, 100):
        for model in ['base', 'base_continual_single', 'base_continual_retroactive']:
            flops, params, running_time = evaluate_model(N, E, 1, 1, data, model, 1, iterations=N_ITERATIONS)
            results = {'N': N,
                       'E': E,
                       'model': model,
                       'M': 0,
                       'fixed_landmarks': False,
                       'running_time': running_time,
                       }
            print(results)
            sys.stdout.flush()
            result_list.append(results)
        for model in ['nystromformer', 'continual_nystrom_single', 'continual_nystrom_retroactive']:
            for M in [2**i for i in range(1, math.ceil(math.log2(N)))]:
                for fixed_landmarks in [True, False]:
                    flops, params, running_time = evaluate_model(N, E, 1, 1, data, model,
                                                                 M=M,
                                                                 fixed_landmarks=fixed_landmarks,
                                                                 iterations=N_ITERATIONS)
                    results = {'N': N,
                               'E': E,
                               'model': model,
                               'M': M,
                               'fixed_landmarks': fixed_landmarks,
                               'running_time': running_time,
                               }
                    print(results)
                    sys.stdout.flush()
                    result_list.append(results)

    pd.DataFrame.from_records(result_list).to_csv('results_runtime.csv')
import fastf1
import pandas as pd
import numpy as np
import time
import torch
import types
import math
import sys
import os

from torch.nn.modules.activation import MultiheadAttention
from continual.transformer import SingleOutputMultiheadAttention
from audio_classification.nystromformer.nystromformer import NystromMultiheadAttention
from audio_classification.nystromformer.continual_nystromformer import ContinualNystromMultiheadAttention

from audio_classification.models import CoTransformerModel, CoNystromTransformerModel
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

def init_model(model_type, layers, B, N, H, E, M, fixed_landmarks, data=None, device=None):
    match model_type:
        case 'base':
            model = MultiheadAttention(E, H, device=device)
        case 'base_continual':
            model = SingleOutputMultiheadAttention(E, H, sequence_len=N, device=device)
        case 'nystromformer':
            model = NystromMultiheadAttention(N, E, H, M, B, device=device, single_output_forward=False)
        case 'continual_nystrom':
            model = ContinualNystromMultiheadAttention(E, H, M, sequence_len=N, batch_size=B, device=device)

    if fixed_landmarks and model_type in ['nystromformer', 'continual_nystrom']:
        if data is None:
            raise Exception("We need data to precompute the landmarks")
        data_split = data[:, :data.size()[1]//N * N]
        model.fix_landmarks(torch.reshape(data_split, (-1, N, E)), alg="kmeans")

    if model == 'base_continual':
        model.call_mode = "forward_steps"
    elif model == 'continual_nystrom':
        model.forward_mode = "forward_steps"

    return model

def compute_flops_and_params(model, model_type, batch_size, input_dim, seq_len):
    config = types.SimpleNamespace()
    config.model = model_type
    flops, params = get_flops_and_params(model, config, batch_size, input_dim, seq_len)
    return flops, params

def evaluate_model(N, E, B, H, data, model_type, layers, M=None, fixed_landmarks=False, iterations=10):
    # Model assertions
    assert M is None or M < N
    assert model_type in ['base', 'base_continual', 'nystromformer', 'continual_nystrom']
    assert layers > 0 # TODO: Check what happens when layers > 2

    # Data assertions
    b, n, d = data.shape
    assert b >= B
    assert n >= N
    assert d >= E

    all_data = data[:, :, :E]
    data = data[:B, :2*N, :E]
    # data = torch.permute(data, (0, 2, 1))

    data_warmup = data[:, :N]  # data[:, :, :N]
    data_inference = data[:, -N:]  # data[:, :, -N:]

    # model = init_model(model_type, layers, 1, N, H, E, M, fixed_landmarks, data, device="cuda").to("cuda")
    # flops, params = compute_flops_and_params(model, model_type, 1, E//H, N)
    flops, params = 0, 0

    # Refresh model
    model = init_model(model_type, layers, B, N, H, E, M, fixed_landmarks, all_data, device="cuda")

    if model_type in ['base_continual', 'continual_nystrom']:
        # Warmup data. Also sets the initial state for the continual models
        for _ in range(5):
            _ = model(data_warmup, data_warmup, data_warmup)

        start_time = time.time()
        for _ in range(iterations):
            _ = model(data_inference, data_inference, data_inference)
        end_time = time.time()
    else:
        for _ in range(5):
            _ = model(data_warmup, data_warmup, data_warmup)
        start_time = time.time()
        for _ in range(iterations):
            for data_index in range(data_inference.size()[1]):
                data_window = data[:, data_index:data_index+N, :]
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

    # print("Transformer:")
    # flops, params, running_time = evaluate_model(400, 50, 1, 1, data, "base", 1, iterations=20)
    # print(running_time)
    #
    # print("Continual Transformer:")
    # flops, params, running_time = evaluate_model(400, 50, 1, 1, data, "base_continual", 1, iterations=20)
    # print(running_time)
    #
    # print("Nystromformer:")
    # flops, params, running_time = evaluate_model(400, 50, 1, 1, data, "nystromformer", 1, iterations=20, M=8, fixed_landmarks=True)
    # print(running_time)
    #
    # print("Continual Nystromformer:")
    # flops, params, running_time = evaluate_model(400, 50, 1, 1, data, "continual_nystrom", 1, iterations=20, M=8, fixed_landmarks=True)
    # print(running_time)
    # exit(0)

    result_list = []
    for N in range(300, 1001, 100):
        for E in range(100, 201, 100):
            # for num_layers in [1, 2]:
            for model in ['base', 'base_continual']:
                flops, params, running_time = evaluate_model(N, E, 1, 1, data, model, 1, iterations=N_ITERATIONS)
                results = {'N': N,
                           'E': E,
                           'num_layers': 1,
                           'model': model,
                           'M': 0,
                           'fixed_landmarks': False,
                           'flops': flops,
                           'params': params,
                           'running_time': running_time}
                print(results)
                sys.stdout.flush()
                result_list.append(results)
            for model in ['nystromformer', 'continual_nystrom']:
                for M in [2**i for i in range(1, math.ceil(math.log2(N)))]:
                    for fixed_landmarks in [True, False]:
                        flops, params, running_time = evaluate_model(N, E, 1, 1, data, model, 1,
                                                                     M=M,
                                                                     fixed_landmarks=fixed_landmarks,
                                                                     iterations=N_ITERATIONS)
                        results = {'N': N,
                                   'E': E,
                                   'num_layers': 1,
                                   'model': model,
                                   'M': M,
                                   'fixed_landmarks': fixed_landmarks,
                                   'flops': flops,
                                   'params': params,
                                   'running_time': running_time}
                        print(results)
                        sys.stdout.flush()
                        result_list.append(results)

    pd.DataFrame.from_records(result_list).to_csv('results_runtime.csv')
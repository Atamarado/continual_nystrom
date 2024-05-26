import fastf1
import pandas as pd
import numpy as np
import time
import torch
import types
import math

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
    f1_data = np.reshape(f1_data, (batch_size, -1, f1_data.shape[-1]))
    return torch.from_numpy(f1_data)

def init_model(model_type, layers, B, N, H, E, M, fixed_landmarks, data=None):
    if model_type in ['base', 'base_continual']:
        model = CoTransformerModel(
            embed_dim=E,
            depth=layers,
            heads=H,
            mlp_dim=E,
            sequence_len=N,
        )
    else:
        model = CoNystromTransformerModel(
            embed_dim=E,
            depth=layers,
            heads=H,
            mlp_dim=E,
            sequence_len=N,
            num_landmarks=M,
            batch_size=B,
            fixed_landmarks=fixed_landmarks
        )

        if fixed_landmarks:
            if data is None:
                raise Exception("We need data to precompute the landmarks")
            model[0][1].fix_landmarks(data)

    return model

def compute_flops_and_params(model, model_type, batch_size, input_dim, seq_len):
    config = types.SimpleNamespace()
    config.model = model_type
    # return 0, 0

    # TODO: Check what's going on with continual nystrom
    flops, params = get_flops_and_params(model, config, batch_size, input_dim, seq_len)

    # # Add cost of transformer module
    # transformer_module = model[0][1]
    # flops += transformer_module.flops()

    return flops, params

def evaluate_model(N, E, B, H, data, model_type, layers, M=None, fixed_landmarks=False, iterations=10, compute_inverse=True):
    # Model assertions
    assert M is None or M < N
    assert model_type in ['base', 'base_continual', 'nystromformer', 'continual_nystrom']
    assert layers > 0 # TODO: Check what happens when layers > 2

    # Data assertions
    b, n, d = data.shape
    assert b >= B
    assert n >= N
    assert d >= E

    data = data[:B, :2*N, :E]
    data = torch.permute(data, (0, 2, 1))

    data_warmup = data[:, :, :N]
    data_inference = data[:, :, -N:]

    model = init_model(model_type, layers, 1, N, H, E, M, fixed_landmarks, data)
    flops, params = compute_flops_and_params(model, model_type, 1, E//H, N)
    # flops, params = 0, 0

    # Refresh model
    model = init_model(model_type, layers, B, N, H, E, M, fixed_landmarks, data)

    if not compute_inverse and model_type in ['nystromformer', 'continual_nystrom']:
        model[0][1].compute_inverse = False

    if model_type in ['base_continual', 'continual_nystrom']:
        model.call_mode = "forward_steps"
        # Warmup data. Also sets the initial state for the continual models
        _ = model(data_warmup)

        model.call_mode = "forward_step"
        start_time = time.time()
        for _ in range(iterations):
            for data_index in range(data_inference.size()[2]):
                _ = model(data[:, :, data_index])
        end_time = time.time()
    else:
        _ = model(data_warmup)
        start_time = time.time()
        for _ in range(iterations):
            for data_index in range(data_inference.size()[2]):
                _ = model(data[:, :, data_index: data_index+N])
        end_time = time.time()

    running_time = end_time - start_time

    return flops, params, running_time

N_ITERATIONS = 5
if __name__ == '__main__':
    data = fetch_f1_data()

    result_list = []
    for N in range(100, 1001, 50):
        for E in range(50, 201, 50):
            for num_layers in [1, 2]:
                for model in ['base', 'base_continual']:
                    flops, params, running_time = evaluate_model(N, E, 16, 1, data, model, num_layers, iterations=N_ITERATIONS)
                    results = {'N': N,
                               'E': E,
                               'num_layers': num_layers,
                               'model': model,
                               'M': 0,
                               'fixed_landmarks': False,
                               'compute_inverse': False,
                               'flops': flops,
                               'params': params,
                               'running_time': running_time}
                    print(results)
                    result_list.append(results)
                for model in ['nystromformer', 'continual_nystrom']:
                    for M in [2**i for i in range(1, math.ceil(math.log2(N)))]:
                        for fixed_landmarks in [True, False]:
                            for compute_inverse in [True, False]:
                                flops, params, running_time = evaluate_model(N, E, 16, 1, data, model, 1,
                                                                             M=M,
                                                                             fixed_landmarks=fixed_landmarks,
                                                                             iterations=N_ITERATIONS,
                                                                             compute_inverse=compute_inverse)
                                results = {'N': N,
                                           'E': E,
                                           'num_layers': num_layers,
                                           'model': model,
                                           'M': M,
                                           'fixed_landmarks': fixed_landmarks,
                                           'compute_inverse': compute_inverse,
                                           'flops': flops,
                                           'params': params,
                                           'running_time': running_time}
                                print(results)
                                result_list.append(results)

    pd.DataFrame.from_records(result_list).to_csv('results_runtime.csv')
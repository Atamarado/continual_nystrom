from ptflops import get_model_complexity_info
import torch

INPUT_DIM = 128
SEQ_LEN = 120

def get_flops_and_params(model, config, batch_size=1, input_dim=INPUT_DIM, seq_len=SEQ_LEN):
    if config.model in ["base_continual", "continual_nystrom"]:
        warm_up_input = torch.randn(batch_size, input_dim, seq_len)
        model.to('cpu')
        assert next(model.parameters()).is_cuda == warm_up_input.is_cuda
        model.forward_steps(warm_up_input)  # Warm up model
        model.call_mode = "forward_step"
        flops, params = get_model_complexity_info(
            model, (input_dim,), as_strings=False, print_per_layer_stat=False
        )
    else:
        flops, params = get_model_complexity_info(
            model, (input_dim, seq_len), as_strings=False, print_per_layer_stat=False
        )
    return flops, params
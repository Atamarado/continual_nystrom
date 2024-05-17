import argparse
import ast

def get_args_parser():
    parser = argparse.ArgumentParser("Audio classification config", add_help=False)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--version", default='v5', type=str)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--model", default='base', type=str,
                        help='[base, base_continual, nystromformer, continual_nystrom]')
    parser.add_argument("--num_landmarks", default=10, type=int)
    parser.add_argument("--fit_layer_epochs", default=[], type=ast.literal_eval)
    parser.add_argument("--freeze_weights", default="both", type=str,
                        help='[both, true, false]')

    parser.add_argument("--data_seed", default=0, type=int)
    parser.add_argument("--model_seed", default=0, type=int)

    return parser

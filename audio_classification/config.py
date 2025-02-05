import argparse
import ast

def get_args_parser():
    parser = argparse.ArgumentParser("Audio classification config", add_help=False)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    # parser.add_argument("--version", default='v5', type=str) # Legacy argument
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--model", default='base', type=str,
                        help='[base, base_continual, nystromformer, continual_nystrom]')
    parser.add_argument("--num_landmarks", default=10, type=int, help="number of landmarks used for the nyström-based transformers")
    parser.add_argument("--fit_layer_epochs", default=[], type=ast.literal_eval,
                        help="Number of epochs used to fine-tune the model after the landmarks in one layer are frozen. Provide a list with the number of epochs for every layer")
    parser.add_argument("--freeze_weights", default="both", type=str,
                        help='[both, true, false].'
                             'If `true`, the model will freeze the weights after the --epochs are performed.'
                             'If `false`, the model will not freeze the weights and will use have continual landmarks'
                             'both tries both configurations')

    parser.add_argument("--data_seed", default=0, type=int, help="seed used to perform the dataset split into train/val/test")
    parser.add_argument("--model_seed", default=0, type=int, help="seed used to initialize the model weights")

    return parser

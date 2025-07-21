import argparse
import ast
import json
import os

def get_args_parser():
    parser = argparse.ArgumentParser("Continual Nyströmformer training config", add_help=False)

    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--model", default='base', type=str,
                        help='[base, base_continual, nystromformer, continual_nystrom]')

    # Nyströmformer args: Unused
    parser.add_argument("--num_landmarks", default=10, type=int, help="number of landmarks used for the nyström-based transformers")
    parser.add_argument("--fit_layer_epochs", default=[], type=ast.literal_eval,
                        help="Number of epochs used to fine-tune the model after the landmarks in one layer are frozen. Provide a list with the number of epochs for every layer")
    parser.add_argument("--freeze_weights", default="both", type=str,
                        help='[both, true, false].'
                             'If `true`, the model will freeze the weights after the --epochs are performed.'
                             'If `false`, the model will not freeze the weights and will use have continual landmarks'
                             'both tries both configurations')

    parser.add_argument("--dataset", default='gtzan', type=str,
                        help='[gtzan, thumos, electricity]')

    parser.add_argument("--data_seed", default=0, type=int, help="seed used to perform the dataset split into train/val/test. Only used if dataset==gtzan")
    parser.add_argument("--model_seed", default=0, type=int, help="seed used to initialize the model weights")

    parser.add_argument("--feature", default="anet", type=str, help="feature type. Only used if dataset==thumos. Options: [anet, kin]")

    parser.add_argument("--seq_len", default=-1, type=int, help="number of tokens fed to the transformer model. It cannot be bigger than the sequence length")

    return parser

def set_dataset_config(config):
    match config.dataset:
        case 'gtzan':
            config.batch_size = 32
            config.lr = 1e-5
            config.weight_decay = 1e-4
            config.epochs = 50

            config.seq_len = 120
            config.num_points_nystrom = -1

            config.out_folder = 'audio_classification/saved_models'
            config.log_folder = 'audio_classification/raw_results'
        case 'thumos':
            config.batch_size = 128
            config.lr = 1e-4
            config.weight_decay = 1e-4
            config.epochs = 5
            config.resize_feature = False
            config.lr_drop = 1
            config.clip_max_norm = 1.0
            config.dataparallel = False
            config.removelog = False
            config.query_num = 8
            config.classification_pred_loss_coef = 0.5
            config.enc_layers = 64
            config.dim_feature = 3072 if config.feature == 'anet' else 4096
            config.patch_dim = 1
            config.embedding_dim = 1024
            config.num_heads = 8
            config.attn_dropout_rate = 0.1
            config.num_embeddings = 64
            config.hidden_dim = 1024
            config.dropout_rate = 0.1
            config.numclass = 22
            config.classification_x_loss_coef = 0.3
            config.classification_h_loss_coef = 1
            config.similar_loss_coef = 0.1
            config.margin = 1.0
            config.dataset_file = 'CoOadTR/data/data_info_new.json'
            config.frozen_weights = None
            config.device = 'cuda'
            config.resume = ''
            config.start_epoch = 1
            config.eval = False
            config.num_workers = 8
            config.world_size = 1
            config.dist_url = 'tcp://127.0.0.1:12342'

            config.out_folder = 'CoOadTR/saved_models'
            config.log_folder = 'CoOadTR/raw_results'

            config.seq_len = 64
            config.num_points_nystrom = 50000

            with open(config.dataset_file, "r") as f:
                data_info = json.load(f)[config.dataset.upper()]
            config.train_session_set = data_info["train_session_set"]
            config.test_session_set = data_info["test_session_set"]
            config.class_index = data_info["class_index"]
            config.numclass = len(config.class_index)

            # utils.init_distributed_mode(config)
        case 'electricity':
            config.batch_size = 32
            config.lr = 1e-5
            config.weight_decay = 1e-4
            config.epochs = 10

            if config.seq_len <= -1:
                config.seq_len = 64

            config.num_points_nystrom = 50000

            config.out_folder = 'electricity/saved_models'
            config.log_folder = 'electricity/raw_results'
    return config

def get_config():
    parser = argparse.ArgumentParser(
        "Audio classification python program", parents=[get_args_parser()]
    )
    config = parser.parse_args()

    config = set_dataset_config(config)
    os.makedirs(config.out_folder, exist_ok=True)
    os.makedirs(config.log_folder, exist_ok=True)

    return config
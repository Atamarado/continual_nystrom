import argparse
import ast


def str2bool(string):
    return True if string.lower() == "true" else False


def get_args_parser():
    parser = argparse.ArgumentParser("Set IDU Online Detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)  # 1e-4
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument(
        "--resize_feature",
        default=False,
        type=str2bool,
        help="run resize prepare_data or not",
    )
    parser.add_argument("--lr_drop", default=1, type=int)
    parser.add_argument(
        "--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm"
    )  # dataparallel
    parser.add_argument(
        "--dataparallel", action="store_true", help="multi-gpus for training"
    )
    parser.add_argument("--removelog", action="store_true", help="remove old log")

    # * Network
    parser.add_argument(
        "--version", default="v3", type=str, help="fixed or learned"
    )  # learned  fixed
    # decoder
    parser.add_argument(
        "--query_num", default=8, type=int, help="Number of query_num (prediction)"
    )
    parser.add_argument(
        "--decoder_layers", default=5, type=int, help="Number of decoder_layers"
    )
    parser.add_argument(
        "--decoder_embedding_dim",
        default=1024,
        type=int,  # 1024
        help="decoder_embedding_dim",
    )
    parser.add_argument(
        "--decoder_embedding_dim_out",
        default=1024,
        type=int,  # 256 512 1024
        help="decoder_embedding_dim_out",
    )
    parser.add_argument(
        "--decoder_attn_dropout_rate",
        default=0.1,
        type=float,  # 0.1=0.2
        help="rate of decoder_attn_dropout_rate",
    )
    parser.add_argument(
        "--decoder_num_heads", default=4, type=int, help="decoder_num_heads"  # 8 4
    )
    parser.add_argument(
        "--classification_pred_loss_coef", default=0.5, type=float
    )  # 0.5

    parser.add_argument(
        "--num_landmarks", default=10, type=int, help="number of landmarks used for the nyström-based transformers"
    )
    parser.add_argument(
        "--enc_layers", default=64, type=int, help="Number of enc_layers"
    )
    parser.add_argument(
        "--lr_backbone", default=1e-4, type=float, help="lr_backbone"  # 2e-4
    )
    parser.add_argument(
        "--feature", default="anet", type=str, help="feature type"
    )
    parser.add_argument(
        "--dim_feature", default=3072, type=int, help="input feature dims"
    )
    parser.add_argument("--patch_dim", default=1, type=int, help="input feature dims")
    parser.add_argument(
        "--embedding_dim", default=1024, type=int, help="input feature dims"  # 1024
    )
    parser.add_argument("--num_heads", default=8, type=int, help="input feature dims")
    parser.add_argument("--num_layers", default=3, type=int, help="input feature dims")
    parser.add_argument(
        "--attn_dropout_rate", default=0.1, type=float, help="attn dropout"
    )
    parser.add_argument(
        "--num_embeddings",
        default=64,
        type=int,  # 64, 127
        help="Number of embeddings in RecyclingPositionalEncoding",
    )
    parser.add_argument(
        "--hidden_dim",
        default=1024,
        type=int,  # 512 1024
        help="Size of the embeddings",
    )
    parser.add_argument(
        "--dropout_rate", default=0.1, type=float, help="Dropout applied "
    )

    parser.add_argument("--numclass", default=22, type=int, help="Number of class")

    # * Loss coefficients
    parser.add_argument("--classification_x_loss_coef", default=0.3, type=float)
    parser.add_argument("--classification_h_loss_coef", default=1, type=float)
    parser.add_argument("--similar_loss_coef", default=0.1, type=float)  # 0.3
    parser.add_argument("--margin", default=1.0, type=float)

    # dataset parameters
    parser.add_argument(
        "--dataset", default="thumos", type=str, help="thumos or tvseries"
    )
    parser.add_argument("--dataset_file", type=str, default="data/data_info_new.json")
    parser.add_argument("--frozen_weights", type=str, default=None)
    parser.add_argument(
        "--thumos_data_path",
        type=str,
        default="/home/dancer/mycode/Temporal.Online.Detection/"
        "Online.TRN.Pytorch/preprocess/",
    )
    parser.add_argument(
        "--thumos_anno_path", type=str, default="data/thumos_{}_anno.pickle"
    )
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument("--seed", default=20, type=int, help="seed used to initialize the model weights")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=1, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:12342",
        help="url used to set up distributed training",
    )

    parser.add_argument("--model", default='base', type=str,
                        help='[base, base_continual, nystromformer, continual_nystrom]')
    parser.add_argument("--fit_layer_epochs", default=[], type=ast.literal_eval,
                        help="Number of epochs used to fine-tune the model after the landmarks in one layer are frozen. Provide a list with the number of epochs for every layer")
    parser.add_argument("--freeze_weights", default="both", type=str,
                        help='[both, true, false].'
                             'If `true`, the model will freeze the weights after the --epochs are performed.'
                             'If `false`, the model will not freeze the weights and will use have continual landmarks'
                             'both tries both configurations')
    # 'env://'
    return parser

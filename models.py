import torch
import torch.nn as nn
from collections import OrderedDict

import continual_dev as co
from continual_dev import RecyclingPositionalEncoding

def CoNystromTransformerModel(
    embed_dim,
    depth,
    heads,
    mlp_dim,
    num_landmarks,
    dropout_rate=0.1,
    sequence_len=64,
    batch_size=32,
    device=None,
    fixed_landmarks=False,
):
    if depth == 1:
        transformer_encoder = co.SingleOutputNystromTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            num_landmarks=num_landmarks,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            batch_size=batch_size,
            device=device,
            single_output_forward=True,
            fixed_landmarks=fixed_landmarks,
        )
    else:
        encoder_layer = co.NystromTransformerEncoderLayerFactory(
            d_model=embed_dim,
            nhead=heads,
            num_landmarks=num_landmarks,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            batch_size=batch_size,
            device=device,
            fixed_landmarks=fixed_landmarks,
        )
        transformer_encoder = co.NystromTransformerEncoder(encoder_layer, num_layers=depth)
    return transformer_encoder

def CoTransformerModel(
    embed_dim,
    depth,
    heads,
    mlp_dim,
    dropout_rate=0.1,
    sequence_len=64,
    device=None,
):
    if depth == 1:
        transformer_encoder = co.SingleOutputTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            single_output_forward=True,
            device=device,
        )
    else:
        encoder_layer = co.TransformerEncoderLayerFactory(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            device=device,
        )
        transformer_encoder = co.TransformerEncoder(encoder_layer, num_layers=depth)
    return transformer_encoder

def CoVisionTransformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1,
):

    assert embedding_dim % num_heads == 0

    linear_encoding = co.Linear(input_dim, embedding_dim, channel_dim=1)
    position_encoding = RecyclingPositionalEncoding(
        embedding_dim,
        int(embedding_dim * 1.0),  # Change num pos enc to cycle between
        forward_update_index_steps=1,
    )

    # pe_dropout = nn.Dropout(p=dropout_rate)
    encoder = CoTransformerModel(
        embedding_dim,
        num_layers,
        num_heads,
        attn_ff_hidden_dim,
        dropout_rate,
        sequence_len,
    )
    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    return co.Sequential(
        linear_encoding,
        position_encoding,
        # pe_dropout,
        encoder,
        pre_head_ln,
        mlp_head,
    )

def NonCoNystromVisionTransformer(
        sequence_len,
        input_dim,
        embedding_dim,
        attn_ff_hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        device=None,
        dropout_rate=0.1,
        num_landmarks=10,
        fixed_landmarks=False,
):
    return CoNystromVisionTransformer(
        sequence_len,
        input_dim,
        embedding_dim,
        attn_ff_hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        device=device,
        num_landmarks=num_landmarks,
        dropout_rate=dropout_rate,
        continual=False,
        fixed_landmarks=fixed_landmarks,
    )

class LearnedPositionalEncoding(co.CoModule, nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        position_embeddings = torch.permute(position_embeddings, (0, 2, 1))
        return x + position_embeddings

def CoNystromVisionTransformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1,
    continual=True,
    device=None,
    batch_size=32,
    num_landmarks=10,
    fixed_landmarks=False,
):

    assert embedding_dim % num_heads == 0

    linear_encoding = co.Linear(input_dim, embedding_dim, channel_dim=1)
    if continual:
        position_encoding = RecyclingPositionalEncoding(
            embedding_dim,
            int(embedding_dim * 1.0),  # Change num pos enc to cycle between
            forward_update_index_steps=1,
        )
    else:
        position_encoding = LearnedPositionalEncoding(
            embedding_dim,
            embedding_dim,
            sequence_len
        )

    # pe_dropout = nn.Dropout(p=dropout_rate)

    encoder = CoNystromTransformerModel(
        embedding_dim,
        num_layers,
        num_heads,
        attn_ff_hidden_dim,
        num_landmarks,
        dropout_rate,
        sequence_len,
        device=device,
        batch_size=batch_size,
        fixed_landmarks=fixed_landmarks,
    )
    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    if continual:
        return co.Sequential(
            linear_encoding,
            position_encoding,
            # pe_dropout,
            encoder,
            pre_head_ln,
            mlp_head,
        )
    else:
        return co.Sequential(
            linear_encoding,
            position_encoding,
            # pe_dropout,
            encoder,
            pre_head_ln,
            mlp_head,
        )


def NonCoVisionTransformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1,
):

    assert embedding_dim % num_heads == 0

    linear_encoding = co.Linear(input_dim, embedding_dim, channel_dim=1)
    position_encoding = LearnedPositionalEncoding(
        embedding_dim,
        embedding_dim,
        sequence_len
    )

    # pe_dropout = nn.Dropout(p=dropout_rate)

    encoder = CoTransformerModel(
        embedding_dim,
        num_layers,
        num_heads,
        attn_ff_hidden_dim,
        dropout_rate,
        sequence_len,
    )
    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    return nn.Sequential(
        linear_encoding,
        position_encoding,
        # pe_dropout,
        encoder,
        pre_head_ln,
        mlp_head,
    )

def get_audio_model(config, training_dataset=None):

    match config.dataset:
        case 'gtzan':
            seq_len = 128 if config.seq_len <= 0 else config.seq_len
            input_dim = 128
            out_dim = 10
            embed_dim = 192
        case 'electricity':
            seq_len = 64 if config.seq_len <= 0 else config.seq_len
            input_dim = 376
            out_dim = 1
            embed_dim = 512
        case _:
            raise NotImplementedError

    match config.model:
        case "base":
            model = NonCoVisionTransformer(
                sequence_len=seq_len,
                input_dim=input_dim,
                embedding_dim=embed_dim,
                attn_ff_hidden_dim=1024,
                out_dim=out_dim,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
            )
        case "base_continual":
            model = CoVisionTransformer(
                sequence_len=seq_len,
                input_dim=input_dim,
                embedding_dim=embed_dim,
                attn_ff_hidden_dim=1024,
                out_dim=out_dim,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
            )
        case "nystromformer":
            model = NonCoNystromVisionTransformer(
                sequence_len=seq_len,
                input_dim=input_dim,
                embedding_dim=embed_dim,
                attn_ff_hidden_dim=1024,
                out_dim=out_dim,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
                device="cuda",
                num_landmarks=config.num_landmarks,
            )
        case "continual_nystrom":
            model = CoNystromVisionTransformer(
                sequence_len=seq_len,
                input_dim=input_dim,
                embedding_dim=embed_dim,
                attn_ff_hidden_dim=1024,
                out_dim=out_dim,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
                batch_size=config.batch_size,
                device="cuda",
                num_landmarks=config.num_landmarks,
            )
    return model
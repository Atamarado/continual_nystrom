import continual as co
import torch
import torch.nn as nn

from continual import RecyclingPositionalEncoding
from nystromformer.transformer import (SingleOutputNystromTransformerEncoderLayer,
                                       NystromTransformerEncoderLayerFactory,
                                       NystromTransformerEncoder)
from nystromformer.nystromformer import LearnedPositionalEncoding

def CoNystromTransformerModel(
    embed_dim,
    depth,
    heads,
    mlp_dim,
    num_landmarks,
    dropout_rate=0.1,
    sequence_len=64,
    batch_size=32,
    device=None
):
    if depth == 1:
        transformer_encoder = SingleOutputNystromTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            num_landmarks=num_landmarks,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            batch_size=batch_size,
            device=device,
            single_output_forward=True
        )
    else:
        encoder_layer = NystromTransformerEncoderLayerFactory( # TODO: Change for Nystrom
            d_model=embed_dim,
            nhead=heads,
            num_landmarks=num_landmarks,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            batch_size=batch_size,
            device=device
        )
        transformer_encoder = NystromTransformerEncoder(encoder_layer, num_layers=depth)
    return transformer_encoder

def CoTransformerModel(
    embed_dim,
    depth,
    heads,
    mlp_dim,
    dropout_rate=0.1,
    sequence_len=64,
):
    if depth == 1:
        transformer_encoder = co.SingleOutputTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len,
            single_output_forward=True
        )
    else:
        encoder_layer = co.TransformerEncoderLayerFactory(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=nn.GELU(),
            sequence_len=sequence_len)
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

    pe_dropout = nn.Dropout(p=dropout_rate)

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
        pe_dropout,
        encoder,
        pre_head_ln,
        mlp_head,
    )

def NonCoNystromVisionTransformer( # TODO: Temporary solution. Fix later
        sequence_len,
        input_dim,
        embedding_dim,
        attn_ff_hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        dropout_rate=0.1
):
    return CoNystromVisionTransformer(
        sequence_len,
        input_dim,
        embedding_dim,
        attn_ff_hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        dropout_rate=dropout_rate,
        continual=False,
    )

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
    batch_size=32
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

    pe_dropout = nn.Dropout(p=dropout_rate)

    encoder = CoNystromTransformerModel(
        embedding_dim,
        num_layers,
        num_heads,
        attn_ff_hidden_dim,
        sequence_len//10,  # TODO: Find better solution
        dropout_rate,
        sequence_len,
        device=device,
        batch_size=batch_size,
    )
    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    if continual:
        return co.Sequential(
            linear_encoding,
            position_encoding,
            pe_dropout,
            encoder,
            pre_head_ln,
            mlp_head,
        )
    else:
        return nn.Sequential(
            linear_encoding,
            position_encoding,
            pe_dropout,
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

    pe_dropout = nn.Dropout(p=dropout_rate)

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
        pe_dropout,
        encoder,
        pre_head_ln,
        mlp_head,
    )
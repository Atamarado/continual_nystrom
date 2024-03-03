import continual as co
import torch
import torch.nn as nn

from continual import RecyclingPositionalEncoding


def CoTransformerModel(
    embed_dim,
    depth,
    heads,
    mlp_dim,
    dropout_rate=0.1,
    sequence_len=64,
):
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

class LearnedPositionalEncoding(nn.Module):
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
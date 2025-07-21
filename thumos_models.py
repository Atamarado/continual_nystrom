import continual_dev as co
import torch.nn as nn
import torch

from collections import OrderedDict

def get_thumos_model(config):
	return VisionTransformer(
		args=config,
		img_dim=config.enc_layers,
		patch_dim=config.patch_dim,
		out_dim=config.numclass,
		embedding_dim=config.embedding_dim,
		num_heads=config.num_heads,
		num_layers=config.num_layers,
		hidden_dim=config.hidden_dim,
		dropout_rate=config.dropout_rate,
		attn_dropout_rate=config.attn_dropout_rate,
		num_channels=config.dim_feature,
		model_type=config.model,
		num_landmarks=config.num_landmarks,
		device=config.device,
		batch_size=config.batch_size,
	)

class CoLearnedPositionalEncoding(co.CoModule, nn.Module):
	def __init__(self, max_position_embeddings, embedding_dim, seq_length):
		super(CoLearnedPositionalEncoding, self).__init__()
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

def VisionTransformer(
	args,
	img_dim,
	patch_dim,
	out_dim,
	embedding_dim,
	num_heads,
	num_layers,
	hidden_dim,
	device,
	dropout_rate=0.0,
	attn_dropout_rate=0.0,
	use_representation=True,
	conv_patch_representation=False,
	with_camera=True,
	with_motion=True,
	num_channels=3072,
	model_type="base",
	num_landmarks=10,
	batch_size=1,
):

	assert embedding_dim % num_heads == 0
	assert img_dim % patch_dim == 0

	num_patches = int(img_dim // patch_dim)
	seq_length = num_patches  # no class token
	flatten_dim = patch_dim * patch_dim * num_channels

	linear_encoding = co.Linear(flatten_dim, embedding_dim, channel_dim=1)

	if model_type in ['base_continual', 'continual_nystrom']:
		position_encoding = co.RecyclingPositionalEncoding(
			embedding_dim,
			args.num_embeddings,
			forward_update_index_steps=1,
		)
	else:
		position_encoding = CoLearnedPositionalEncoding(
			args.num_embeddings,
			embedding_dim,
			seq_length
		)


	pe_dropout = nn.Dropout(p=dropout_rate)

	if model_type in ['base', 'base_continual']:
		encoder = CoTransformerModel(
			embedding_dim,
			num_layers,
			num_heads,
			hidden_dim,
			device,
			dropout_rate,
			attn_dropout_rate,
		)
	else: # model_type in ['nystromformer', 'continual_nystrom']
		encoder = CoNystromTransformerModel(
			embedding_dim,
			num_layers,
			num_heads,
			hidden_dim,
			device,
			dropout_rate,
			attn_dropout_rate,
			num_landmarks=num_landmarks,
			batch_size=batch_size,
		)
	pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
	mlp_head = co.Linear(hidden_dim, out_dim, channel_dim=1)

	return co.Sequential(
		linear_encoding,
		position_encoding,
		pe_dropout,
		encoder,
		pre_head_ln,
		mlp_head,
	)

def CoNystromTransformerModel(
	dim,
	depth,
	heads,
	mlp_dim,
	device,
	dropout_rate=0.1,
	attn_dropout_rate=0.1,
	sequence_len=64,
	num_landmarks=10,
	batch_size=1,
):
	assert depth in {1, 2}

	if depth == 1:
		return co.SingleOutputNystromTransformerEncoderLayer(
			sequence_len=sequence_len,
			d_model=dim,
			nhead=heads,
			dropout=dropout_rate,
			query_index=-1,
			dim_feedforward=mlp_dim,
			activation=nn.GELU(),
			device=device,
			dtype=None,
			single_output_forward=True,
			num_landmarks=num_landmarks,
			batch_size=batch_size,
		)

	# depth == 2
	layer_factory = co.NystromTransformerEncoderLayerFactory(
		sequence_len=sequence_len,
		d_model=dim,
		nhead=heads,
		dropout=dropout_rate,
		dim_feedforward=mlp_dim,
		activation=nn.GELU(),
		device=device,
		dtype=None,
		num_landmarks=num_landmarks,
		batch_size=batch_size,
	)
	return co.NystromTransformerEncoder(layer_factory, num_layers=depth)

def CoTransformerModel(
	dim,
	depth,
	heads,
	mlp_dim,
	device,
	dropout_rate=0.1,
	attn_dropout_rate=0.1,
	sequence_len=64,
):

	if depth == 1:
		return co.SingleOutputTransformerEncoderLayer(
			sequence_len=sequence_len,
			d_model=dim,
			nhead=heads,
			dropout=dropout_rate,
			query_index=-1,
			dim_feedforward=mlp_dim,
			activation=nn.GELU(),
			device=device,
			dtype=None,
			single_output_forward=True
		)

	# depth >= 2
	layer_factory = co.TransformerEncoderLayerFactory(
		sequence_len=sequence_len,
		d_model=dim,
		nhead=heads,
		dropout=dropout_rate,
		dim_feedforward=mlp_dim,
		activation=nn.GELU(),
		device=device,
		dtype=None,
	)
	return co.TransformerEncoder(layer_factory, num_layers=depth)

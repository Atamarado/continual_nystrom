import torch

import pandas as pd
import numpy as np
import os

try:
	import cPickle as pickle
except ModuleNotFoundError:
	import pickle

RAW_DATA_PATH = 'electricity/data/LD2011_2014.txt'
PROCESSED_DATA_PATH = 'electricity/data/processed'

SEQ_LEN = 64

class StandardScaler:
	def __str__(self):
		return "std"

	def fit(self, data):
		self.mean = data.mean(0, keepdim=True)
		self.std = data.std(0, keepdim=True)

		self.std[self.std < 10e-3] = 10e-3

	def predict(self, data):
		data = data - self.mean
		data = data / self.std
		return data

	def reverse(self, data):
		data = data * self.std
		data = data + self.mean
		return data

# Window sliding technique used for Deep Learning models
def split_sequence(X, Y, steps, out, y_offset=-4):
	Xs, Ys = list(), list()
	for i in range(len(X)):
		end = i + steps
		outi = end + out
		if outi > len(X) - 1:
			break
		seqx, seqy = X[i:end], Y[end:outi]
		Xs.append(seqx)
		Ys.append(seqy)

	Xs, Ys = torch.stack(Xs), torch.cat(Ys)

	if y_offset < 0:
		Ys = Ys.roll(y_offset, dims=0)[:y_offset]
		Xs = Xs[:y_offset]

	return Xs, Ys

OUTPUT_DATA = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
OUTPUT_SCALERS = ['X_scaler', 'y_scaler']
def prepare_dataset(scaler=None, scale_y=False, seq_len=SEQ_LEN):
	df = pd.read_csv(RAW_DATA_PATH, sep=';', low_memory=False)
	df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
	df['Timestamp'] = pd.to_datetime(df["Timestamp"])

	# df.dropna(inplace=True)
	df.loc[:, df.columns != 'Timestamp'] = df.loc[:, df.columns != 'Timestamp'].astype('str').apply(
		lambda col: pd.to_numeric(col.str.replace(',', '.'), errors='raise')
	)
	df['y'] = df.drop(columns='Timestamp').sum(axis=1)

	df['timestamp_numeric'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / 60
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])
	df['hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
	df['hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
	df['day_of_week_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.dayofweek / 7)
	df['day_of_week_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.dayofweek / 7)
	df['interval_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.minute // 15 / 4)
	df['interval_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.minute // 15 / 4)

	# Feature MT_178 is always 0 in the training set
	X = torch.from_numpy(df.drop(columns=['Timestamp', 'MT_178', 'MT_132']).to_numpy().astype(float)).type(torch.float32)
	y = torch.from_numpy(df['y'].to_numpy().astype(float)).type(torch.float32)

	train_end = int(len(X) * 0.80)
	val_end = int(len(X) * 0.90)
	X_train, y_train = X[:train_end], y[:train_end]
	X_val, y_val = X[train_end:val_end], y[train_end:val_end]
	X_test, y_test = X[val_end:], y[val_end:]

	if scaler:
		scaler_X = StandardScaler()
		scaler_X.fit(X_train)

		X_train = scaler_X.predict(X_train)
		X_val = scaler_X.predict(X_val)
		X_test = scaler_X.predict(X_test)

		# filename = OUTPUT_SCALERS[0] + '_' + str(scaler_X) + '_' + str(seq_len) + '.pkl'
		# path = os.path.join(OUTPUT_FOLDER, filename)
		# with open(path, 'wb') as f:
		# 	pickle.dump(scaler_X, f)

		if scale_y:
			scaler_y = StandardScaler()
			scaler_y.fit(y_train)

			y_train = scaler_y.predict(y_train)
			y_val = scaler_y.predict(y_val)
			y_test = scaler_y.predict(y_test)

			# filename = OUTPUT_SCALERS[1] + '_' + str(scaler_y) + '_' + str(seq_len) + '.pkl'
			# path = os.path.join(OUTPUT_FOLDER, filename)
			# with open(path, 'wb') as f:
			# 	pickle.dump(scaler_y, f)


	# Create the sliding windows
	X_train, y_train = split_sequence(X_train, y_train, seq_len, 1)
	X_val, y_val = split_sequence(X_val, y_val, seq_len, 1)
	X_test, y_test = split_sequence(X_test, y_test, seq_len, 1)

	tensors = (X_train, y_train, X_val, y_val, X_test, y_test)
	for tensor, filename in zip(tensors, OUTPUT_DATA):
		if scaler and (scale_y or filename.startswith('X_')):
			filename = filename + '_' + scaler
		filename += '_{}.pth'.format(seq_len)

		path = os.path.join(PROCESSED_DATA_PATH, filename)
		torch.save(tensor, path)

class ElectricityDataset(torch.utils.data.Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

	def get_all_data(self, num_samples=-1):
		if num_samples <= 0:
			features = self.X
		else:
			perm = torch.randperm(len(self))[:num_samples]
			features = self.X[perm]
		features = torch.tensor(features)
		return torch.permute(features, (0, 2, 1))

def get_data(scaler=None, scale_y=False, seq_len=SEQ_LEN):
	data = []
	for filename in OUTPUT_DATA:
		if scaler and (scale_y or filename.startswith('X_')):
			filename = filename + '_' + scaler
		filename += '_{}.pth'.format(seq_len)

		path = os.path.join(PROCESSED_DATA_PATH, filename)
		if not os.path.exists(path):
			prepare_dataset(scaler=scaler, scale_y=scale_y, seq_len=seq_len)
		data.append(torch.load(path))

	# if scaler:
	# 	filename = OUTPUT_SCALERS[0] + '_' + scaler + '_' + str(seq_len) + '.pkl'
	# 	path = os.path.join(PROCESSED_DATA_PATH, filename)
	# 	with open(path, 'rb') as f:
	# 		data.append(pickle.load(f))
	#
	# 	if scale_y:
	# 		filename = OUTPUT_SCALERS[1] + '_' + scaler + '_' + str(seq_len) + '.pkl'
	# 		path = os.path.join(PROCESSED_DATA_PATH, filename)
	# 		with open(path, 'rb') as f:
	# 			data.append(pickle.load(f))
	# 	else:
	# 		data.append(None)
	return tuple(data)

def get_datasets(scaler='std', scale_y=True, seq_len=-1):
	os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

	if seq_len < 0:
		seq_len = SEQ_LEN

	data = list(get_data(scaler=scaler, scale_y=scale_y, seq_len=seq_len))

	train_dataset = ElectricityDataset(data[0], data[1])
	val_dataset = ElectricityDataset(data[2], data[3])
	test_dataset = ElectricityDataset(data[4], data[5])

	return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
	get_datasets()
import torch
assert torch.cuda.is_available()

import numpy as np

from audio_classification.gtzan_config import *

def get_features_and_labels(seed=None):
    if os.path.exists(GTZAN_FEATURES_CACHE_PATH):
        with open(GTZAN_FEATURES_CACHE_PATH, 'rb') as gtzan_file:
            cached_data = pickle.load(gtzan_file)
        all_features = cached_data['all_features']
        all_labels = cached_data['all_labels']
    else:
        raise Exception("Tensorflow is required for this operation")
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    if seed:
        per = np.random.RandomState(seed=seed).permutation(len(all_features))
        all_features = all_features[per]
        all_labels = all_labels[per]

    val_split = int((1 - GTZAN_VIT_VAL_RATIO - GTZAN_VIT_TEST_RATIO) * all_features.shape[0])
    test_split = int((1 - GTZAN_VIT_TEST_RATIO) * all_features.shape[0])
    train_features = all_features[:val_split]
    train_labels = all_labels[:val_split]
    val_features = all_features[val_split:test_split]
    val_labels = all_labels[val_split:test_split]
    test_features = all_features[test_split:]
    test_labels = all_labels[test_split:]
    return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)

class TorchGTZANDataset(torch.utils.data.Dataset):
    def __init__(self, split, seed):
        self.split = split
        (train_features, train_labels), \
        (val_features, val_labels), \
        (test_features, test_labels) = get_features_and_labels(seed=seed)
        if split == 'train':
            self.features = train_features
            self.labels = train_labels
        elif split == 'val':
            self.features = val_features
            self.labels = val_labels
        else:
            self.features = test_features
            self.labels = test_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def get_all_data(self, num_samples=-1):
        if num_samples <= 0:
            features = self.features
        else:
            perm = torch.randperm(len(self))[:num_samples]
            features = self.features[perm]
        features = torch.tensor(features)
        return torch.permute(features, (0, 2, 1))

DATASET_FOLDER = "audio_classification/gtzan_datasets"
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def get_dataset(split, seed):
    path = os.path.join(DATASET_FOLDER, split+"_"+str(seed)+".pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        dataset = TorchGTZANDataset(split, seed)
        os.makedirs(DATASET_FOLDER, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset
# Tensorboard must be imported before
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

import torch
import torch.optim as optim
import torch.nn as nn
assert torch.cuda.is_available()

import pickle
import argparse
import gdown
import urllib.request
import librosa
import math
import numpy as np
import pandas as pd
import sys
from audioread import NoBackendError
import random

from config import get_args_parser
from preprocess_sound import preprocess_sound
from gtzan_config import *
from models import NonCoVisionTransformer, CoVisionTransformer, CoNystromVisionTransformer, NonCoNystromVisionTransformer
#from audio_classification.nystromformer.nystromformer import Nystromformer, ContinualNystromformer

# ROOT_DIR = '.'
# os.chdir(ROOT_DIR)

# Select a single GPU to perform the training
SELECTED_GPUS = ["0"]
os.environ['CUDA_VISIBLE_DEVICES'] = SELECTED_GPUS[0]

# Configure  GPUS
tf.get_logger().setLevel('INFO')
assert len(tf.config.list_physical_devices('GPU')) > 0
GPUS = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)

def VGGish(load_weights=True, weights='audioset',
           input_tensor=None, input_shape=AUDIO_SIZE,
           out_dim=128, include_top=True, pooling='avg'):
    if weights not in {'audioset', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `audioset` '
                         '(pre-training on audioset).')
    if input_tensor is None:
        aud_input = tf.keras.layers.Input(shape=input_shape, name='input_1')
    else:
        aud_input = input_tensor

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

    if include_top:
        x = tf.keras.layers.Flatten(name='flatten_')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
        x = tf.keras.layers.Dense(out_dim, activation='relu', name='vggish_fc2')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.models.Model(aud_input, x, name='VGGish')

    if load_weights:
        if weights == 'audioset':
            if include_top:
                model.load_weights(VGGISH_FOLDER+'/vggish_audioset_weights.h5')
            else:
                model.load_weights(VGGISH_FOLDER+'/vggish_audioset_weights_without_fc2.h5')
        else:
            print("failed to load weights")

    return model

def correct_waveform_size(waveform):
    correct_size = GTZAN_LENGTH * GTZAN_SAMPLING_RATE
    if waveform.shape[0] < correct_size:
        zero_padding = np.zeros(correct_size - waveform.shape[0])
        waveform = np.concatenate([waveform, zero_padding])
    elif waveform.shape[0] > correct_size:
        waveform = waveform[:correct_size]
    return waveform

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def get_waveforms(waveform):
    correct_waveform_size(waveform)
    waveforms = []
    for i in drange(0, GTZAN_LENGTH, GTZAN_SUB_HOP):
        start_index = int(i * GTZAN_SUB_HOP * GTZAN_SAMPLING_RATE)
        end_index = start_index + int(GTZAN_SUB_LENGTH * GTZAN_SAMPLING_RATE)
        sub_waveform = waveform[start_index:end_index]
        waveforms.append(sub_waveform)
    return waveforms

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_waveforms_and_labels():
    csv_data = pd.read_csv(GTZAN_CSV_PATH)
    string_labels = csv_data.label.unique()
    csv_data['num_label'] = csv_data.apply(lambda row: np.where(string_labels == row['label'])[0][0], axis=1)
    all_waveforms = []
    all_labels = []
    for index, row in csv_data.iterrows():
        sys.stdout.write('\rWaveform %d' % (index + 1))
        sys.stdout.flush()
        waveform_path = os.path.join(GTZAN_WAVEFORM_DIR, row['label'], row['filename'])
        try:
            waveform, _ = librosa.load(waveform_path, sr=GTZAN_SAMPLING_RATE)
        except NoBackendError:  # one file in the dataset is known to be corrupt
            print('Skipping corrupt file: %s' % waveform_path)
            continue
        waveforms = get_waveforms(waveform)
        label = row['num_label']
        all_waveforms.append(waveforms)
        all_labels.append(label)
    print()  # newline
    all_waveforms = np.array(all_waveforms)
    all_labels = tf.keras.utils.to_categorical(np.array(all_labels))
    all_waveforms, all_labels = unison_shuffled_copies(all_waveforms, all_labels)
    return all_waveforms, all_labels

def get_spectrograms_and_labels():
    if os.path.exists(GTZAN_SPECTROGRAMS_CACHE_PATH):
        with open(GTZAN_SPECTROGRAMS_CACHE_PATH, 'rb') as gtzan_file:
            cached_data = pickle.load(gtzan_file)
        all_spectrograms = cached_data['all_spectrograms']
        all_labels = cached_data['all_labels']
    else:
        all_waveforms, all_labels = get_waveforms_and_labels()
        all_spectrograms = []
        for index, waveforms in enumerate(all_waveforms):
            sys.stdout.write('\rSpectrogram %d/%d' % (index + 1, len(all_waveforms)))
            sys.stdout.flush()
            spectrograms = []
            for waveform in waveforms:
                spectrogram = preprocess_sound(waveform, GTZAN_SAMPLING_RATE)
                spectrogram = np.moveaxis(spectrogram, 0, -1)
                spectrograms.append(spectrogram)
            all_spectrograms.append(spectrograms)
        print()  # newline
        with open(GTZAN_SPECTROGRAMS_CACHE_PATH, 'wb') as cache_file:
            pickle.dump({
                'all_spectrograms': all_spectrograms,
                'all_labels': all_labels,
            }, cache_file, protocol=4)
    return all_spectrograms, all_labels

class TFGTZANSequence(tf.keras.utils.Sequence):
    def __init__(self, split, batch_size):
        self.split = split
        self.batch_size = batch_size
        all_spectrograms, all_labels = get_spectrograms_and_labels()
        all_spectrograms = np.array(all_spectrograms)
        all_labels = np.array(all_labels)
        reshaped_spectrograms = np.reshape(
            all_spectrograms,
            (
                all_spectrograms.shape[0] * all_spectrograms.shape[1],
                all_spectrograms.shape[2],
                all_spectrograms.shape[3],
                all_spectrograms.shape[4]
            )
        )
        reshaped_labels = np.repeat(all_labels, all_spectrograms.shape[1], axis=0)
        val_split = int((1 - GTZAN_VGGISH_VAL_RATIO - GTZAN_VGGISH_TEST_RATIO) * reshaped_spectrograms.shape[0])
        test_split = int((1 - GTZAN_VGGISH_TEST_RATIO) * reshaped_spectrograms.shape[0])
        if self.split == 'train':
            self.spectrograms = reshaped_spectrograms[:val_split]
            self.labels = reshaped_labels[:val_split]
        elif self.split == 'val':
            self.spectrograms = reshaped_spectrograms[val_split:test_split]
            self.labels = reshaped_labels[val_split:test_split]
        else:
            self.spectrograms = reshaped_spectrograms[test_split:]
            self.labels = reshaped_labels[test_split:]
        self.random_permutation = np.random.permutation(len(self.spectrograms))

    def __len__(self):
        return math.ceil(len(self.spectrograms) / self.batch_size)

    def on_epoch_end(self):
        self.random_permutation = np.random.permutation(len(self.spectrograms))

    def __getitem__(self, index):
        if self.split == 'train':
            return self.spectrograms[self.random_permutation[index * self.batch_size: (index + 1) * self.batch_size]], \
                   self.labels[self.random_permutation[index * self.batch_size: (index + 1) * self.batch_size]]
        else:
            return self.spectrograms[index * self.batch_size: (index + 1) * self.batch_size], \
                   self.labels[index * self.batch_size: (index + 1) * self.batch_size]

def softmax_to_one_hot(softmax):
    one_hot = np.zeros(softmax.shape)
    for i in range(len(softmax)):
        one_hot[i, np.argmax(softmax[i])] = 1
    return one_hot

def get_majority_voting_accuracy(model):
    all_spectrograms, all_labels = get_spectrograms_and_labels()
    all_spectrograms = np.array(all_spectrograms)
    all_labels = np.array(all_labels)
    val_split = int((1 - GTZAN_VGGISH_VAL_RATIO - GTZAN_VGGISH_TEST_RATIO) * all_spectrograms.shape[0])
    test_split = int((1 - GTZAN_VGGISH_TEST_RATIO) * all_spectrograms.shape[0])
    spectrograms = all_spectrograms[test_split:]
    labels = all_labels[test_split:]
    correct_count = 0
    for clip_index in range(len(spectrograms)):
        clip_spectrograms = spectrograms[clip_index]
        clip_label = labels[clip_index]
        clip_predictions = model(clip_spectrograms)
        clip_majority_vote = np.argmax(np.sum(softmax_to_one_hot(clip_predictions), axis=0))
        if clip_majority_vote == np.argmax(clip_label):
            correct_count += 1
    return correct_count / len(labels)

def tf_train(config, seed=None):
    if seed:
        tf.keras.utils.set_random_seed(seed)

    tf.keras.backend.clear_session()

    test_sequence = TFGTZANSequence('test', config.batch_size)
    if config.retrain or not os.path.exists(FINE_TUNED_VGGISH_PATH):
        train_sequence = TFGTZANSequence('train', config.batch_size)
        val_sequence = TFGTZANSequence('val', config.batch_size)
        vggish = VGGish(
            include_top=True,
            load_weights=True,
            input_shape=AUDIO_SIZE
        )
        output = vggish.get_layer('vggish_fc2').output
        output = tf.keras.layers.Dense(
            units=10,
            activation='sigmoid'
        )(output)
        model = tf.keras.models.Model(
            vggish.get_layer(index=0).input,
            outputs=output
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.6,
            patience=2,
            verbose=1,
            mode='max',
            min_lr=1e-7
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            mode='max'
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            FINE_TUNED_VGGISH_PATH,
            monitor='val_accuracy',
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
            mode='max',
            save_freq='epoch'
        )
        history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=config.epochs,
            shuffle=True,
            callbacks=[
                lr_reduce,
                early_stop,
                checkpoint
            ],
            verbose=1
        )
    else:
        model = tf.keras.models.load_model(FINE_TUNED_VGGISH_PATH)
    test_accuracy = model.evaluate(test_sequence)[1] * 100
    majority_accuracy = get_majority_voting_accuracy(model) * 100
    print('Test Acc: %.2f, Majority Acc: %.2f' % (test_accuracy, majority_accuracy))

def get_tf_params(model):
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    for string in string_list:
        if string.startswith('Trainable params:'):
            return int(string.split()[-1].replace(',', ''))
    return None

def get_tf_flops(model):
    """
    from https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-768977280
    """
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops / 2

def get_features_and_labels(seed=None):
    if os.path.exists(GTZAN_FEATURES_CACHE_PATH):
        with open(GTZAN_FEATURES_CACHE_PATH, 'rb') as gtzan_file:
            cached_data = pickle.load(gtzan_file)
        all_features = cached_data['all_features']
        all_labels = cached_data['all_labels']
    else:
        all_spectrograms, all_labels = get_spectrograms_and_labels()
        fine_tuned_vggish = tf.keras.models.load_model(FINE_TUNED_VGGISH_PATH)
        model = tf.keras.models.Model(
            fine_tuned_vggish.get_layer(index=0).input,
            outputs=fine_tuned_vggish.get_layer('vggish_fc2').output
        )
        all_features = []
        for index, spectrograms in enumerate(all_spectrograms):
            sys.stdout.write('\rFeature %d/%d' % (index + 1, len(all_spectrograms)))
            sys.stdout.flush()
            features = []
            for spectrogram in spectrograms:
                feature = model(np.expand_dims(spectrogram, axis=0))
                features.append(tf.squeeze(feature))
            all_features.append(features)
        print()  # newline
        with open(GTZAN_FEATURES_CACHE_PATH, 'wb') as cache_file:
            pickle.dump({
                'all_features': all_features,
                'all_labels': all_labels,
            }, cache_file, protocol=4)
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


INPUT_DIM = 128
SEQ_LEN = 120

def calculate_accuracy(model, data_loader):
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            features, labels = data
            if features.size(0) < data_loader.batch_size:
                continue # Skip the last batch to avoid any problems
            features = torch.permute(features, (0, 2, 1))
            features = features.to("cuda:"+str(SELECTED_GPUS[0]))
            labels = labels.to("cuda:"+str(SELECTED_GPUS[0]))
            predicted_labels = model(features)
            predicted_labels = torch.squeeze(predicted_labels, dim=-1)
            if predicted_labels.ndim > 2:
                predicted_labels = predicted_labels[:, :, -1]
            correct_count += torch.sum(torch.argmax(predicted_labels, dim=1) == torch.argmax(labels, dim=1))
            total_count += len(labels)
    accuracy = correct_count / total_count * 100  # percent
    return accuracy

MODEL_FOLDER = "saved_models"
def get_model_path(config, folder=MODEL_FOLDER, fixed_landmarks=False, freeze_weights=False, extension="pth"):
    if config.model in ['base', 'base_continual']:
        return '%s/%s_%d_layers_seeds_%d_%d.%s' % (
            folder,
            config.model,
            config.num_layers,
            config.model_seed,
            config.data_seed,
            extension
        )
    elif fixed_landmarks:
        fit_layer_epochs = str(config.fit_layer_epochs).replace('[', '-').replace(']', '-')
        return '%s/%s_%d_layers_%d_landmarks_%s_%d_seeds_%d_%d.%s' % (
            folder,
            config.model,
            config.num_layers,
            config.num_landmarks,
            fit_layer_epochs,
            freeze_weights,
            config.model_seed,
            config.data_seed,
            extension
        )
    else:
        return '%s/%s_%d_layers_%d_landmarks_seeds_%d_%d.%s' % (
            folder,
            config.model,
            config.num_layers,
            config.num_landmarks,
            config.model_seed,
            config.data_seed,
            extension,
        )

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def compute_test_accuracy(model, test_loader):
    # if config.model == "base_continual":
    #     if config.num_layers == 1:
    #         model.call_mode = "forward_steps"
    #     else:
    #         model[3][0][0][0].self_attn.call_mode = "forward_steps"
    # if config.model == "continual_nystrom":
    #     if config.num_layers == 1:
    #         model[3][0].self_attn.forward_mode = "forward_steps"
    #     else:
    #         model[3][0][0][0].self_attn.forward_mode = "forward_steps"
    test_accuracy = calculate_accuracy(model, test_loader)
    return test_accuracy


DATASET_FOLDER = "gtzan_datasets"
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

def fix_landmarks(model, dataset, config, freeze_weights=True, layer_number=0, kmeans_attempts=10):
    print("Computing and fixing landmarks for encoder layer number "+str(layer_number)+"...")

    seed = config.model_seed
    total_layers = config.num_layers

    assert layer_number < total_layers

    features = dataset.features
    features = torch.tensor(features)
    features = torch.permute(features, (0, 2, 1))
    features = features.to("cuda:"+str(SELECTED_GPUS[0]))

    # TODO: Not very good code
    if config.num_layers == 1:
        nystrom_module = model[3][0][1]
    elif layer_number == 0:
        nystrom_module = model[3][0][0][0][1]
    else:
        nystrom_module = list(model[3][0][layer_number].children())[0][0][1]

    if freeze_weights:
        for param in model[:3].parameters():
            param.requires_grad = False
        if layer_number > 0 and total_layers > 1:
            for param in model[3][0][:layer_number].parameters():
                param.requires_grad = False

        linear_q = nystrom_module.W_q
        linear_k = nystrom_module.W_k
        for linear_q_elem in linear_q:
            for param in linear_q_elem.parameters():
                param.requires_grad = False
        for linear_k_elem in linear_k:
            for param in linear_k_elem.parameters():
                param.requires_grad = False

    with torch.no_grad():
        features = model[:3](features)
        if layer_number > 0 and total_layers > 1:
            features = model[3][0][:layer_number](features)

    # Add the new landmarks
    nystrom_module.fix_landmarks(features, kmeans_attempts=kmeans_attempts, seed=seed)
    print("Finished fixing landmarks for encoder layer " + str(layer_number))

def train_fixed_landmarks(model, config, out_file, train_dataset, optimizer, criterion, train_loader, val_loader, test_loader, writer, best_val_accuracy, freeze_weights=True):
    total_epochs = config.epochs + sum(config.fit_layer_epochs)
    cum_epoch = config.epochs
    for num_layer, num_epochs_fit in enumerate(config.fit_layer_epochs):
        fix_landmarks(model, train_dataset, config, freeze_weights=freeze_weights, layer_number=num_layer)
        for _ in range(num_epochs_fit):
            train_accuracy, best_val_accuracy = train_one_epoch(model, config, cum_epoch, total_epochs, optimizer,
                                                                criterion, train_loader, val_loader, writer,
                                                                best_val_accuracy, fixed_landmarks=True, freeze_weights=freeze_weights)
            cum_epoch += 1

    # Reload best model for test
    best_path = get_model_path(config, fixed_landmarks=True, freeze_weights=freeze_weights)
    if not os.path.exists(best_path):
        best_path = get_model_path(config, fixed_landmarks=False)
    model.load_state_dict(torch.load(best_path, map_location="cuda")) # .to("cuda:"+str(SELECTED_GPUS[0])))

    test_accuracy = compute_test_accuracy(model, test_loader)

    output_content = {
        "config": config,
        "freeze_weights": freeze_weights,
        "train_accuracy": train_accuracy,
        "val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
    }
    with open(out_file, 'wb') as f:
        pickle.dump(output_content, f)

def train_one_epoch(model, config, epoch_number, total_epochs, optimizer, criterion, train_loader, val_loader, writer, best_val_accuracy, fixed_landmarks=False, freeze_weights=False):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # load data
        features, labels = data
        features = torch.permute(features, (0, 2, 1))
        features = features.to("cuda:"+str(SELECTED_GPUS[0]))
        labels = labels.to("cuda:"+str(SELECTED_GPUS[0]))

        # train the model
        optimizer.zero_grad()

        # with torch.autograd.detect_anomaly(check_nan=True):
        predicted_labels = model(features)
        predicted_labels = torch.squeeze(predicted_labels, dim=-1)
        loss = criterion(predicted_labels, labels)
        if torch.isnan(loss):
            pass
        loss.backward()
        optimizer.step()

        # update training metrics
        running_loss += loss.item()

    train_accuracy = calculate_accuracy(model, train_loader)
    val_accuracy = calculate_accuracy(model, val_loader)

    writer.add_scalar("Loss/train", running_loss, epoch_number)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch_number)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch_number)

    improved = False
    if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        improved = True
        torch.save(model.state_dict(), get_model_path(config, fixed_landmarks=fixed_landmarks, freeze_weights=freeze_weights))

    print('Epoch: %d/%d; Loss: %.2e; Train Acc: %.2f; Val Acc: %.2f%s' % (
        epoch_number + 1,
        total_epochs,
        running_loss,
        train_accuracy,
        val_accuracy,
        '; saved' if improved else ''
    ))

    return train_accuracy, best_val_accuracy

def torch_train(config, folder="raw_results"):
    print("\n\n\n" + str(config))

    assert config.model in ["base", "base_continual", "nystromformer", "continual_nystrom"]
    assert config.freeze_weights in ["both", "true", "false"]
    assert config.num_layers >= len(config.fit_layer_epochs)

    g = torch.Generator()
    g.manual_seed(config.data_seed)

    train_dataset = get_dataset('train', config.data_seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        #num_workers=config.batch_size,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_dataset = get_dataset('val', config.data_seed)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        #num_workers=config.batch_size,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_dataset = get_dataset('test', config.data_seed)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        #num_workers=config.batch_size'],
        worker_init_fn=seed_worker,
        generator=g
    )

    if config.model_seed:
        torch.manual_seed(config.model_seed)
        random.seed(config.model_seed)
        np.random.seed(config.model_seed)

    # create and load mode
    match config.model:
        case "base":
            model = NonCoVisionTransformer(
                sequence_len=SEQ_LEN,
                input_dim=INPUT_DIM,
                embedding_dim=192,
                attn_ff_hidden_dim=192,
                out_dim=10,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
            )
        case "base_continual":
            model = CoVisionTransformer(
                sequence_len=SEQ_LEN,
                input_dim=INPUT_DIM,
                embedding_dim=192,
                attn_ff_hidden_dim=192,
                out_dim=10,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
            )
        case "nystromformer":
            model = NonCoNystromVisionTransformer(
                sequence_len=SEQ_LEN,
                input_dim=INPUT_DIM,
                embedding_dim=192,
                attn_ff_hidden_dim=192,
                out_dim=10,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
                device="cuda:" + str(SELECTED_GPUS[0]),
                num_landmarks=config.num_landmarks,
            )
        case "continual_nystrom":
            model = CoNystromVisionTransformer(
                sequence_len=SEQ_LEN,
                input_dim=INPUT_DIM,
                embedding_dim=192,
                attn_ff_hidden_dim=192,
                out_dim=10,
                num_heads=16,
                num_layers=config.num_layers,
                dropout_rate=0.1,
                batch_size=config.batch_size,
                device="cuda:"+str(SELECTED_GPUS[0]),
                num_landmarks=config.num_landmarks,
            )

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to("cuda:"+str(SELECTED_GPUS[0]))

    # optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss includes the softmax

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    writer = SummaryWriter()

    total_epochs = config.epochs + sum(config.fit_layer_epochs)

    # training loop
    best_val_accuracy = 0.0
    for epoch in range(config.epochs):
        train_accuracy, best_val_accuracy = train_one_epoch(model, config, epoch, total_epochs, optimizer, criterion, train_loader, val_loader,
                                            writer, best_val_accuracy, fixed_landmarks=False)

    if config.model not in ['base', 'base_continual']:
        # Copy current state of the model
        torch.save(model.state_dict(), get_model_path(config, fixed_landmarks=False, extension="ptht"))

    # Reload best model for test
    model.load_state_dict(torch.load(get_model_path(config, fixed_landmarks=False), map_location="cuda"))

    test_accuracy = compute_test_accuracy(model, test_loader)

    output_content = {
        "config": config,
        "freeze_weights": None,
        "train_accuracy": train_accuracy,
        "val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
    }
    dump_file_path = get_model_path(config, folder, False, False, extension="pkl")
    os.makedirs(folder, exist_ok=True)
    with open(dump_file_path, 'wb') as f:
        pickle.dump(output_content, f)

    if config.model not in ['base', 'base_continual']:
        model.load_state_dict(torch.load(get_model_path(config, fixed_landmarks=False, freeze_weights=False, extension="ptht"), map_location="cuda"))


        if config.freeze_weights in ["true", "both"] and config.fit_layer_epochs != []:
            train_fixed_landmarks(model, config,
                                  get_model_path(config, folder, True, True, "pkl"),
                                  train_dataset, optimizer, criterion, train_loader, val_loader,
                                  test_loader, writer, best_val_accuracy, freeze_weights=True)

        if config.freeze_weights == "both":
            model.load_state_dict(torch.load(get_model_path(config, fixed_landmarks=False, freeze_weights=False, extension="ptht"), map_location="cuda"))

        if config.freeze_weights in ["false", "both"] and config.fit_layer_epochs != []:
            train_fixed_landmarks(model, config,
                                  get_model_path(config, folder, True, False, "pkl"),
                                  train_dataset, optimizer, criterion, train_loader, val_loader,
                                  test_loader, writer, best_val_accuracy, freeze_weights=False)

    writer.flush()
    writer.close()
    return model, train_accuracy, best_val_accuracy, test_accuracy

def std(lst):
    mean = sum(lst) / len(lst)
    variance = sum([((x - mean) ** 2) for x in lst]) / len(lst)
    return variance ** 0.5

def evaluate_config(config, num_seeds=5, filename="results.txt", flops=False, params=False):

    for data_seed in range(num_seeds):
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        flops_list = []
        params_list = []


        config["data_seed"] = data_seed
        for model_seed in range(num_seeds):
            config["model_seed"] = model_seed
            print(config["model"], "model with "+str(config["num_layers"])+" layers. Seed: "+str(config["model_seed"]))
            model, train_accuracy, val_accuracy, test_accuracy = torch_train(config)
            #flops, params = get_flops_and_params(model, config)
            # TODO: Remove later
            flops = 0
            params = 0
            print(test_accuracy, flops, params)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            test_accuracies.append(test_accuracy)
            flops_list.append(flops)
            params_list.append(params)

        with open(filename, 'a') as f:

            f.write("---------------------\n")
            f.write("Data seed "+str(data_seed))
            f.write(config["model"]+" model with " + str(config["num_layers"]) + " layers\n")
            f.write("Mean:")
            f.write("\t train_acc: %.2f" % (sum(train_accuracies) / len(train_accuracies)))
            f.write("\t val_acc: %.2f" % (sum(val_accuracies)/len(val_accuracies)))
            f.write("\t test_acc: %.2f" % (sum(test_accuracies)/len(test_accuracies)))
            if flops:
                f.write("\t flops: %.2f" % (sum(flops_list)/len(flops_list)))
            if params:
                f.write("\t params: %.2f" % (sum(params_list)/len(params_list)))
            f.write("\n")

            f.write("Std:")
            f.write("\t train_acc: %.2f" % (std(train_accuracies)))
            f.write("\t val_acc: %.2f" % (std(val_accuracies)))
            f.write("\t test_acc: %.2f" % (std(test_accuracies)))
            if flops:
                f.write("\t flops: %.2f" % (std(flops_list)))
            if params:
                f.write("\t params: %.2f" % (std(params_list)))
            f.write("\n")

    # print("---------------------")
    #
    # print("Individual values:")
    # for seed in range(len(val_accuracies)):
    #     print("Seed", str(seed))
    #     print("\t", str(val_accuracies[seed]), str(test_accuracies[seed]), str(flops_list[seed]), str(params_list[seed]))

    print("End of evaluation ---------------------------------")


if __name__ == "__main__":
    os.makedirs(VGGISH_FOLDER, exist_ok=True)
    os.makedirs(GTZAN_CACHE_FOLDER, exist_ok=True)

    # Download weights
    download_list = [
        (
            'https://drive.google.com/u/0/uc?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6',
            VGGISH_FOLDER+'/vggish_audioset_weights.h5'
        ),
        (
            'https://drive.google.com/u/0/uc?id=16JrWEedwaZFVZYvn1woPKCuWx85Ghzkp',
            VGGISH_FOLDER+'/vggish_audioset_weights_without_fc2.h5'
        ),
        (
            'https://drive.google.com/u/0/uc?id=1QS0EQzLTF7IgMeS1n0g2BbMIr_a3cCA8',
            VGGISH_FOLDER+'/fine_tuned_vggish.h5'
        )
    ]
    for url, file_path in download_list:
        if not os.path.exists(file_path):
            if 'drive.google.com' in url:
                gdown.download(
                    url,
                    file_path,
                    quiet=False
                )
            else:
                urllib.request.urlretrieve(url, file_path)


    # Solving this issue: https://github.com/DTaoo/VGGish/issues/11
    params_path = 'vggish_params.py'
    with open(params_path, 'rt') as read_file:
        text = read_file.read()
    with open(params_path, 'wt') as write_file:
        write_file.write(text.replace('496', '96').replace('4.96', '0.96'))

    parser = argparse.ArgumentParser(
        "Audio classification python program", parents=[get_args_parser()]
    )
    config = parser.parse_args()

    for data_seed in range(5):
        config.data_seed = data_seed
        for model_seed in range(5):
            config.model_seed = model_seed
            for model in ["base", "base_continual"]:
                config.model = model
                for num_layers in [1, 2]:
                    config.num_layers = num_layers
                    torch_train(config)
            for model in ["nystromformer", "continual_nystrom"]:
                config.model = model
                for num_landmarks in [2, 4, 8, 16, 32, 64]:
                    config.num_landmarks = num_landmarks

                    config.num_layers = 1
                    config.fit_layer_epochs = [25]
                    torch_train(config)

                    config.num_layers = 2
                    config.fit_layer_epochs = [25, 25]
                    torch_train(config)

        config.fit_layer_epochs = []

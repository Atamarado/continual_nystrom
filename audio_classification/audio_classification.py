# Tensorboard must be imported before
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

import torch
import torch.optim as optim
import torch.nn as nn
assert torch.cuda.is_available()

import gdown
import urllib.request
import librosa
import math
import numpy as np
import pandas as pd
import sys
from ptflops import get_model_complexity_info # TODO: Check whether to use the version in continual_transformers
from audioread import NoBackendError
import random


from preprocess_sound import preprocess_sound
from gtzan_config import *
from models import NonCoVisionTransformer, CoVisionTransformer, CoNystromVisionTransformer, NonCoNystromVisionTransformer
#from audio_classification.nystromformer.nystromformer import Nystromformer, ContinualNystromformer

# ROOT_DIR = '.'
# os.chdir(ROOT_DIR)

# Select a single GPU to perform the training
SELECTED_GPUS = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_number) for gpu_number in SELECTED_GPUS])

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

    test_sequence = TFGTZANSequence('test', config['batch_size'])
    if config['retrain'] or not os.path.exists(FINE_TUNED_VGGISH_PATH):
        train_sequence = TFGTZANSequence('train', config['batch_size'])
        val_sequence = TFGTZANSequence('val', config['batch_size'])
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
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
            epochs=config['epochs'],
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
            features = torch.permute(features, (0, 2, 1))
            features = features.cuda()
            labels = labels.cuda()
            predicted_labels = model(features)
            predicted_labels = torch.squeeze(predicted_labels, dim=-1)
            correct_count += torch.sum(torch.argmax(predicted_labels, dim=1) == torch.argmax(labels, dim=1))
            total_count += len(labels)
    accuracy = correct_count / total_count * 100  # percent
    return accuracy

MODEL_FOLDER = "saved_models"
def get_model_path(config):
    return '%s/%s_%d_layers_%s_%s_%s.pth' % (
        MODEL_FOLDER,
        config['model'],
        config['num_layers'],
        config['version'],
        config['model_seed'],
        config['data_seed']
    )

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset


def torch_train(config):
    assert config["model"] in ["base", "base_continual", "nystromformer", "continual_nystrom"]

    g = torch.Generator()
    g.manual_seed(config["data_seed"])

    train_dataset = get_dataset('train', config["data_seed"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        #num_workers=config['batch_size'],
        worker_init_fn=seed_worker,
        generator=g
    )
    val_dataset = get_dataset('val', config["data_seed"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        #num_workers=config['batch_size'],
        worker_init_fn=seed_worker,
        generator=g
    )
    test_dataset = get_dataset('test', config["data_seed"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        #num_workers=config['batch_size'],
        worker_init_fn=seed_worker,
        generator=g
    )

    if config['model_seed']:
        torch.manual_seed(config['model_seed'])
        random.seed(config['model_seed'])
        np.random.seed(config['model_seed'])

    # create and load mode
    match config["model"]:
        case "base":
            model = NonCoVisionTransformer(
                sequence_len=SEQ_LEN,
                input_dim=INPUT_DIM,
                embedding_dim=192,
                attn_ff_hidden_dim=192,
                out_dim=10,
                num_heads=16,
                num_layers=config['num_layers'],
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
                num_layers=config['num_layers'],
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
                num_layers=config['num_layers'],
                dropout_rate=0.1,
            )
        case "continual_nystrom":
            model = CoNystromVisionTransformer(
                sequence_len=SEQ_LEN,
                input_dim=INPUT_DIM,
                embedding_dim=192,
                attn_ff_hidden_dim=192,
                out_dim=10,
                num_heads=16,
                num_layers=config['num_layers'],
                dropout_rate=0.1,
                batch_size=config['batch_size'],
                device="cuda:"+str(SELECTED_GPUS[0])
            )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()

    # optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss includes the softmax

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    writer = SummaryWriter()

    # training loop
    best_val_accuracy = 0.0
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # load data
            features, labels = data
            features = torch.permute(features, (0, 2, 1))
            features = features.cuda()
            labels = labels.cuda()

            # train the model
            optimizer.zero_grad()

            #with torch.autograd.detect_anomaly(check_nan=True):
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

        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        improved = False
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            improved = True
            torch.save(model.state_dict(), get_model_path(config))

        print('Epoch: %d/%d; Loss: %.2e; Train Acc: %.2f; Val Acc: %.2f%s' % (
            epoch + 1,
            config['epochs'],
            running_loss,
            train_accuracy,
            val_accuracy,
            '; saved' if improved else ''
        ))
    writer.flush()
    writer.close()

    # Reload best model for test
    model.load_state_dict(torch.load(get_model_path(config)))

    if config["model"] == "continual_nystrom":
        model[3].call_mode = "forward_steps"
    test_accuracy = calculate_accuracy(model, test_loader)

    return model, train_accuracy, best_val_accuracy, test_accuracy

def get_flops_and_params(model, config):
    if config['model'] in ["base_continual", "continual_nystrom"]:
        warm_up_input = torch.randn(1, INPUT_DIM, SEQ_LEN)
        model.to('cpu')
        assert next(model.parameters()).is_cuda == warm_up_input.is_cuda
        model.forward_steps(warm_up_input)  # Warm up model
        model.call_mode = "forward_step"
        flops, params = get_model_complexity_info(
            model, (INPUT_DIM,), as_strings=False, print_per_layer_stat=False
        )
    else:
        flops, params = get_model_complexity_info(
            model, (INPUT_DIM, SEQ_LEN), as_strings=False, print_per_layer_stat=False
        )
    return flops, params

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

    tf_config = {
        'batch_size': 64,
        'epochs': 100,
        'lr': 1e-4,
        'retrain': False,
    }
    tf_train(tf_config)

    head_model = tf.keras.models.load_model(FINE_TUNED_VGGISH_PATH)
    head_params = get_tf_params(head_model)
    head_flops = get_tf_flops(head_model)
    print('Head: params %.2fM; FLOPS %.2fM' % (head_params / 10 ** 6, head_flops / 10 ** 6))

    torch_config = {
        'batch_size': 32,
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'epochs': 50,
        'version': 'v5',
        'num_layers': 1,
        'model': 'nystromformer'
    }
    evaluate_config(torch_config, filename="results_conystrom.txt")

    torch_config["num_layers"] = 2
    evaluate_config(torch_config, filename="results_conystrom.txt")

    torch_config["num_layers"] = 1
    torch_config["model"] = "continual_nystrom"
    evaluate_config(torch_config, filename="results_conystrom.txt")

    torch_config["num_layers"] = 2
    evaluate_config(torch_config, filename="results_conystrom.txt")

        # torch_config = {
        #     'batch_size': 32,
        #     'lr': 1e-5,
        #     'weight_decay': 1e-4,
        #     'epochs': 50,
        #     'version': 'v5',
        #     'num_layers': 1,
        #     'model': 'continual',
        #     'seed': seed
        # }
        # continual_model, test_accuracy = torch_train(torch_config)
        # flops, params = get_flops_and_params(continual_model, torch_config)
        # print(test_accuracy, flops, params)

        # torch_config["model"] = "continual_nystrom"
        # continual_model, test_accuracy = torch_train(torch_config)
        # flops, params = get_flops_and_params(continual_model, torch_config)
        # print(test_accuracy, flops, params)

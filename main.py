import torch
import torch.nn as nn
assert torch.cuda.is_available()

import numpy as np
import random
from collections import OrderedDict
import re
import time
from tqdm import tqdm

import continual_dev as co

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from config import get_config
from audio_classification.gtzan_config import *
from models import get_audio_model
from thumos_models import get_thumos_model

from audio_classification.audio_classification import get_dataset as get_audio_dataset
from audio_classification.audio_classification import TorchGTZANDataset

from CoOadTR.dataset import TRNTHUMOSDataLayer
from CoOadTR import util as utl
from CoOadTR.utils import frame_level_map_n_cap

from dataset_electricity import get_datasets as get_datasets_electricity

from config import get_config

def forward_data(model, config, data, fill_batch=False, test_mode=False):
    features, out_target = get_data(data, config)
    if config.seq_len > 0:
        features = features[:, :, :config.seq_len]

    if fill_batch:
        # Append zero inputs for the last batch to avoid problems
        zero_inputs = config.batch_size - features.size(0)
        if zero_inputs > 0:
            pad_shape = (zero_inputs,) + tuple(features[0].shape)
            features = torch.cat([features, torch.zeros(pad_shape, device=features.device)], dim=0)

    out = model(features)

    if fill_batch and zero_inputs > 0:
        out = out[:-zero_inputs]

    if test_mode and config.model in CONTINUAL_MODELS:
        out = out[:, :, -1]

    match config.dataset:
        case 'gtzan' | 'thumos':
            out = torch.squeeze(out, dim=-1)
        case 'electricity':
            out = torch.squeeze(out)

    return out_target, out

def get_transformer_modules(model, config):
    modules = []
    match config.dataset:
        case 'gtzan' | 'electricity':
            if config.num_layers == 1:
                modules.append(model[2][0].self_attn)

            elif config.num_layers == 2:
                modules.append(model[2].layers[0][0].self_attn)
                modules.append(model[2].layers[1].fn[0].self_attn)
            else:
                raise Exception('Invalid number of layers')
        case 'thumos':
            if config.num_layers == 1:
                modules.append(model[3][0].self_attn)

            elif config.num_layers == 2:
                modules.append(model[3].layers[0][0].self_attn)
                modules.append(model[3].layers[1].fn[0].self_attn)
            else:
                raise Exception('Invalid number of layers')
        case _:
            raise NotImplementedError
    return modules

def clean_state(model, config):
    if config.model not in CONTINUAL_MODELS:
        return

    modules = get_transformer_modules(model, config)

    for module in modules:
        module.clean_state()
        module.stride_index = torch.tensor(-module.sequence_len)

CONTINUAL_MODELS = ['base_continual', 'continual_nystrom']
def get_transformer_costs(model, config, fixed_landmarks=False):
    modules = get_transformer_modules(model, config)

    flops = 0
    valley_mem_cost = 0
    peak_mem_cost = 0

    call_mode = "forward_steps" if config.model in CONTINUAL_MODELS else "forward"
    for module in modules:
        if fixed_landmarks:
            module.fixed_landmarks = True

        flops += module.flops(call_mode)
        v, p = module.mem_costs(call_mode)
        valley_mem_cost += v
        peak_mem_cost += p
    return flops, valley_mem_cost, peak_mem_cost

def get_dataset(split, config):
    match config.dataset:
        case 'gtzan':
            return get_audio_dataset(split, config.data_seed)
        case 'thumos':
            return TRNTHUMOSDataLayer(phase=split, args=config)
        case _:
            raise NotImplementedError
    return

def get_data_loaders(config):
    match config.dataset:
        case 'gtzan':
            g = torch.Generator()
            g.manual_seed(config.data_seed)
            train_dataset = get_dataset('train', config)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                # num_workers=config.batch_size,
                worker_init_fn=seed_worker,
                generator=g
            )

            test_dataset = get_dataset('test', config)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                # num_workers=config.batch_size'],
                worker_init_fn=seed_worker,
                generator=g
            )

            val_dataset = get_dataset('val', config)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                # num_workers=config.batch_size,
                worker_init_fn=seed_worker,
                generator=g
            )

        case 'thumos':
            dataset_train = get_dataset('train', config)
            dataset_val = get_dataset('test', config)

            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, config.batch_size, drop_last=True
            )

            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                pin_memory=True,
                num_workers=config.num_workers,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset_val,
                config.batch_size,
                sampler=sampler_val,
                drop_last=False,
                pin_memory=True,
                num_workers=config.num_workers,
            )

            test_loader = None

        case "electricity":
            g = torch.Generator()
            g.manual_seed(config.data_seed)
            train_dataset, val_dataset, test_dataset = get_datasets_electricity(seq_len=config.seq_len)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                # num_workers=config.batch_size,
                worker_init_fn=seed_worker,
                generator=g
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                # num_workers=config.batch_size'],
                worker_init_fn=seed_worker,
                generator=g
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                # num_workers=config.batch_size'],
                worker_init_fn=seed_worker,
                generator=g
            )
        case _:
            raise NotImplementedError
    return train_loader, val_loader, test_loader

def get_model(config, training_dataset=None):
    match config.dataset:
        case 'gtzan' | 'electricity':
            model = get_audio_model(config, training_dataset=training_dataset)
            model = model.to("cuda")
            return model
        case 'thumos':
            model = get_thumos_model(config)
            model = model.to("cuda")
            return model
        case _:
            raise NotImplementedError
    return

def get_optimizer(config, model):
    match config.dataset:
        case 'gtzan' | 'electricity':
            return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        case 'thumos':
            return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        case _:
            raise NotImplementedError
    return

def get_criterion(config):
    match config.dataset:
        case 'gtzan':
            return nn.CrossEntropyLoss()  # CrossEntropyLoss includes the softmax
        case 'thumos':
            loss_need = [
                "labels_encoder",
                # "labels_decoder",
            ]
            criterion = utl.SetCriterion(
                num_classes=config.numclass, losses=loss_need, args=config
            ).to(config.device)
            criterion = criterion.loss_labels
            return criterion
        case 'electricity':
            return nn.HuberLoss()
        case _:
            raise NotImplementedError
    return

def get_lr_scheduler(config, optimizer):
    match config.dataset:
        case 'gtzan' | 'electricity':
            return None
        case 'thumos':
            return torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
        case _:
            raise NotImplementedError
    return


criterion_huber = nn.HuberLoss()
criterion_MSE = nn.MSELoss()
criterion_MAE = nn.L1Loss()
def get_performance(config, preds, labels, performance=None):
    batch_performance = {}
    match config.dataset:
        case "gtzan":
            if preds.ndim > 2:
                preds = preds[:, :, -1]
            correct_count = torch.sum(torch.argmax(preds, dim=1) == torch.argmax(labels, dim=1))
            total_count = len(labels)

            batch_performance["accuracy"] = correct_count * 100 / total_count
        case "thumos":
            preds = np.asarray(preds.detach().cpu()).T[:21]
            labels = np.asarray(labels.detach().cpu()).T[:21]

            batch_performance["probs"] = [preds]
            batch_performance["labels"] = [labels]
        case 'electricity':
            batch_performance['huber_loss'] = criterion_huber(preds, labels) / len(preds)
            batch_performance['MSE_loss'] = criterion_MSE(preds, labels) / len(preds)
            batch_performance['MAE_loss'] = criterion_MAE(preds, labels) / len(preds)
        case _:
            raise NotImplementedError

    batch_performance["count"] = 1

    if performance is None:
        return batch_performance

    for key, value in batch_performance.items():
        performance[key] += value

    return performance

def normalize_performance(performance):
    if config.dataset == 'thumos':
        preds = np.concatenate(performance['probs'], axis=1)
        labels = np.concatenate(performance['labels'], axis=1)

        results = {"probs": preds, "labels": labels}
        map, _, mcap, _ = frame_level_map_n_cap(results)

        performance = {"map": map, "mcap": mcap}
    else:
        for key, value in performance.items():
            if key == "count":
                continue
            performance[key] = value / performance["count"]
        del performance["count"]
    return performance

def evaluate(model, data_loader, config, test_mode=False):
    model.eval()

    performance = None
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            out_target, out = forward_data(model, config, data, fill_batch=True, test_mode=test_mode)

            performance = get_performance(config, out, out_target, performance)
            torch.cuda.empty_cache()

    performance = normalize_performance(performance)
    return performance

def get_model_path(config, fixed_landmarks=False, freeze_weights=False, extension="pth", model_weights=True, log=False):
    folder = config.log_folder if log else config.out_folder
    if config.model in ['base', 'base_continual']:
        model_type = config.model
        return '%s/%s_%d_layers_seeds_%d_%d_%d%s.%s' % (
            folder,
            model_type,
            config.num_layers,
            config.model_seed,
            config.data_seed,
            config.seq_len,
            '_' + config.feature if config.dataset=="thumos" else '',
            extension
        )
    elif fixed_landmarks:
        fit_layer_epochs = str(config.fit_layer_epochs).replace('[', '-').replace(']', '-')
        return '%s/%s_%d_layers_%d_landmarks_%s_%d_seeds_%d_%d%s.%s' % (
            folder,
            config.model,
            config.num_layers,
            config.num_landmarks,
            fit_layer_epochs,
            freeze_weights,
            config.model_seed,
            config.data_seed,
            '_' + config.feature if config.dataset=="thumos" else '',
            extension
        )
    else:
        return '%s/%s_%d_layers_%d_landmarks_seeds_%d_%d%s.%s' % (
            folder,
            config.model,
            config.num_layers,
            config.num_landmarks,
            config.model_seed,
            config.data_seed,
            '_' + config.feature if config.dataset=="thumos" else '',
            extension,
        )

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_running_time(model, data_loader, config, iterations=100):
    total_time = 0.
    with torch.no_grad():
        print("Computing running time")
        for i, data in tqdm(enumerate(data_loader)):
            if i >= iterations:
                break

            features, _ = get_data(data, config)
            if features.size(0) < data_loader.batch_size:
                continue # Skip the last batch to avoid any problems

            torch.cuda.empty_cache()

            if config.model in ['base_continual', 'continual_nystrom']:
                start_time = time.time()
                _ = model(features)
                end_time = time.time()
                total_time += end_time - start_time
            else:
                seq_len = features.size(2)
                for _ in range(seq_len):
                    features = torch.roll(features, shifts=-1, dims=(-1,))
                    start_time = time.time()
                    _ = model(features)
                    end_time = time.time()
                    total_time += end_time - start_time
        print("Running time computation completed")

    return total_time

def compute_test_accuracy(model, test_loader, config):
    if config.model in ['base_continual', 'continual_nystrom']:
        model.call_mode = "forward_steps"
    model.eval()
    test_performance = evaluate(model, test_loader, config, test_mode=True)
    running_time = calculate_running_time(model, test_loader, config)
    return test_performance, running_time

def get_data(data, config, cut_sequence=True):
    match config.dataset:
        case 'gtzan' | 'electricity':
            features, labels = data
            features = torch.permute(features, (0, 2, 1))
            features = features.to("cuda")
            labels = labels.to("cuda")
        case 'thumos':
            camera_inputs, motion_inputs, _, _, labels, _ = data
            camera_inputs = camera_inputs.to("cuda")
            motion_inputs = motion_inputs.to("cuda")
            features = torch.cat((camera_inputs, motion_inputs), 2).transpose(1, 2)
        case _:
            raise NotImplementedError

    labels = labels.to("cuda")

    if cut_sequence and config.seq_len > 0:
        features = features[:, :, :config.seq_len]
    return features, labels

def check_best_val_performance(val_performance, best_val_performance):
    match config.dataset:
        case 'gtzan':
            val_metric = val_performance['accuracy']
            improved = val_metric >= best_val_performance
            best_val_performance = val_metric
        case 'electricity':
            val_metric = val_performance['huber_loss']
            improved = val_metric <= best_val_performance
            best_val_performance = val_metric
        case _:
            improved = True
    return best_val_performance, improved


def train_one_epoch(model, config, epoch_number, total_epochs, optimizer, criterion, train_loader, val_loader, best_val_performance, fixed_landmarks=False, freeze_weights=False, lr_scheduler=None):
    running_loss = 0.0

    model.train()
    model.call_mode = "forward"

    train_performance = None

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        # load data
        out_target, out = forward_data(model, config, data)
        loss = criterion(out, out_target)
        loss.backward()
        optimizer.step()

        # update training metrics
        running_loss += loss.item()

        train_performance = get_performance(config, out, out_target, train_performance)

    train_performance = normalize_performance(train_performance)
    model.eval()

    val_performance = evaluate(model, val_loader, config)

    if lr_scheduler:
        lr_scheduler.step()

    best_val_performance, improved = check_best_val_performance(val_performance, best_val_performance)
    
    if improved:
        torch.save(model.state_dict(),
                   get_model_path(config, fixed_landmarks=fixed_landmarks, freeze_weights=freeze_weights))


    print('Epoch: %d/%d; Loss: %.2e; ' % (epoch_number + 1, total_epochs, running_loss), end='')
    for key, value in train_performance.items():
        print('train_%s: %.2e; ' % (key, value), end='')
    for key, value in val_performance.items():
        print('val_%s: %.2e; ' % (key, value), end='')

    print('; saved' if improved else '')

    return train_performance, val_performance, best_val_performance

def fix_landmarks(model, dataset, config, freeze_weights=True, layer_number=0, kmeans_attempts=10):
    print("Computing and fixing landmarks for encoder layer "+str(layer_number)+"...")

    seed = config.model_seed
    total_layers = config.num_layers

    assert layer_number < total_layers

    features = dataset.get_all_data(num_samples=config.num_points_nystrom//config.seq_len + 1)
    # features = torch.tensor(features)
    # features = torch.permute(features, (0, 2, 1))
    features = features.to("cuda")

    if config.num_layers == 1:
        module_index = 3 if config.dataset=="thumos" else 2
        nystrom_module = model[module_index][0][1]
    elif layer_number==0:
        module_index = 2
        nystrom_module = model[2].layers[layer_number][0].self_attn
    else:
        module_index = 2
        nystrom_module = model[2].layers[layer_number].fn[0].self_attn

    if freeze_weights:
        for param in model[:module_index].parameters():
            param.requires_grad = False
        if layer_number > 0 and total_layers > 1:
            for param in model[module_index].layers[:layer_number].parameters():
                param.requires_grad = False

        linear_q = nystrom_module.W_q
        linear_k = nystrom_module.W_k

        for param in linear_q.parameters():
            param.requires_grad = False
        for param in linear_k.parameters():
            param.requires_grad = False

    with torch.no_grad():
        features = model[:module_index](features)
        if layer_number > 0 and total_layers > 1:
            features = model[module_index].layers[:layer_number](features)

        # Add the new landmarks
        nystrom_module.fix_landmarks(features, kmeans_attempts=kmeans_attempts, seed=seed, num_points=config.num_points_nystrom)
    print("Finished fixing landmarks for encoder layer " + str(layer_number))


def train_fixed_landmarks(model, config, out_file, train_dataset, optimizer, criterion, train_loader, val_loader, test_loader, freeze_weights=True):
    total_epochs = config.epochs + sum(config.fit_layer_epochs)
    cum_epoch = config.epochs
    train_performance, val_performance = 0., 0.

    if not os.path.exists(get_model_path(config, fixed_landmarks=True, freeze_weights=freeze_weights)):
        for num_layer, num_epochs_fit in enumerate(config.fit_layer_epochs):
            fix_landmarks(model, train_dataset, config, freeze_weights=freeze_weights, layer_number=num_layer)

            best_val_performance = 0.0 if config.dataset != 'electricity' else float('inf')
            # Get val_performance at epoch 0
            if num_layer == len(config.fit_layer_epochs)-1:
                val_performance = evaluate(model, val_loader, config)
                best_val_performance, _ = check_best_val_performance(val_performance, best_val_performance)
                torch.save(model.state_dict(), get_model_path(config, fixed_landmarks=True, freeze_weights=freeze_weights))

            for _ in range(num_epochs_fit):
                train_performance, val_performance, best_val_performance = train_one_epoch(model, config, cum_epoch, total_epochs, optimizer,
                                                                    criterion, train_loader, val_loader,
                                                                    best_val_performance, fixed_landmarks=True, freeze_weights=freeze_weights)
                cum_epoch += 1

    # Reload best model for test
    best_path = get_model_path(config, fixed_landmarks=True, freeze_weights=freeze_weights)
    if not os.path.exists(best_path):
        best_path = get_model_path(config, fixed_landmarks=False)
    model.load_state_dict(torch.load(best_path, map_location="cuda"))

    if test_loader:
        test_performance, test_time = compute_test_accuracy(model, test_loader, config)
    else:
        test_performance, test_time = compute_test_accuracy(model, val_loader, config)

    flops, valley_mem_cost, peak_mem_cost = get_transformer_costs(model, config, fixed_landmarks=True)

    output_content = {
        "config": config,
        "freeze_weights": freeze_weights,
        "train_performance": train_performance,
        "val_performance": val_performance,
        "test_performance": test_performance,
        "test_time": test_time,
        "flops": flops,
        "valley_mem_cost": valley_mem_cost,
        "peak_mem_cost": peak_mem_cost,
    }
    with open(out_file, 'wb') as f:
        pickle.dump(output_content, f)

def torch_train(config):
    print("\n\n\n" + str(config))

    assert config.model in ["base", "base_continual", "nystromformer", "continual_nystrom"]
    assert config.freeze_weights in ["both", "true", "false"]
    assert config.num_layers >= len(config.fit_layer_epochs)
    assert config.dataset in ["gtzan", "thumos", "electricity"]

    train_loader, val_loader, test_loader = get_data_loaders(config)

    if config.model_seed:
        torch.manual_seed(config.model_seed)
        random.seed(config.model_seed)
        np.random.seed(config.model_seed)

    model = get_model(config, train_loader.dataset)

    train_performance = torch.tensor(0.0)
    val_performance = torch.tensor(0.0)

    # optimizer and loss
    optimizer = get_optimizer(config, model)
    criterion = get_criterion(config)

    lr_scheduler = get_lr_scheduler(config, optimizer)

    total_epochs = config.epochs + sum(config.fit_layer_epochs)

    best_val_performance = 0.0 if config.dataset != 'electricity' else float('inf')

    # If we need to retrain the models, we have to adapt this part of the 'if'
    if os.path.exists(get_model_path(config, fixed_landmarks=False, extension="pth")):
        model.load_state_dict(torch.load(get_model_path(config), map_location="cuda"))
    else:
        # training loop
        for epoch in range(config.epochs):
            train_performance, val_performance, best_val_performance = train_one_epoch(
                model, config, epoch, total_epochs, optimizer, criterion, train_loader, val_loader, best_val_performance, fixed_landmarks=False, lr_scheduler=lr_scheduler)

        if config.model not in ['base', 'base_continual']:
            # Copy current state of the model
            torch.save(model.state_dict(), get_model_path(config, fixed_landmarks=False, extension="pth_temp"))

        # Reload best model for test
        model.load_state_dict(torch.load(get_model_path(config, fixed_landmarks=False), map_location="cuda"))

    if test_loader:
        test_performance, test_time = compute_test_accuracy(model, test_loader, config)
    else:
        test_performance, test_time = compute_test_accuracy(model, val_loader, config)

    flops, valley_mem_cost, peak_mem_cost = get_transformer_costs(model, config)

    output_content = {
        "config": config,
        "freeze_weights": None,
        "train_performance": train_performance,
        "val_performance": val_performance,
        "test_performance": test_performance,
        "test_time": test_time,
        "flops": flops,
        "valley_mem_cost": valley_mem_cost,
        "peak_mem_cost": peak_mem_cost,
    }
    dump_file_path = get_model_path(config, log=True, extension="pkl", model_weights=False)
    os.makedirs(config.log_folder, exist_ok=True)
    with open(dump_file_path, 'wb') as f:
        pickle.dump(output_content, f)

    # Continual Nyströmformer's loop
    if config.model not in ['base', 'base_continual']:
        model.load_state_dict(torch.load(get_model_path(config, fixed_landmarks=False, freeze_weights=False, extension="pth_temp"), map_location="cuda"))

        if config.freeze_weights in ["true", "both"] and config.fit_layer_epochs != []:
            train_fixed_landmarks(model, config,
                                  get_model_path(config, True, True, "pkl", log=True),
                                  train_loader.dataset, optimizer, criterion, train_loader, val_loader,
                                  test_loader, freeze_weights=True)

        if config.freeze_weights == "both":
            model.load_state_dict(torch.load(get_model_path(config, fixed_landmarks=False, freeze_weights=False, extension="pth_temp"), map_location="cuda"))

        if config.freeze_weights in ["false", "both"] and config.fit_layer_epochs != []:
            train_fixed_landmarks(model, config,
                                  get_model_path(config, True, False, "pkl", log=True),
                                  train_loader.dataset, optimizer, criterion, train_loader, val_loader,
                                  test_loader, freeze_weights=False)

    return model, test_performance

if __name__ == "__main__":
    os.makedirs(VGGISH_FOLDER, exist_ok=True)
    os.makedirs(GTZAN_CACHE_FOLDER, exist_ok=True)

    config = get_config()

    torch_train(config)

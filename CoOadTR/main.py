import argparse
import datetime
import json
import random
import time
from pathlib import Path
from config import get_args_parser
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, DistributedSampler
import util as utl
import os
import utils

import time

import transformer_models
from dataset import TRNTHUMOSDataLayer, TRNTVSeriesDataLayer
from train import train_one_epoch, evaluate
from test import test_one_epoch
import torch.nn as nn


def one_epoch_train_loop(model, criterion, data_loader_train, data_loader_val, logger, optimizer, n_parameters,
                         device, epoch, sampler_train, lr_scheduler, output_dir, model_without_ddp):
    if args.distributed:
        sampler_train.set_epoch(epoch)

    train_stats = train_one_epoch(
        model,
        criterion,
        data_loader_train,
        optimizer,
        device,
        epoch,
        args.clip_max_norm,
    )

    lr_scheduler.step()
    if args.output_dir:
        checkpoint_paths = [output_dir / "checkpoint.pth"]
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )

    test_stats = evaluate(
        model,
        criterion,
        data_loader_val,
        device,
        logger,
        args,
        epoch,
        nprocs=utils.get_world_size(),
    )

    log_stats = {
        **{f"train_{k}": v for k, v in train_stats.items()},
        **{f"test_{k}": v for k, v in test_stats.items()},
        "epoch": epoch,
        "n_parameters": n_parameters,
    }

    if args.output_dir and utils.is_main_process():
        with (output_dir / "log_tran&test.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")


def fix_landmarks(model, data_loader, config, device, freeze_weights=True, layer_number=0, kmeans_attempts=10):
    print("Computing and fixing landmarks for encoder layer number "+str(layer_number)+"...")

    seed = config.seed
    total_layers = config.num_layers

    assert layer_number < total_layers

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

    # Feature data extraction
    all_features = []
    for camera_inputs, motion_inputs, _, _, _, _ in data_loader:
        features = torch.cat((camera_inputs.to(device), motion_inputs.to(device)), 2).transpose(1, 2)

        with torch.no_grad():
            features = model[:3](features)
            if layer_number > 0 and total_layers > 1:
                features = model[3][0][:layer_number](features)
            all_features.append(features.detach().to("cpu"))
    all_features = torch.cat(all_features, dim=0)

    # Add the new landmarks
    start_time = time.time()
    nystrom_module.fix_landmarks(all_features, kmeans_attempts=kmeans_attempts, seed=seed, num_points=50000)
    end_time = time.time()

    print("Landmark fixing took "+str((end_time-start_time)/60)+" minutes")
    print("Finished fixing landmarks for encoder layer " + str(layer_number))


def train_fixed_landmarks(model, criterion, data_loader_train, data_loader_val, logger, optimizer, n_parameters,
                             device, config, sampler_train, lr_scheduler, output_dir, model_without_ddp, freeze_weights=True):
    cum_epoch = config.epochs
    for num_layer, num_epochs_fit in enumerate(config.fit_layer_epochs):
        fix_landmarks(model, data_loader_train, config, device, freeze_weights=freeze_weights, layer_number=num_layer)
        for _ in range(num_epochs_fit):
            one_epoch_train_loop(model, criterion, data_loader_train, data_loader_val, logger, optimizer, n_parameters,
                         device, cum_epoch, sampler_train, lr_scheduler, output_dir, model_without_ddp)
            cum_epoch += 1

def main(args):
    args.output_dir = get_model_folder(args, fixed_landmarks=False, freeze_weights=False)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    command = "python " + " ".join(sys.argv)
    this_dir = args.output_dir
    if args.removelog:
        if args.distributed:
            print("distributed training !")
            if utils.is_main_process():
                print("remove logs !")
                if os.path.exists(os.path.join(this_dir, "log_dist.txt")):
                    os.remove(os.path.join(this_dir, "log_dist.txt"))
                if os.path.exists(Path(args.output_dir) / "log_tran&test.txt"):
                    os.remove(Path(args.output_dir) / "log_tran&test.txt")
        else:
            print("remove logs !")
            if os.path.exists(os.path.join(this_dir, "log_dist.txt")):
                os.remove(os.path.join(this_dir, "log_dist.txt"))
            if os.path.exists(Path(args.output_dir) / "log_tran&test.txt"):
                os.remove(Path(args.output_dir) / "log_tran&test.txt")
    logger = utl.setup_logger(os.path.join(this_dir, "log_dist.txt"), command=command)
    # logger.output_print("git:\n  {}\n".format(utils.get_sha()))

    # save args
    for arg in vars(args):
        logger.output_print("{}:{}".format(arg, getattr(args, arg)))

    # print(args)
    # set devise
    if args.distributed:
        print("args.gpu : ", args.gpu)
        torch.cuda.set_device(args.gpu)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model = transformer_models.VisionTransformer_v3(
    model = transformer_models.CoVisionTransformer(
        args=args,
        img_dim=args.enc_layers,
        patch_dim=args.patch_dim,
        out_dim=args.numclass,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        attn_dropout_rate=args.attn_dropout_rate,
        num_channels=args.dim_feature,
        positional_encoding_type="recycling_learned" if args.model in ["base_continual", "nystromformer"] else "recycling_fixed",
        nystrom=args.model in ["nystromformer", "continual_nystrom"],
        num_landmarks=args.num_landmarks,
        device=device,
    )

    # Compute FLOPs
    try:
        from ptflops import get_model_complexity_info

        # Warm up model
        model.forward_steps(torch.randn(1, args.dim_feature, args.enc_layers))
        model.call_mode = "forward_step"

        flops, params = get_model_complexity_info(
            model, (args.dim_feature,), as_strings=False
        )
        print(f"Model FLOPs: {flops}")
        print(f"Model params: {params}")

        # Check max mem
        with torch.no_grad():
            model.clean_state()
            model = model.to(device)
            t = torch.randn(1, args.dim_feature, device=device)
            for _ in range(args.enc_layers):
                model.forward_step(t)

            torch.cuda.reset_peak_memory_stats(device=device)
            pre_mem = torch.cuda.memory_allocated(device=device)

            model.forward_step(t)

            print(torch.cuda.memory_summary(device=device, abbreviated=True))

            post_mem = torch.cuda.memory_allocated(device=device)
            max_mem = torch.cuda.max_memory_allocated(device=device)

            print("Memory state pre, max, post inference:", pre_mem, max_mem, post_mem)

    except Exception as e:
        print(e)
        ...

    model.call_mode = "forward"
    model = model.to(device)

    loss_need = [
        "labels_encoder",
        # "labels_decoder",
    ]
    criterion = utl.SetCriterion(
        num_classes=args.numclass, losses=loss_need, args=args
    ).to(device)

    model_without_ddp = model
    if args.distributed:
        # torch.cuda.set_device(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    elif args.dataparallel:
        args.gpu = "0,1,2,3"
        model = nn.DataParallel(
            model, device_ids=[int(iii) for iii in args.gpu.split(",")]
        )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    DataLayer = {"thumos": TRNTHUMOSDataLayer, "tvseries": TRNTVSeriesDataLayer}[
        args.dataset
    ]
    dataset_train = DataLayer(phase="train", args=args)
    dataset_val = DataLayer(phase="test", args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            print("checkpoint: ", args.resume)
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        print("start testing for one epoch !!!")
        with torch.no_grad():
            test_stats = test_one_epoch(
                model,
                criterion,
                data_loader_val,
                device,
                logger,
                args,
                epoch=0,
                nprocs=4,
            )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        one_epoch_train_loop(model, criterion, data_loader_train, data_loader_val, logger, optimizer, n_parameters,
                             device, epoch, sampler_train, lr_scheduler, output_dir, model_without_ddp)

    if args.model in ["nystromformer", "continual_nystrom"] and len(args.fit_layer_epochs) > 0:
        if args.freeze_weights == "both":
            after_normal_checkpoint = output_dir / f"checkpoint{args.epochs:04}.pth"
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": args.epochs,
                    "args": args,
                },
                after_normal_checkpoint,
            )

        if args.freeze_weights in ["true", "both"]:
            train_fixed_landmarks(model, criterion, data_loader_train, data_loader_val, logger, optimizer, n_parameters,
                             device, args, sampler_train, lr_scheduler, output_dir, model_without_ddp,
                                  freeze_weights=True)

        if args.freeze_weights == "both":
            model.load_state_dict(torch.load(after_normal_checkpoint)["model"])

        if args.freeze_weights in ["false", "both"]:
            train_fixed_landmarks(model, criterion, data_loader_train, data_loader_val, logger, optimizer, n_parameters,
                                  device, args, sampler_train, lr_scheduler, output_dir, model_without_ddp,
                                  freeze_weights=False)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

def get_model_folder(config, fixed_landmarks=True, freeze_weights=True):
    if config.model in ['base', 'base_continual'] or not fixed_landmarks:
        return '%s_%d_layers_seeds_%d' % (
            config.model,
            config.num_layers,
            config.seed,
        )
    else:
        fit_layer_epochs = str(config.fit_layer_epochs).replace('[', '-').replace(']', '-')
        return '%s_%d_layers_%d_landmarks_%s_%d_seeds_%d' % (
            config.model,
            config.num_layers,
            config.num_landmarks,
            fit_layer_epochs,
            freeze_weights,
            config.seed,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "OadTR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    # args.dataset = osp.basename(osp.normpath(args.data_root)).upper()
    with open(args.dataset_file, "r") as f:
        data_info = json.load(f)[args.dataset.upper()]
    args.train_session_set = data_info["train_session_set"]
    args.test_session_set = data_info["test_session_set"]
    args.class_index = data_info["class_index"]
    args.numclass = len(args.class_index)

    for seed in range(5):
        args.seed = seed
        for model in ["base", "base_continual"]:
            args.model = model
            for num_layers in [1, 2]:
                args.num_layers = num_layers
                main(args)
        for model in ["nystromformer", "continual_nystrom"]:
            args.model = model
            for num_landmarks in [2, 4, 8, 16, 32]:
                args.num_landmarks = num_landmarks

                args.num_layers = 1
                args.fit_layer_epochs = [5]
                main(args)

                args.num_layers = 2
                args.fit_layer_epochs = [5, 5]
                main(args)

        args.fit_layer_epochs = []

import argparse
import os
import random

import yaml

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import load_dataset
from model import build_model

CUDA_IS_AVAILABLE = torch.cuda.is_available()


def setup_distributed():
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    if CUDA_IS_AVAILABLE:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_distributed:
        dist.init_process_group(
            backend="nccl" if CUDA_IS_AVAILABLE else "gloo",
            device_id=device if CUDA_IS_AVAILABLE else None,
        )

    return is_distributed, rank, local_rank, world_size, local_world_size, device


def unwrap_model(model):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return model


def move_to_device(batch, local_rank):
    if not CUDA_IS_AVAILABLE:
        return batch
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, local_rank) for item in batch)
    if isinstance(batch, list):
        return [move_to_device(item, local_rank) for item in batch]
    return batch.cuda(local_rank, non_blocking=True)


def get_available_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        return os.cpu_count() or 1


def resolve_local_batch_size(batch_size, world_size, label):
    if batch_size <= 0:
        raise ValueError(f"{label} batch_size must be positive, got {batch_size}")
    if batch_size % world_size != 0:
        raise ValueError(
            f"{label} batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        )
    return batch_size // world_size


def validate_dataset_config(dataset_cfg, dataset, split):
    num_classes = len(dataset.classes)
    if num_classes != dataset_cfg.num_classes:
        raise ValueError(
            f"{split} dataset num_classes ({num_classes}) does not match config ({dataset_cfg.num_classes})"
        )

    image, _ = dataset[0]
    if image.shape[-2:] != (dataset_cfg.resolution, dataset_cfg.resolution):
        raise ValueError(
            f"{split} dataset resolution ({image.shape[-2:]}) does not match config ({dataset_cfg.resolution})"
        )


def run_epoch(
    model,
    dataloader,
    local_rank,
    device,
    is_distributed,
    training,
    optimizer=None,
    scheduler=None,
):
    if training and (optimizer is None or scheduler is None):
        raise ValueError("optimizer and scheduler must be provided for training")

    model.train(training)
    running_loss = torch.zeros((), device=device)

    for batch in dataloader:
        x, y = move_to_device(batch, local_rank)

        with torch.set_grad_enabled(training):
            loss = model(x, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        running_loss += loss.detach()

    if is_distributed:
        dist.reduce(running_loss, 0)

    return running_loss


def train_epoch(
    cfg,
    model,
    optimizer,
    scheduler,
    dataloader,
    writer,
    epoch,
    rank,
    local_rank,
    world_size,
    device,
    is_distributed,
):
    running_loss = run_epoch(
        model=model,
        dataloader=dataloader,
        local_rank=local_rank,
        device=device,
        is_distributed=is_distributed,
        training=True,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if rank == 0:
        writer.add_scalar(
            "loss/train",
            running_loss.item() / (world_size * cfg.train.steps_per_epoch),
            cfg.train.steps_per_epoch * epoch,
        )


@torch.no_grad()
def test_epoch(
    cfg,
    model,
    dataloader,
    writer,
    epoch,
    rank,
    local_rank,
    world_size,
    device,
    is_distributed,
):
    running_loss = run_epoch(
        model=model,
        dataloader=dataloader,
        local_rank=local_rank,
        device=device,
        is_distributed=is_distributed,
        training=False,
    )

    if rank == 0:
        writer.add_scalar(
            "loss/test",
            running_loss.item() / (world_size * cfg.test.steps_per_epoch),
            cfg.train.steps_per_epoch * epoch,
        )


def create_random_loader(
    dataset, batch_size, steps_per_epoch, world_size, generator, label, **kwargs
):
    local_batch_size = resolve_local_batch_size(batch_size, world_size, label)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=steps_per_epoch * local_batch_size,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=local_batch_size, sampler=sampler, **kwargs)


def create_dataloader(cfg, rank, world_size, local_world_size):
    train_set = load_dataset(cfg.dataset, split="train")
    test_set = load_dataset(cfg.dataset, split="test") if cfg.test.enabled else None
    validate_dataset_config(cfg.dataset, train_set, "train")
    if test_set is not None:
        validate_dataset_config(cfg.dataset, test_set, "test")

    num_workers = cfg.num_workers
    if num_workers is None:
        num_workers = get_available_cpu_count() // max(local_world_size, 1)
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": CUDA_IS_AVAILABLE,
        "persistent_workers": num_workers > 0,
    }

    generator = torch.Generator()
    generator.manual_seed(world_size * cfg.seed + rank)

    train_loader = create_random_loader(
        dataset=train_set,
        batch_size=cfg.train.batch_size,
        steps_per_epoch=cfg.train.steps_per_epoch,
        world_size=world_size,
        generator=generator,
        label="train",
        **kwargs,
    )

    if test_set is None:
        return train_loader, None

    test_loader = create_random_loader(
        dataset=test_set,
        batch_size=cfg.test.batch_size,
        steps_per_epoch=cfg.test.steps_per_epoch,
        world_size=world_size,
        generator=generator,
        label="test",
        **kwargs,
    )

    return train_loader, test_loader


def main(config_path):
    cfg = Config.load(config_path)

    is_distributed, rank, local_rank, world_size, local_world_size, device = (
        setup_distributed()
    )

    local_seed = world_size * cfg.seed + rank
    torch.manual_seed(local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)

    train_loader, test_loader = create_dataloader(
        cfg, rank, world_size, local_world_size
    )

    model = build_model(cfg.model).to(device)
    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if CUDA_IS_AVAILABLE else None
        )
    if cfg.compile:
        model = torch.compile(model)

    optimizer = Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lambda step: 1.0)

    checkpoint_dir = None
    if rank == 0:
        writer = SummaryWriter(cfg.log_dir)
        cfg.log_dir = writer.log_dir
        with open(os.path.join(writer.log_dir, "config.yaml"), "w") as f:
            f.write(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
        checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        writer = None

    start_epoch = 0
    if cfg.log_dir is not None:
        checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            unwrap_model(model).load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            if rank == 0:
                print(f"Resuming from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg.train.num_epochs)):
        if cfg.test.enabled and test_loader is not None:
            test_epoch(
                cfg,
                model,
                test_loader,
                writer,
                epoch,
                rank,
                local_rank,
                world_size,
                device,
                is_distributed,
            )

        train_epoch(
            cfg,
            model,
            optimizer,
            scheduler,
            train_loader,
            writer,
            epoch,
            rank,
            local_rank,
            world_size,
            device,
            is_distributed,
        )

        if rank == 0 and (epoch + 1) % cfg.save_freq == 0:
            checkpoint = {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
            }

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pt")
            torch.save(checkpoint, checkpoint_path)

            link_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(os.path.abspath(checkpoint_path), link_path)

    if writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Data Parallel Training")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "conf/config.yaml"),
    )
    args = parser.parse_args()
    main(args.config_path)

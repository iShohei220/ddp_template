import argparse
import os
import random
import yaml

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

from config import Config
from dataset import load_dataset
from model import MLP


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


def train(
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
    model.train()
    running_loss = torch.zeros((), device=device)

    for x, y in dataloader:
        optimizer.zero_grad()

        if CUDA_IS_AVAILABLE:
            x = x.cuda(local_rank, non_blocking=True)
            y = y.cuda(local_rank, non_blocking=True)

        loss = unwrap_model(model).loss(x, y)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.detach()

    if is_distributed:
        dist.reduce(running_loss, 0)

    if rank == 0:
        writer.add_scalar(
            "loss/train",
            running_loss.item() / (world_size * cfg.train.steps_per_epoch),
            cfg.train.steps_per_epoch * epoch,
        )


@torch.no_grad()
def test(
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
    model.eval()
    running_loss = torch.zeros((), device=device)

    for x, y in dataloader:
        if CUDA_IS_AVAILABLE:
            x = x.cuda(local_rank, non_blocking=True)
            y = y.cuda(local_rank, non_blocking=True)

        loss = unwrap_model(model).loss(x, y)
        running_loss += loss.detach()

    if is_distributed:
        dist.reduce(running_loss, 0)

    if rank == 0:
        writer.add_scalar(
            "loss/test",
            running_loss.item() / (world_size * cfg.test.steps_per_epoch),
            cfg.train.steps_per_epoch * epoch,
        )


def create_dataloader(cfg, rank, world_size, local_world_size):
    train_set = load_dataset(cfg.dataset, train=True)
    test_set = load_dataset(cfg.dataset, train=False)

    num_workers = cfg.num_workers
    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0)) // local_world_size
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": CUDA_IS_AVAILABLE,
        "persistent_workers": num_workers > 0,
    }

    generator = torch.Generator()
    generator.manual_seed(world_size * cfg.seed + rank)

    train_sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=cfg.train.steps_per_epoch * cfg.train.batch_size,
        generator=generator,
    )
    test_sampler = RandomSampler(
        test_set,
        replacement=True,
        num_samples=cfg.test.steps_per_epoch * cfg.test.batch_size,
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        **kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test.batch_size,
        sampler=test_sampler,
        **kwargs,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed Data Parallel Training"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "conf/config.yaml")
    )

    args = parser.parse_args()
    cfg = Config(**yaml.safe_load(open(args.config_path)))

    is_distributed, rank, local_rank, world_size, local_world_size, device = setup_distributed()

    local_seed = world_size * cfg.seed + rank
    torch.manual_seed(local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)

    train_loader, test_loader = create_dataloader(cfg, rank, world_size, local_world_size)

    model = MLP(cfg.model).to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if CUDA_IS_AVAILABLE else None)
    if cfg.compile:
        model = torch.compile(model)

    optimizer = Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay
    )
    scheduler = LambdaLR(optimizer, lambda step: 1.0)

    if rank == 0:
        writer = SummaryWriter(cfg.log_dir)
        cfg.log_dir = writer.log_dir
        with open(os.path.join(writer.log_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg.model_dump(), f)
    else:
        writer = None

    start_epoch = 0
    if cfg.log_dir is not None:
        checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
            )
            unwrap_model(model).load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg.train.num_epochs)):
        if cfg.test.enabled:
            test(
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

        train(
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

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)

            link_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.islink(link_path):
                os.remove(link_path)

            os.symlink(
                os.path.abspath(checkpoint_path),
                link_path,
            )

    if writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()

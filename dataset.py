import os

import torch
import torch.distributed as dist
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from config import DataConfig


def load_dataset(
    cfg: DataConfig,
    root: str = os.path.join(os.path.dirname(__file__), "data"),
    train=True
):
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if cfg.name == "cifar10":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ])

        if not is_distributed:
            dataset = CIFAR10(
                root,
                train=train,
                download=True,
                transform=transform
            )
        elif rank == 0:
            dataset = CIFAR10(
                root,
                train=train,
                download=True,
                transform=transform
            )

        if is_distributed:
            dist.barrier()
        if is_distributed and rank != 0 and local_rank == 0:
            dataset = CIFAR10(
                root,
                train=train,
                download=True,
                transform=transform
            )

        if is_distributed:
            dist.barrier()
        if is_distributed and local_rank != 0:
            dataset = CIFAR10(
                root,
                train=train,
                download=False,
                transform=transform
            )
    else:
        raise NotImplementedError(f"Dataset {cfg.name} is not supported.")

    return dataset

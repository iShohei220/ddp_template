import os

import torch
import torch.distributed as dist
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from config import DataConfig


def load_dataset(cfg: DataConfig, split: str = "train"):
    if cfg.name == "cifar10":
        return load_cifar10_dataset(cfg, split)
    raise NotImplementedError(f"Dataset {cfg.name} is not supported.")


def load_cifar10_dataset(cfg: DataConfig, split: str):
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")

    root = cfg.root
    if root is None:
        root = os.path.join(os.path.dirname(__file__), "data")

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_train = split == "train"

    transform = v2.Compose(
        [
            v2.Resize((cfg.resolution, cfg.resolution)),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ]
    )

    if not is_distributed:
        dataset = CIFAR10(root, train=is_train, download=True, transform=transform)
    elif rank == 0:
        dataset = CIFAR10(root, train=is_train, download=True, transform=transform)

    if is_distributed:
        dist.barrier()
    if is_distributed and rank != 0 and local_rank == 0:
        dataset = CIFAR10(root, train=is_train, download=True, transform=transform)

    if is_distributed:
        dist.barrier()
    if is_distributed and local_rank != 0:
        dataset = CIFAR10(root, train=is_train, download=False, transform=transform)

    return dataset

import os

import torch
from torch.distributed import barrier
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from config import DataConfig


def load_dataset(
    cfg: DataConfig,
    root: str = os.path.join(os.path.dirname(__file__), "data"),
    train=True
):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Dataset
    if cfg.name == "cifar10":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ])

        if rank == 0:
            dataset = CIFAR10(
                root,
                train=train,
                download=True,
                transform=transform
            )

        barrier()
        if rank != 0 and local_rank == 0:
            dataset = CIFAR10(
                root,
                train=train,
                download=True,
                transform=transform
            )

        barrier()
        if local_rank != 0:
            dataset = CIFAR10(
                root,
                train=train,
                download=False,
                transform=transform
            )
    else:
        raise NotImplementedError(f"Dataset {cfg.name} is not supported.")

    return dataset
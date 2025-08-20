from pydantic.dataclasses import dataclass
from typing import List


@dataclass
class Train:
    batch_size: int = 128
    num_epochs: int = 1000
    steps_per_epoch: int = 100


@dataclass
class Test:
    batch_size: int = 128
    steps_per_epoch: int = 10


@dataclass
class Model:
    resolution: int = 32
    in_channels: int = 3
    out_channels: int = 10


@dataclass
class Optimizer:
    lr: float = 2e-4
    betas: List[float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class Dataset:
    name: str = "cifar10"
    resolution: int = 32
    num_classes: int = 10


@dataclass
class Config:
    seed: int = 0
    num_workers: int = 4
    dataset: Dataset
    model: Model
    optim: Optimizer
    train: Train
    test: Test
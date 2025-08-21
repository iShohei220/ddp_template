from pydantic import BaseModel
from typing import List


class TrainConfig(BaseModel):
    batch_size: int = 128
    num_epochs: int = 1000
    steps_per_epoch: int = 100


class TestConfig(BaseModel):
    batch_size: int = 128
    steps_per_epoch: int = 10


class ModelConfig(BaseModel):
    resolution: int = 32
    in_channels: int = 3
    out_channels: int = 10


class OptimConfig(BaseModel):
    lr: float = 2e-4
    betas: List[float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


class DataConfig(BaseModel):
    name: str = "cifar10"
    resolution: int = 32
    num_classes: int = 10


class Config(BaseModel):
    seed: int = 0
    num_workers: int = 4
    dataset: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig = TrainConfig()
    test: TestConfig = TestConfig()
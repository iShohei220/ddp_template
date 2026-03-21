from pathlib import Path

from pydantic import BaseModel, Field, model_validator
import yaml


class TrainConfig(BaseModel):
    batch_size: int = 128
    num_epochs: int = 1000
    steps_per_epoch: int = 100


class TestConfig(BaseModel):
    enabled: bool = True
    batch_size: int = 128
    steps_per_epoch: int = 10


class ModelConfig(BaseModel):
    resolution: int = 32
    in_channels: int = 3
    out_channels: int | None = None


class OptimConfig(BaseModel):
    lr: float = 2e-4
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0


class DataConfig(BaseModel):
    name: str = "cifar10"
    resolution: int = 32
    num_classes: int = 10
    root: str | None = None


class Config(BaseModel):
    seed: int = 0
    save_freq: int = 100
    log_dir: str | None = None
    num_workers: int | None = None
    compile: bool = True
    dataset: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    test: TestConfig = Field(default_factory=TestConfig)

    @model_validator(mode="after")
    def sync_model_shape(self) -> "Config":
        self.model.resolution = self.dataset.resolution
        if self.model.out_channels is None:
            self.model.out_channels = self.dataset.num_classes
        if self.model.out_channels != self.dataset.num_classes:
            raise ValueError("model.out_channels must match dataset.num_classes")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

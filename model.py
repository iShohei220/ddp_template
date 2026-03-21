import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class ConvClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.out_channels is None:
            raise ValueError("out_channels must be set for ConvClassifier")
        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.out_channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.predict_logits(x)
        return F.cross_entropy(logits, y)


def build_model(cfg: ModelConfig) -> nn.Module:
    return ConvClassifier(cfg)

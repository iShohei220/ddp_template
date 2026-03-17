import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.out_channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)
    
    def loss(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(logits, y)
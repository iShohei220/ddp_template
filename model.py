import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput

from arguments import ModelArguments


class ConvClassifier(nn.Module):
    def __init__(self, cfg: ModelArguments, num_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, num_labels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def predict_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ImageClassifierOutput:
        logits = self.predict_logits(pixel_values)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits)


def build_model(cfg: ModelArguments, num_labels: int) -> nn.Module:
    return ConvClassifier(cfg, num_labels=num_labels)

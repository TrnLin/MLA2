"""Scratch convolutional models for Task 1 article-type classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Task1ModelConfig:
    """Fixed architecture settings for the Task 1 scratch CNN."""

    num_classes: int = 124
    adaptive_size: tuple[int, int] = (5, 7)
    hidden_features: int = 128


class Task1SmallCNN(nn.Module):
    """Small CNN that maps RGB fashion images to raw article-type logits."""

    def __init__(self, num_classes: int = 124) -> None:
        super().__init__()
        self.config = Task1ModelConfig(num_classes=num_classes)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.config.adaptive_size)
        self.fc1 = nn.Linear(64 * 5 * 7, self.config.hidden_features)
        self.fc2 = nn.Linear(self.config.hidden_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw class scores for a batch of ``(N, 3, H, W)`` images."""
        if inputs.ndim != 4:
            raise ValueError("expected input with shape (N, 3, H, W)")
        if inputs.shape[1] != 3:
            raise ValueError("expected input with 3 channels")

        outputs = torch.relu(self.conv1(inputs))
        outputs = torch.relu(self.conv2(outputs))
        outputs = self.pool(outputs)
        outputs = torch.relu(self.conv3(outputs))
        outputs = self.pool(outputs)
        outputs = torch.relu(self.conv4(outputs))
        outputs = torch.relu(self.conv5(outputs))
        outputs = self.pool(outputs)
        outputs = self.adaptive_pool(outputs)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = torch.relu(self.fc1(outputs))
        return self.fc2(outputs)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that are updated during training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

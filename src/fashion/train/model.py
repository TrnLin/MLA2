"""The exact scratch CNN used as Task 3's first learnable parent."""

from __future__ import annotations

import math

import torch
from torch import nn

from fashion.train.config import Task3BaselineConfig, baseline_parameter_count


class Task3BaselineCNN(nn.Module):
    """Four-block native-resolution CNN with three spatial reductions."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        super().__init__()
        channel_pairs = zip((3, *config.channels[:-1]), config.channels, strict=True)
        blocks: list[nn.Module] = []
        for index, (input_channels, output_channels) in enumerate(channel_pairs):
            blocks.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < 3:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(config.channels[-1], config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = baseline_parameter_count(config.target)
        if actual != expected:
            raise RuntimeError(
                f"baseline parameter contract failed: expected {expected}, found {actual}"
            )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)

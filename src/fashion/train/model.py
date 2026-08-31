"""Exact scratch image models used by the Task 3 experiment chain."""

from __future__ import annotations

import math

import torch
from torch import nn

from fashion.train.config import (
    TINYRESNET18_PM_WIDTHS,
    Task3BaselineConfig,
    baseline_parameter_count,
    tinyresnet18_pm_parameter_count,
)


class Task3BaselineCNN(nn.Module):
    """Four-block native-resolution CNN with three spatial reductions."""

    def __init__(
        self, config: Task3BaselineConfig, *, classifier_dropout: float = 0.0
    ) -> None:
        super().__init__()
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier dropout must be in [0, 1)")
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
        self.classifier_dropout = (
            nn.Dropout(p=classifier_dropout) if classifier_dropout > 0.0 else nn.Identity()
        )
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
        return self.classifier(self.classifier_dropout(pooled))


class _Task3ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity or projected residual path."""

    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        features = self.relu(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        return self.relu(features + residual)


class Task3TinyResNet18PM(nn.Module):
    """Parameter-matched residual CNN with no early pooling or pretrained weights."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        super().__init__()
        widths = TINYRESNET18_PM_WIDTHS
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        input_channels = widths[0]
        for stage, output_channels in enumerate(widths):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(
                    _Task3ResidualBlock(
                        input_channels,
                        output_channels,
                        stride=stride,
                    )
                )
                input_channels = output_channels
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(widths[-1], config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = tinyresnet18_pm_parameter_count(config.target)
        if actual != expected:
            raise RuntimeError(
                f"TinyResNet parameter contract failed: expected {expected}, found {actual}"
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
        features = self.features(self.stem(images))
        return self.classifier(self.pool(features).flatten(1))

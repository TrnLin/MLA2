"""Scratch-only candidate architectures for four-class Season prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import resnet18


@dataclass(frozen=True)
class SeasonModelSpec:
    """Architecture settings that belong in an immutable experiment config."""

    family: str
    num_classes: int = 4
    input_channels: int = 3
    dropout: float = 0.20

    def validate(self) -> None:
        if self.family not in {"smallcnn", "resnet18_small_stem"}:
            raise ValueError(f"unknown Season model family: {self.family}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class SmallSeasonCNN(nn.Module):
    """Four convolutional blocks with global pooling and no external weights."""

    training_origin = "scratch"
    benchmark_only = False
    final_eligible = True

    def __init__(
        self,
        *,
        num_classes: int = 4,
        input_channels: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        channels = (32, 64, 128, 256)
        blocks = []
        current_channels = input_channels
        for output_channels in channels:
            blocks.append(_conv_block(current_channels, output_channels))
            current_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.num_classes = num_classes
        self.input_channels = input_channels
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return the last spatial convolution for deterministic Grad-CAM review."""
        return self.features[-1][-4]

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the final spatial feature map before global pooling."""
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                f"expected [batch, {self.input_channels}, height, width] images, "
                f"got {tuple(images.shape)}"
            )
        return self.features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        return self.classifier(self.pool(features))


class ScratchSmallStemResNet18(nn.Module):
    """TorchVision ResNet18 initialized from scratch with a small-image stem."""

    training_origin = "scratch"
    benchmark_only = False
    final_eligible = True
    weights = None

    def __init__(
        self,
        *,
        num_classes: int = 4,
        input_channels: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes),
        )
        self.num_classes = num_classes
        self.input_channels = input_channels
        nn.init.kaiming_normal_(
            self.backbone.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.normal_(self.backbone.fc[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.backbone.fc[-1].bias)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return the last residual convolution for Grad-CAM review."""
        return self.backbone.layer4[-1].conv2

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the layer-4 spatial feature map before global pooling."""
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                f"expected [batch, {self.input_channels}, height, width] images, "
                f"got {tuple(images.shape)}"
            )
        features = self.backbone.conv1(images)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        return self.backbone.layer4(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        pooled = self.backbone.avgpool(features)
        return self.backbone.fc(torch.flatten(pooled, 1))


def build_season_model(spec: SeasonModelSpec) -> nn.Module:
    """Build a final-eligible Season model without downloading or loading weights."""
    spec.validate()
    if spec.family == "smallcnn":
        return SmallSeasonCNN(
            num_classes=spec.num_classes,
            input_channels=spec.input_channels,
            dropout=spec.dropout,
        )
    if spec.family == "resnet18_small_stem":
        return ScratchSmallStemResNet18(
            num_classes=spec.num_classes,
            input_channels=spec.input_channels,
            dropout=spec.dropout,
        )
    raise AssertionError(f"validated but unhandled Season model family: {spec.family}")

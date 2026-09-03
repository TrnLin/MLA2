"""Scratch-only candidate architectures for four-class Season prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, mobilenet_v3_small, resnet18


class ModelBoundaryError(ValueError):
    """Raised when a benchmark or pretrained model enters a final-model path."""


@dataclass(frozen=True)
class SeasonModelSpec:
    """Architecture settings that belong in an immutable experiment config."""

    family: str
    num_classes: int = 4
    input_channels: int = 3
    dropout: float = 0.20

    def validate(self) -> None:
        if self.family not in {"smallcnn", "resnet18_small_stem", "mobilenet_v3_small"}:
            raise ValueError(f"unknown Season model family: {self.family}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class BenchmarkModelSpec:
    """Explicit non-final comparison model settings."""

    family: str
    num_classes: int = 4
    dropout: float = 0.20

    def validate(self) -> None:
        if self.family not in {
            "resnet18_standard_scratch",
            "resnet18_standard_pretrained",
        }:
            raise ValueError(f"unknown benchmark model family: {self.family}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
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
    weights = None

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
        self.embedding_dimension = channels[-1]
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
        return self.classify_embedding(self.forward_embedding(images))

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.forward_features(images)), 1)

    def classify_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


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
        self.embedding_dimension = self.backbone.fc[-1].in_features
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
        return self.classify_embedding(self.forward_embedding(images))

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone.avgpool(self.forward_features(images)), 1)

    def classify_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(embedding)


class ScratchMobileNetV3Small(nn.Module):
    """Compact TorchVision MobileNetV3-Small initialized without external weights."""

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
        self.backbone = mobilenet_v3_small(weights=None, dropout=dropout)
        if input_channels != 3:
            original = self.backbone.features[0][0]
            replacement = nn.Conv2d(
                input_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                dilation=original.dilation,
                groups=original.groups,
                bias=False,
            )
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.features[0][0] = replacement
        input_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(input_features, num_classes)
        nn.init.normal_(self.backbone.classifier[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.backbone.classifier[-1].bias)
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.embedding_dimension = self.backbone.classifier[0].in_features

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return the final pointwise convolution before global pooling."""
        return self.backbone.features[-1][0]

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the final spatial feature map before global pooling."""
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                f"expected [batch, {self.input_channels}, height, width] images, "
                f"got {tuple(images.shape)}"
            )
        return self.backbone.features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_embedding(self.forward_embedding(images))

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone.avgpool(self.forward_features(images)), 1)

    def classify_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(embedding)


class BenchmarkStandardStemResNet18(nn.Module):
    """Matched standard-stem scratch or ImageNet comparison, never a final candidate."""

    benchmark_only = True
    final_eligible = False

    def __init__(
        self,
        *,
        pretrained: bool,
        num_classes: int = 4,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        selected_weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=selected_weights)
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_features, num_classes),
        )
        nn.init.normal_(self.backbone.fc[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.backbone.fc[-1].bias)
        self.training_origin = "imagenet_pretrained" if pretrained else "scratch"
        self.weights = "ResNet18_Weights.DEFAULT" if pretrained else None
        self.input_channels = 3
        self.embedding_dimension = input_features

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].conv2

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
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

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone.avgpool(self.forward_features(images)), 1)

    def classify_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(embedding)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_embedding(self.forward_embedding(images))


class SeasonArticleTypeMultiTaskModel(nn.Module):
    """Shared scratch image encoder with Season and training-only ArticleType heads."""

    training_origin = "scratch"
    benchmark_only = False
    final_eligible = True
    weights = None

    def __init__(self, base_model: nn.Module, *, article_type_classes: int = 124) -> None:
        super().__init__()
        assert_final_model(base_model)
        if article_type_classes < 2:
            raise ValueError("article_type_classes must be at least 2")
        embedding_dimension = getattr(base_model, "embedding_dimension", None)
        if not isinstance(embedding_dimension, int):
            raise TypeError("base model must expose an integer embedding_dimension")
        self.base_model = base_model
        self.article_type_head = nn.Linear(embedding_dimension, article_type_classes)
        nn.init.normal_(self.article_type_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.article_type_head.bias)
        self.article_type_classes = article_type_classes
        self.embedding_dimension = embedding_dimension

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.base_model.gradcam_target_layer

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.base_model.forward_embedding(images)
        return {
            "season_logits": self.base_model.classify_embedding(embedding),
            "article_type_logits": self.article_type_head(embedding),
        }

    def predict_season_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Image-only deployment path; no ArticleType input is accepted."""
        return self.forward(images)["season_logits"]


def model_boundary_audit(model: nn.Module) -> dict[str, Any]:
    """Return the metadata used to prevent benchmark leakage into final artifacts."""
    return {
        "class": type(model).__name__,
        "training_origin": getattr(model, "training_origin", None),
        "benchmark_only": getattr(model, "benchmark_only", None),
        "final_eligible": getattr(model, "final_eligible", None),
        "weights": getattr(model, "weights", "undeclared"),
    }


def assert_final_model(model: nn.Module) -> dict[str, Any]:
    """Fail closed unless a model is declared scratch, unweighted, and final-eligible."""
    audit = model_boundary_audit(model)
    failures = []
    if audit["training_origin"] != "scratch":
        failures.append("training_origin must be scratch")
    if audit["benchmark_only"] is not False:
        failures.append("benchmark_only must be false")
    if audit["final_eligible"] is not True:
        failures.append("final_eligible must be true")
    if audit["weights"] is not None:
        failures.append("weights must be None")
    if failures:
        raise ModelBoundaryError(f"model is not final-eligible: {'; '.join(failures)}")
    return audit


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
    if spec.family == "mobilenet_v3_small":
        return ScratchMobileNetV3Small(
            num_classes=spec.num_classes,
            input_channels=spec.input_channels,
            dropout=spec.dropout,
        )
    raise AssertionError(f"validated but unhandled Season model family: {spec.family}")


def build_benchmark_model(spec: BenchmarkModelSpec) -> BenchmarkStandardStemResNet18:
    """Build an explicitly benchmark-only standard-stem ResNet18 comparison."""
    spec.validate()
    return BenchmarkStandardStemResNet18(
        pretrained=spec.family == "resnet18_standard_pretrained",
        num_classes=spec.num_classes,
        dropout=spec.dropout,
    )


def build_multitask_season_model(
    base_spec: SeasonModelSpec,
    *,
    article_type_classes: int = 124,
) -> SeasonArticleTypeMultiTaskModel:
    """Build I2 from a final-eligible scratch Season family."""
    return SeasonArticleTypeMultiTaskModel(
        build_season_model(base_spec),
        article_type_classes=article_type_classes,
    )

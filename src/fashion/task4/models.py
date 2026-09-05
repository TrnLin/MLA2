"""Learned retrieval models for the frozen Task 4 comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as torchvision_models
from torchvision.models import ResNet18_Weights

Architecture: TypeAlias = Literal["resnet18", "resnet34"]

EMBEDDING_DIM = 128
EMBEDDING_NORM_ATOL = 1e-5
B1_WEIGHT_ORIGIN = "ResNet18_Weights.IMAGENET1K_V1"
SCRATCH_WEIGHT_ORIGIN = "random_initialization"
_FACTORY_TOKEN = object()

__all__ = (
    "B1_WEIGHT_ORIGIN",
    "EMBEDDING_DIM",
    "ConvolutionalAutoencoder",
    "ModelMetadata",
    "RetrievalEncoder",
    "build_autoencoder",
    "build_b1_encoder",
    "build_retrieval_encoder",
)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Immutable provenance and deployment status for one learned model."""

    architecture: str
    pretrained: bool
    weight_origin: str
    deployment_eligible: bool

    def __post_init__(self) -> None:
        if self.architecture not in {"resnet18", "resnet34"}:
            raise ValueError("model metadata architecture is unsupported")
        if type(self.pretrained) is not bool or type(self.deployment_eligible) is not bool:
            raise ValueError("model metadata flags must be boolean")
        if self.pretrained:
            if (
                self.architecture != "resnet18"
                or self.weight_origin != B1_WEIGHT_ORIGIN
                or self.deployment_eligible
            ):
                raise ValueError("pretrained model metadata must describe comparison-only B1")
        elif self.weight_origin != SCRATCH_WEIGHT_ORIGIN:
            raise ValueError("scratch model metadata must record random initialization")


def _normalized_embedding(values: torch.Tensor) -> torch.Tensor:
    embeddings = values.float()
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"embedding must have shape (batch, {EMBEDDING_DIM})")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embedding values must be finite")
    norms = torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise ValueError("embedding pre-normalization norms must be finite")
    if torch.any(norms <= 0):
        raise ValueError("embedding rows must have non-zero norm")
    normalized = F.normalize(embeddings, p=2, dim=1)
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized embedding values must be finite")
    normalized_norms = torch.linalg.vector_norm(normalized, dim=1)
    if not torch.allclose(
        normalized_norms,
        torch.ones_like(normalized_norms),
        atol=EMBEDDING_NORM_ATOL,
        rtol=0.0,
    ):
        raise ValueError(
            f"normalized embeddings must have unit norm within {EMBEDDING_NORM_ATOL}"
        )
    return normalized


def _projection_head() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, EMBEDDING_DIM),
    )


class _MetadataBoundModule(nn.Module):
    def __init__(self, metadata: ModelMetadata) -> None:
        super().__init__()
        self._metadata = metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def __setattr__(self, name: str, value: object) -> None:
        if name == "metadata":
            raise AttributeError("model metadata is read-only")
        super().__setattr__(name, value)

    def _require_retrieval_mode(self) -> None:
        if self.training:
            raise RuntimeError("encode is retrieval-only; call eval() before encoding")


class RetrievalEncoder(_MetadataBoundModule):
    """A ResNet feature extractor with the projector ruled 128-value head."""

    def __init__(
        self,
        backbone: nn.Module,
        metadata: ModelMetadata,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise RuntimeError("RetrievalEncoder must be constructed by its factory")
        super().__init__(metadata)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = _projection_head()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized projections for training objectives such as VICReg."""
        return self.projection(self.backbone(images))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return finite float32 unit embeddings at the retrieval boundary."""
        self._require_retrieval_mode()
        return _normalized_embedding(self(images))


def build_retrieval_encoder(
    architecture: Architecture = "resnet18",
    *,
    weights: ResNet18_Weights | None = None,
    deployment_eligible: bool = True,
) -> RetrievalEncoder:
    """Build a scratch encoder by default, enforcing pretrained safety."""
    if architecture not in {"resnet18", "resnet34"}:
        raise ValueError("architecture must be 'resnet18' or 'resnet34'")
    if weights is not None and deployment_eligible:
        raise ValueError("a pretrained model cannot be deployment eligible")
    if weights is not None and (
        architecture != "resnet18" or weights is not ResNet18_Weights.IMAGENET1K_V1
    ):
        raise ValueError("B1 is the only pretrained model and must use the pinned weight")

    constructor = (
        torchvision_models.resnet18
        if architecture == "resnet18"
        else torchvision_models.resnet34
    )
    backbone = constructor(weights=weights)
    pretrained = weights is not None
    metadata = ModelMetadata(
        architecture=architecture,
        pretrained=pretrained,
        weight_origin=B1_WEIGHT_ORIGIN if pretrained else SCRATCH_WEIGHT_ORIGIN,
        deployment_eligible=deployment_eligible,
    )
    return RetrievalEncoder(backbone, metadata, _factory_token=_FACTORY_TOKEN)


def build_b1_encoder() -> RetrievalEncoder:
    """Build the pinned ImageNet benchmark, permanently comparison-only."""
    return build_retrieval_encoder(
        "resnet18",
        weights=ResNet18_Weights.IMAGENET1K_V1,
        deployment_eligible=False,
    )


def _build_b1_checkpoint_encoder() -> RetrievalEncoder:
    """Rebuild B1 offline so a complete checkpoint can restore its weights."""

    backbone = torchvision_models.resnet18(weights=None)
    metadata = ModelMetadata(
        architecture="resnet18",
        pretrained=True,
        weight_origin=B1_WEIGHT_ORIGIN,
        deployment_eligible=False,
    )
    return RetrievalEncoder(backbone, metadata, _factory_token=_FACTORY_TOKEN)


def _decoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(EMBEDDING_DIM, 256 * 10 * 8),
        nn.ReLU(),
        nn.Unflatten(1, (256, 10, 8)),
        nn.Upsample(size=(20, 15), mode="bilinear", align_corners=False),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Upsample(size=(40, 30), mode="bilinear", align_corners=False),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(size=(80, 60), mode="bilinear", align_corners=False),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Upsample(size=(160, 120), mode="bilinear", align_corners=False),
        nn.Conv2d(32, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Upsample(size=(320, 240), mode="bilinear", align_corners=False),
        nn.Conv2d(16, 3, kernel_size=3, padding=1),
    )


class ConvolutionalAutoencoder(_MetadataBoundModule):
    """Scratch ResNet-18 autoencoder without encoder-to-decoder skips."""

    def __init__(
        self,
        backbone: nn.Module,
        metadata: ModelMetadata,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise RuntimeError("ConvolutionalAutoencoder must be constructed by its factory")
        super().__init__(metadata)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.bottleneck = nn.Linear(512, EMBEDDING_DIM)
        self.decoder = _decoder()

    def _bottleneck(self, images: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.encoder(images))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the reconstruction and unnormalized 128-value bottleneck."""
        bottleneck = self._bottleneck(images)
        return self.decoder(bottleneck), bottleneck

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return the normalized bottleneck used by retrieval."""
        self._require_retrieval_mode()
        return _normalized_embedding(self._bottleneck(images))


def build_autoencoder() -> ConvolutionalAutoencoder:
    """Build the scratch, deployment-eligible R5 autoencoder."""
    backbone = torchvision_models.resnet18(weights=None)
    metadata = ModelMetadata(
        architecture="resnet18",
        pretrained=False,
        weight_origin=SCRATCH_WEIGHT_ORIGIN,
        deployment_eligible=True,
    )
    return ConvolutionalAutoencoder(backbone, metadata, _factory_token=_FACTORY_TOKEN)

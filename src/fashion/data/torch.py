"""Fold-fitted image statistics and picklable PyTorch tensor transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fashion.config import ROOT
from fashion.data.images import StreamingStats, resolve_image_size, transform_image_with_mask
from fashion.train.artifacts import canonical_sha256

AugmentationPolicy = Literal["none", "a0", "a1"]


@dataclass(frozen=True)
class FoldImageStats:
    """Normalization fitted only on the training side of one canonical fold."""

    validation_fold: int
    image_size: tuple[int, int]
    image_count: int
    content_pixel_count: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    training_id_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageTransformSpec:
    """Exact geometry and mild augmentation settings for a registered run."""

    image_size: tuple[int, int]
    augmentation: AugmentationPolicy
    horizontal_flip_probability: float = 0.5
    affine_degrees: float = 8.0
    affine_translate: tuple[float, float] = (0.05, 0.05)
    affine_scale: tuple[float, float] = (0.95, 1.05)
    brightness: float = 0.10
    contrast: float = 0.10
    saturation: float = 0.08
    hue: float = 0.02

    @property
    def transform_id(self) -> str:
        return f"{self.augmentation}-{canonical_sha256(asdict(self))[:16]}"


class TensorImageTransform:
    """Load one image, apply optional training augmentation, and return CHW float32."""

    def __init__(
        self,
        *,
        stats: FoldImageStats,
        spec: ImageTransformSpec,
    ) -> None:
        if spec.image_size != stats.image_size:
            raise ValueError("transform and fitted-stat image sizes must match")
        if any(value <= 0 for value in stats.std):
            raise ValueError("fitted channel std values must be positive")
        self.stats = stats
        self.spec = spec
        neutral_fill = tuple(round(value * 255) for value in stats.mean)
        operations: list[Any] = []
        if spec.augmentation in {"a0", "a1"}:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(spec.horizontal_flip_probability),
                    transforms.RandomAffine(
                        degrees=spec.affine_degrees,
                        translate=spec.affine_translate,
                        scale=spec.affine_scale,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=neutral_fill,
                    ),
                ]
            )
        if spec.augmentation == "a1":
            operations.append(
                transforms.ColorJitter(
                    brightness=spec.brightness,
                    contrast=spec.contrast,
                    saturation=spec.saturation,
                    hue=spec.hue,
                )
            )
        self.augmentation = transforms.Compose(operations)

    def __call__(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = self.augmentation(image)
            array, _ = transform_image_with_mask(
                image,
                image_size=self.stats.image_size,
                normalize_range=True,
                mean=self.stats.mean,
                std=self.stats.std,
            )
        return torch.from_numpy(np.moveaxis(array, -1, 0).copy())


def _safe_image_path(root: Path, relative_path: Any) -> Path:
    raw_path = Path(str(relative_path))
    if raw_path.is_absolute():
        raise ValueError(f"manifest image paths must be relative: {raw_path}")
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest image path escapes project root: {raw_path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest image does not exist: {resolved}")
    return resolved


def fit_fold_stats(
    training_frame: pd.DataFrame,
    *,
    validation_fold: int,
    image_size: int | tuple[int, int],
    root: str | Path = ROOT,
) -> FoldImageStats:
    """Fit RGB mean/std on training-fold content pixels, never letterbox padding."""
    required = {"id", "path", "partition", "cv_fold"}
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    if training_frame.empty:
        raise ValueError("training frame must not be empty")
    if training_frame["id"].duplicated().any():
        raise ValueError("training frame IDs must be unique")
    if set(training_frame["partition"].astype(str)) != {"development"}:
        raise ValueError("fold statistics may use development rows only")
    folds = pd.to_numeric(training_frame["cv_fold"], errors="raise").astype(int)
    if folds.eq(validation_fold).any():
        raise ValueError("validation-fold rows cannot fit image statistics")

    resolved_size = resolve_image_size(image_size)
    project_root = Path(root).resolve()
    streaming = StreamingStats(channels=3)
    ordered = training_frame.sort_values("id", kind="stable")
    for row in ordered.itertuples(index=False):
        path = _safe_image_path(project_root, getattr(row, "path"))
        try:
            with Image.open(path) as image:
                array, content_mask = transform_image_with_mask(
                    image,
                    image_size=resolved_size,
                    normalize_range=True,
                )
        except Exception as error:
            raise RuntimeError(f"failed to fit image statistics for ID {row.id}: {path}") from error
        streaming.update(array, content_mask=content_mask)

    mean = tuple(float(value) for value in streaming.mean)
    std = tuple(float(value) for value in streaming.std)
    if any(value <= 0 for value in std):
        raise ValueError("training-fold image statistics contain a non-positive channel std")
    training_ids = sorted(int(value) for value in ordered["id"])
    return FoldImageStats(
        validation_fold=validation_fold,
        image_size=resolved_size,
        image_count=len(ordered),
        content_pixel_count=streaming.total_pixels,
        mean=mean,
        std=std,
        training_id_sha256=canonical_sha256(training_ids),
    )


def build_image_transform(
    stats: FoldImageStats,
    *,
    training: bool,
    augmentation: AugmentationPolicy | None = None,
) -> TensorImageTransform:
    """Build an evaluation transform or one of the predeclared A0/A1 training policies."""
    policy: AugmentationPolicy = augmentation or ("a0" if training else "none")
    if policy not in {"none", "a0", "a1"}:
        raise ValueError(f"unknown augmentation policy: {policy}")
    if not training and policy != "none":
        raise ValueError("validation and inference transforms cannot use random augmentation")
    return TensorImageTransform(
        stats=stats,
        spec=ImageTransformSpec(image_size=stats.image_size, augmentation=policy),
    )

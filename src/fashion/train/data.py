"""Fold-safe PyTorch data path for the Task 3 baseline."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from fashion.config import ROOT
from fashion.data.dataset import get_cv_split, get_samples
from fashion.data.images import StreamingStats, transform_image, transform_image_with_mask
from fashion.train.augmentation import TRAINING_AUGMENTATIONS, apply_training_augmentation

CORE_CORRUPTIONS = (
    "jpeg_75",
    "brightness_085",
    "brightness_115",
    "translation_003",
    "grayscale",
)


def _load_corrupted(path: Path, corruption: str | None) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if corruption is None:
        return image
    if corruption == "jpeg_75":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")
    if corruption == "brightness_085":
        return ImageEnhance.Brightness(image).enhance(0.85)
    if corruption == "brightness_115":
        return ImageEnhance.Brightness(image).enhance(1.15)
    if corruption == "translation_003":
        shift_x = max(1, round(image.width * 0.03))
        shift_y = max(1, round(image.height * 0.03))
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, (shift_x, shift_y))
        return canvas
    if corruption == "grayscale":
        return image.convert("L").convert("RGB")
    raise ValueError(f"unknown core corruption: {corruption}")


def task3_target_frames(
    splits: pd.DataFrame,
    *,
    target: str,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid target rows from one canonical training/validation fold."""
    training, validation = get_cv_split(splits, validation_fold)
    training = get_samples(training, target=target).reset_index(drop=True)
    validation = get_samples(validation, target=target).reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError(f"target {target} has an empty fold {validation_fold}")
    overlap = set(training["product_family_group"]).intersection(validation["product_family_group"])
    if overlap:
        raise ValueError("a product family crosses the requested training and validation fold")
    return training, validation


def fit_fold_rgb_stats(
    training: pd.DataFrame,
    *,
    root: str | Path = ROOT,
    image_size: tuple[int, int] = (80, 60),
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Fit RGB mean/std on training content pixels, excluding letterbox padding."""
    root = Path(root)
    stats = StreamingStats(channels=3)
    total = len(training)
    for position, relative_path in enumerate(training["path"], start=1):
        path = root / str(relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            array, mask = transform_image_with_mask(image, image_size=image_size)
        stats.update(array, content_mask=mask)
        if progress is not None:
            progress(position, total)
    result = stats.to_dict()
    if any(value <= 0 for value in result["std"]):
        raise ValueError("fold-training RGB statistics contain a non-positive standard deviation")
    return result


class Task3ImageDataset(Dataset[dict[str, Any]]):
    """Load one target and the metadata required for OOF traceability."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        target: str,
        label_to_index: dict[str, int],
        mean: Sequence[float],
        std: Sequence[float],
        root: str | Path = ROOT,
        image_size: tuple[int, int] = (80, 60),
        corruption: str | None = None,
        augmentation: str = "none",
    ) -> None:
        if corruption not in (None, *CORE_CORRUPTIONS):
            raise ValueError(f"unknown core corruption: {corruption}")
        if augmentation not in TRAINING_AUGMENTATIONS:
            raise ValueError(f"unknown Task 3 training augmentation: {augmentation}")
        if corruption is not None and augmentation != "none":
            raise ValueError("training augmentation and diagnostic corruption cannot be combined")
        self.frame = frame.reset_index(drop=True)
        self.target = target
        self.label_to_index = dict(label_to_index)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.root = Path(root)
        self.image_size = image_size
        self.corruption = corruption
        self.augmentation = augmentation
        unknown = sorted(set(self.frame[target]).difference(self.label_to_index))
        if unknown:
            raise ValueError(f"unknown {target} labels: {unknown}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = _load_corrupted(self.root / str(row["path"]), self.corruption)
        image = apply_training_augmentation(image, self.augmentation)
        array = transform_image(
            image,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        return {
            "image": tensor,
            "label": int(self.label_to_index[str(row[self.target])]),
            "id": int(row["id"]),
            "cv_fold": int(row["cv_fold"]),
            "product_family_group": str(row["product_family_group"]),
            "path": str(row["path"]),
        }

"""Deterministic image transforms and streaming RGB statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


def transform_image(
    image: Image.Image,
    image_size: int = 128,
    pad_color: tuple[int, int, int] = (255, 255, 255),
    normalize_range: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """EXIF-normalize, RGB-convert, letterbox, and optionally standardize an image."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")

    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    scale = image_size / max(original_width, original_height)
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    resized = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    canvas.paste(resized, ((image_size - width) // 2, (image_size - height) // 2))
    array = np.asarray(canvas, dtype=np.float32)
    if normalize_range:
        array = array / 255.0

    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(std_array <= 0):
            raise ValueError("std values must be positive")
        array = (array - mean_array) / std_array
    return array.astype(np.float32, copy=False)


def load_and_transform_image(
    path: str | Path,
    image_size: int = 128,
    pad_color: tuple[int, int, int] = (255, 255, 255),
    normalize_range: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Load an image and apply :func:`transform_image`."""
    with Image.open(path) as image:
        return transform_image(
            image,
            image_size=image_size,
            pad_color=pad_color,
            normalize_range=normalize_range,
            mean=mean,
            std=std,
        )


class StreamingStats:
    """Accumulate per-channel population statistics with bounded memory."""

    def __init__(self, channels: int = 3) -> None:
        self.channels = channels
        self.total_pixels = 0
        self.channel_sum = np.zeros(channels, dtype=np.float64)
        self.channel_sum_sq = np.zeros(channels, dtype=np.float64)

    def update(self, image: np.ndarray) -> None:
        """Add one HWC image array."""
        if image.ndim != 3 or image.shape[-1] != self.channels:
            raise ValueError(f"expected HWC image with {self.channels} channels")
        pixels = image.reshape(-1, self.channels).astype(np.float64)
        self.total_pixels += len(pixels)
        self.channel_sum += pixels.sum(axis=0)
        self.channel_sum_sq += np.square(pixels).sum(axis=0)

    @property
    def mean(self) -> list[float]:
        if self.total_pixels == 0:
            return [0.0] * self.channels
        return (self.channel_sum / self.total_pixels).tolist()

    @property
    def std(self) -> list[float]:
        if self.total_pixels == 0:
            return [1.0] * self.channels
        mean = self.channel_sum / self.total_pixels
        variance = np.maximum(self.channel_sum_sq / self.total_pixels - np.square(mean), 0.0)
        return np.sqrt(variance).tolist()

    def to_dict(self) -> dict[str, object]:
        return {
            "channels": self.channels,
            "total_pixels": self.total_pixels,
            "mean": self.mean,
            "std": self.std,
        }

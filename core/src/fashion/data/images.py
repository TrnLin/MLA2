"""Deterministic image transforms and streaming RGB statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

ImageSize = int | tuple[int, int]


def resolve_image_size(image_size: ImageSize) -> tuple[int, int]:
    """Return ``(height, width)`` while keeping the old square integer API valid."""
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must contain height and width")
    height, width = (int(value) for value in image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size values must be positive")
    return height, width


def transform_image_with_mask(
    image: Image.Image,
    image_size: ImageSize = (128, 96),
    pad_color: tuple[int, int, int] = (255, 255, 255),
    normalize_range: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an image and return a mask for resized content rather than padding."""
    output_height, output_width = resolve_image_size(image_size)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")

    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    scale = min(output_width / original_width, output_height / original_height)
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    resized = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    left = (output_width - width) // 2
    top = (output_height - height) // 2
    canvas = Image.new("RGB", (output_width, output_height), pad_color)
    canvas.paste(resized, (left, top))
    content_mask = np.zeros((output_height, output_width), dtype=bool)
    content_mask[top : top + height, left : left + width] = True
    array = np.asarray(canvas, dtype=np.float32)
    if normalize_range:
        array = array / 255.0

    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(std_array <= 0):
            raise ValueError("std values must be positive")
        array = (array - mean_array) / std_array
        # Zero is the channel mean after standardization, so padding stays neutral.
        array[~content_mask] = 0.0
    return array.astype(np.float32, copy=False), content_mask


def transform_image(
    image: Image.Image,
    image_size: ImageSize = (128, 96),
    pad_color: tuple[int, int, int] = (255, 255, 255),
    normalize_range: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """EXIF-normalize, RGB-convert, letterbox, and optionally standardize an image."""
    array, _ = transform_image_with_mask(
        image,
        image_size=image_size,
        pad_color=pad_color,
        normalize_range=normalize_range,
        mean=mean,
        std=std,
    )
    return array


def load_and_transform_image(
    path: str | Path,
    image_size: ImageSize = (128, 96),
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

    def update(self, image: np.ndarray, content_mask: np.ndarray | None = None) -> None:
        """Add one HWC image array, optionally excluding letterbox padding."""
        if image.ndim != 3 or image.shape[-1] != self.channels:
            raise ValueError(f"expected HWC image with {self.channels} channels")
        if content_mask is not None:
            if content_mask.shape != image.shape[:2]:
                raise ValueError("content mask shape must match image height and width")
            pixels = image[content_mask].reshape(-1, self.channels).astype(np.float64)
        else:
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

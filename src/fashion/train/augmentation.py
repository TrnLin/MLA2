"""The one predeclared training augmentation used by Task 3 children."""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance

TRAINING_AUGMENTATIONS = (
    "none",
    "brightness_uniform_085_115",
)


def apply_training_augmentation(image: Image.Image, augmentation: str) -> Image.Image:
    if augmentation == "none":
        return image
    if augmentation == "brightness_uniform_085_115":
        factor = random.uniform(0.85, 1.15)
        return ImageEnhance.Brightness(image).enhance(factor)
    raise ValueError(f"unknown Task 3 training augmentation: {augmentation}")

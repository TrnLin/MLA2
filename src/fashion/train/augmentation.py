"""Predeclared single-factor training augmentations for Task 3 children."""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance

TRAINING_AUGMENTATIONS = (
    "none",
    "brightness_uniform_085_115",
    "translation_uniform_2px_p05",
)


def apply_training_augmentation(image: Image.Image, augmentation: str) -> Image.Image:
    if augmentation == "none":
        return image
    if augmentation == "brightness_uniform_085_115":
        factor = random.uniform(0.85, 1.15)
        return ImageEnhance.Brightness(image).enhance(factor)
    if augmentation == "translation_uniform_2px_p05":
        if random.random() >= 0.5:
            return image
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(255, 255, 255),
        )
    raise ValueError(f"unknown Task 3 training augmentation: {augmentation}")

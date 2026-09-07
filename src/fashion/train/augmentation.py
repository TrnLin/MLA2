"""Predeclared single-factor training augmentations for Task 3 children."""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance

GRAYSCALE_AUGMENTATION = "translation_2px_p05_mild_darkening_p025_grayscale_p010"
GRAYSCALE_PROBABILITY = 0.10
GRAYSCALE_SEED_XOR = 0x47524159

TRAINING_AUGMENTATIONS = (
    "none",
    "brightness_uniform_085_115",
    "translation_uniform_2px_p05",
    "translation_2px_p05_mild_darkening_p025",
    GRAYSCALE_AUGMENTATION,
)


def apply_training_augmentation(
    image: Image.Image,
    augmentation: str,
    *,
    darkening_rng: random.Random | None = None,
    grayscale_rng: random.Random | None = None,
) -> Image.Image:
    if augmentation == GRAYSCALE_AUGMENTATION:
        if grayscale_rng is None:
            raise ValueError("occasional grayscale requires its own seeded random stream")
        image = apply_training_augmentation(
            image, "translation_2px_p05_mild_darkening_p025", darkening_rng=darkening_rng
        )
        if grayscale_rng.random() < GRAYSCALE_PROBABILITY:
            image = image.convert("L").convert("RGB")
        return image
    if augmentation == "translation_2px_p05_mild_darkening_p025":
        if darkening_rng is None:
            raise ValueError("mild darkening requires its own seeded random stream")
        image = apply_training_augmentation(image, "translation_uniform_2px_p05")
        if darkening_rng.random() < 0.25:
            image = ImageEnhance.Brightness(image).enhance(darkening_rng.uniform(0.90, 1.00))
        return image
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

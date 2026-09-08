"""Leakage-safe image preprocessing for Task 1 article-type models."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

from fashion.config import CV_FOLD_COUNT, ROOT
from fashion.data.images import StreamingStats, transform_image_with_mask
from fashion.task1.image_contract import TASK1_IMAGE_SIZE, TASK1_PAD_COLOR


@dataclass(frozen=True)
class Task1PreprocessingConfig:
    """Fixed geometry and mild Option B augmentation settings."""

    preprocessing_id: str = "task1_rgb_60x80_mild_aug_v1"
    image_size: tuple[int, int] = TASK1_IMAGE_SIZE
    pad_color: tuple[int, int, int] = TASK1_PAD_COLOR
    horizontal_flip_probability: float = 0.5
    max_rotation_degrees: float = 5.0
    max_translation_fraction: float = 0.05
    scale_range: tuple[float, float] = (0.95, 1.05)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size values must be positive")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")
        if self.max_rotation_degrees < 0:
            raise ValueError("max_rotation_degrees must be non-negative")
        if not 0.0 <= self.max_translation_fraction <= 1.0:
            raise ValueError("max_translation_fraction must be between 0 and 1")
        for name, value_range in (
            ("scale_range", self.scale_range),
            ("brightness_range", self.brightness_range),
            ("contrast_range", self.contrast_range),
        ):
            low, high = value_range
            if low <= 0 or high < low:
                raise ValueError(f"{name} must contain positive values in ascending order")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready settings for run records and checkpoints."""
        return asdict(self)


@dataclass(frozen=True)
class Task1Normalization:
    """Per-channel statistics fitted on one allowed set of development rows."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    fitted_products: int
    fitted_ids_sha256: str

    def __post_init__(self) -> None:
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must contain three RGB values")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")
        if self.fitted_products <= 0:
            raise ValueError("fitted_products must be positive")
        if len(self.fitted_ids_sha256) != 64:
            raise ValueError("fitted_ids_sha256 must contain a SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready fitted values for run records and checkpoints."""
        return asdict(self)


DEFAULT_TASK1_PREPROCESSING = Task1PreprocessingConfig()
TASK1_CONTROL_PREPROCESSING = Task1PreprocessingConfig(
    preprocessing_id="task1_rgb_60x80_no_aug_v1",
    horizontal_flip_probability=0.0,
    max_rotation_degrees=0.0,
    max_translation_fraction=0.0,
    scale_range=(1.0, 1.0),
    brightness_range=(1.0, 1.0),
    contrast_range=(1.0, 1.0),
)


def _ids_sha256(ids: Sequence[int]) -> str:
    payload = "".join(f"{int(product_id)}\n" for product_id in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_task1_normalization(
    training_rows: pd.DataFrame,
    *,
    validation_fold: int,
    root: str | Path = ROOT,
    config: Task1PreprocessingConfig = DEFAULT_TASK1_PREPROCESSING,
) -> Task1Normalization:
    """Fit RGB statistics using only the supplied development training rows."""
    required = {"id", "path", "partition", "cv_fold"}
    missing = required.difference(training_rows.columns)
    if missing:
        raise ValueError(f"training rows are missing columns: {sorted(missing)}")
    if training_rows.empty:
        raise ValueError("training rows must not be empty")
    if not training_rows["partition"].eq("development").all():
        raise ValueError("normalization may use development rows only")
    if validation_fold not in range(CV_FOLD_COUNT):
        raise ValueError(f"validation_fold must be in range({CV_FOLD_COUNT})")
    fold_values = pd.to_numeric(training_rows["cv_fold"], errors="raise").astype(int)
    if fold_values.eq(validation_fold).any():
        raise ValueError("normalization rows must not include the validation fold")
    if not fold_values.isin(range(CV_FOLD_COUNT)).all():
        raise ValueError(f"cv_fold values must be in range({CV_FOLD_COUNT})")
    if training_rows["id"].isna().any() or training_rows["id"].duplicated().any():
        raise ValueError("training row IDs must be present and unique")
    if training_rows["path"].astype(str).str.strip().eq("").any():
        raise ValueError("training image paths must not be blank")

    stats = StreamingStats(channels=3)
    project_root = Path(root)
    ordered_rows = training_rows.sort_values("id", kind="stable")
    for row in ordered_rows.itertuples(index=False):
        with Image.open(project_root / str(row.path)) as image:
            array, content_mask = transform_image_with_mask(
                image,
                image_size=config.image_size,
                pad_color=config.pad_color,
                normalize_range=True,
            )
        stats.update(array, content_mask=content_mask)

    return Task1Normalization(
        mean=tuple(float(value) for value in stats.mean),
        std=tuple(float(value) for value in stats.std),
        fitted_products=len(ordered_rows),
        fitted_ids_sha256=_ids_sha256(ordered_rows["id"].astype(int).tolist()),
    )


def _scale_canvas(
    image: Image.Image,
    content_mask: Image.Image,
    scale: float,
    pad_color: tuple[int, int, int],
) -> tuple[Image.Image, Image.Image]:
    if scale == 1.0:
        return image, content_mask
    width, height = image.size
    scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    scaled_image = image.resize(scaled_size, Image.Resampling.BILINEAR)
    scaled_mask = content_mask.resize(scaled_size, Image.Resampling.NEAREST)
    if scale < 1.0:
        image_canvas = Image.new("RGB", (width, height), pad_color)
        mask_canvas = Image.new("L", (width, height), 0)
        offset = ((width - scaled_size[0]) // 2, (height - scaled_size[1]) // 2)
        image_canvas.paste(scaled_image, offset)
        mask_canvas.paste(scaled_mask, offset)
        return image_canvas, mask_canvas

    left = (scaled_size[0] - width) // 2
    top = (scaled_size[1] - height) // 2
    crop = (left, top, left + width, top + height)
    return scaled_image.crop(crop), scaled_mask.crop(crop)


class Task1ImageTransform:
    """Load one image and apply deterministic or mildly random Task 1 transforms."""

    def __init__(
        self,
        normalization: Task1Normalization,
        *,
        training: bool,
        seed: int | None = None,
        epoch: int = 0,
        config: Task1PreprocessingConfig = DEFAULT_TASK1_PREPROCESSING,
    ) -> None:
        if training and seed is None:
            raise ValueError("a seed is required for the training transform")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.normalization = normalization
        self.training = training
        self.config = config
        self.seed = seed
        self.epoch = epoch

    def _random_for_path(self, path: str | Path) -> random.Random:
        """Return sample-local randomness that is stable across loading order and workers."""
        payload = f"{self.seed}\0{self.epoch}\0{Path(path).name}".encode("utf-8")
        sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(sample_seed)

    def _augment(
        self,
        image: Image.Image,
        content_mask: Image.Image,
        sample_random: random.Random,
    ) -> tuple[Image.Image, Image.Image]:
        config = self.config
        if sample_random.random() < config.horizontal_flip_probability:
            image = ImageOps.mirror(image)
            content_mask = ImageOps.mirror(content_mask)

        scale = sample_random.uniform(*config.scale_range)
        image, content_mask = _scale_canvas(image, content_mask, scale, config.pad_color)

        width, height = image.size
        angle = sample_random.uniform(
            -config.max_rotation_degrees,
            config.max_rotation_degrees,
        )
        translate = (
            round(sample_random.uniform(-1.0, 1.0) * config.max_translation_fraction * width),
            round(sample_random.uniform(-1.0, 1.0) * config.max_translation_fraction * height),
        )
        if angle != 0.0 or translate != (0, 0):
            image = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                translate=translate,
                fillcolor=config.pad_color,
            )
            content_mask = content_mask.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                translate=translate,
                fillcolor=0,
            )

        brightness = sample_random.uniform(*config.brightness_range)
        contrast = sample_random.uniform(*config.contrast_range)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        return image, content_mask

    def __call__(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as source:
            base, content = transform_image_with_mask(
                source,
                image_size=self.config.image_size,
                pad_color=self.config.pad_color,
                normalize_range=False,
            )
        image = Image.fromarray(base.astype(np.uint8), mode="RGB")
        content_mask = Image.fromarray(content.astype(np.uint8) * 255, mode="L")
        if self.training:
            image, content_mask = self._augment(
                image,
                content_mask,
                self._random_for_path(path),
            )

        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(self.normalization.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.normalization.std, dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / std
        array[np.asarray(content_mask) == 0] = 0.0
        return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def build_task1_training_transform(
    normalization: Task1Normalization,
    *,
    seed: int,
    epoch: int = 0,
    config: Task1PreprocessingConfig = DEFAULT_TASK1_PREPROCESSING,
) -> Task1ImageTransform:
    """Build Option B with seeded mild augmentation for training rows."""
    return Task1ImageTransform(
        normalization,
        training=True,
        seed=seed,
        epoch=epoch,
        config=config,
    )


def build_task1_validation_transform(
    normalization: Task1Normalization,
    *,
    config: Task1PreprocessingConfig = DEFAULT_TASK1_PREPROCESSING,
) -> Task1ImageTransform:
    """Build the deterministic validation transform with no fitted state changes."""
    return Task1ImageTransform(
        normalization,
        training=False,
        config=config,
    )

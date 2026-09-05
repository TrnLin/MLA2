"""Seeded geometry-only augmentation for Task 4 learned comparisons."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from PIL import Image

from fashion.config import RANDOM_SEED
from fashion.task4.preprocessing import (
    PreprocessedImage,
    PreprocessingContract,
    preprocess_image,
)

CANVAS_ASPECTS = ("normal", "wide_2_1", "tall_1_2")

__all__ = (
    "CANVAS_ASPECTS",
    "DEFAULT_GEOMETRY_POLICY",
    "GeometryPolicy",
    "GeometrySample",
    "apply_geometry",
    "sample_geometry",
)


@dataclass(frozen=True)
class GeometryPolicy:
    """The immutable, approved R3 geometry policy."""

    seed: int = RANDOM_SEED
    crop_min: float = 0.8
    crop_max: float = 1.0
    content_scale_min: float = 0.5
    content_scale_max: float = 1.0
    canvas_probabilities: tuple[float, float, float] = (0.50, 0.25, 0.25)
    contract: PreprocessingContract = field(
        default_factory=lambda: PreprocessingContract(width=240, height=320)
    )

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("geometry seed must be an integer")
        if not (0 < self.crop_min <= self.crop_max <= 1):
            raise ValueError("geometry crop bounds must lie in (0, 1]")
        if not (0 < self.content_scale_min <= self.content_scale_max <= 1):
            raise ValueError("geometry content-scale bounds must lie in (0, 1]")
        probabilities = np.asarray(self.canvas_probabilities, dtype=np.float64)
        if probabilities.shape != (3,) or np.any(probabilities < 0):
            raise ValueError("geometry canvas probabilities must contain three non-negative values")
        if not np.isclose(float(probabilities.sum()), 1.0):
            raise ValueError("geometry canvas probabilities must total one")
        if self.contract != PreprocessingContract(width=240, height=320):
            raise ValueError("geometry requires the frozen 240x320 preprocessing contract")


DEFAULT_GEOMETRY_POLICY = GeometryPolicy()


@dataclass(frozen=True)
class GeometrySample:
    """All random choices for one pair, retained for audit and testing."""

    augmented_source: str
    crop_height_fraction: float
    crop_width_fraction: float
    crop_top_fraction: float
    crop_left_fraction: float
    content_scale: float
    placement_x_fraction: float
    placement_y_fraction: float
    canvas_aspect: str

    def resolve_crop(
        self,
        content_bounds: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Turn sampled fractions into a valid integer crop inside known content."""
        top, left, bottom, right = (int(value) for value in content_bounds)
        height = bottom - top
        width = right - left
        if height <= 0 or width <= 0:
            raise ValueError("content bounds must have positive area")
        retained_fraction = max(
            0.8,
            min(1.0, self.crop_height_fraction * self.crop_width_fraction),
        )
        target_area = math.ceil(height * width * retained_fraction)
        candidates = [
            (candidate_height, math.ceil(target_area / candidate_height))
            for candidate_height in range(1, height + 1)
            if math.ceil(target_area / candidate_height) <= width
        ]
        crop_height, crop_width = min(
            candidates,
            key=lambda size: (
                abs(size[0] / height - self.crop_height_fraction)
                + abs(size[1] / width - self.crop_width_fraction),
                size,
            ),
        )
        available_y = height - crop_height
        available_x = width - crop_width
        crop_top = top + int(round(self.crop_top_fraction * available_y))
        crop_left = left + int(round(self.crop_left_fraction * available_x))
        return crop_top, crop_left, crop_top + crop_height, crop_left + crop_width


def _rng(*, seed: int, epoch: int, product_id: int) -> np.random.Generator:
    payload = f"task4-geometry-v1\0{seed}\0{epoch}\0{product_id}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def sample_geometry(
    *,
    product_id: int,
    epoch: int,
    policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY,
) -> GeometrySample:
    """Sample one pair transform without consulting mutable global RNG state."""
    if isinstance(product_id, bool) or not isinstance(product_id, Integral):
        raise ValueError("product_id must be an integer")
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    random = _rng(seed=int(policy.seed), epoch=int(epoch), product_id=int(product_id))
    retained_area = float(random.uniform(policy.crop_min, policy.crop_max))
    crop_height_fraction = float(random.uniform(retained_area, 1.0))
    return GeometrySample(
        augmented_source="teacher" if int(random.integers(0, 2)) == 0 else "v1",
        crop_height_fraction=crop_height_fraction,
        crop_width_fraction=retained_area / crop_height_fraction,
        crop_top_fraction=float(random.random()),
        crop_left_fraction=float(random.random()),
        content_scale=float(
            random.uniform(policy.content_scale_min, policy.content_scale_max)
        ),
        placement_x_fraction=float(random.random()),
        placement_y_fraction=float(random.random()),
        canvas_aspect=str(
            random.choice(CANVAS_ASPECTS, p=policy.canvas_probabilities)
        ),
    )


def _canvas_size(sample: GeometrySample, policy: GeometryPolicy) -> tuple[int, int]:
    if sample.canvas_aspect == "normal":
        return policy.contract.width, policy.contract.height
    if sample.canvas_aspect == "wide_2_1":
        return policy.contract.height, policy.contract.height // 2
    if sample.canvas_aspect == "tall_1_2":
        return policy.contract.width * 2 // 3, policy.contract.height
    raise ValueError(f"unknown canvas aspect: {sample.canvas_aspect!r}")


def _validate_cached_input(
    image: np.ndarray,
    content_bounds: tuple[int, int, int, int],
) -> None:
    if image.ndim != 3 or image.shape[2:] != (3,) or image.dtype != np.uint8:
        raise ValueError("geometry input must be an HWC uint8 RGB image")
    if len(content_bounds) != 4:
        raise ValueError("content bounds must contain four integers")
    top, left, bottom, right = (int(value) for value in content_bounds)
    if not (0 <= top < bottom <= image.shape[0] and 0 <= left < right <= image.shape[1]):
        raise ValueError("content bounds must lie inside the cached image")


def _preprocess_synthetic_mask(
    *,
    canvas_size: tuple[int, int],
    placement: tuple[int, int, int, int],
    contract: PreprocessingContract,
) -> np.ndarray:
    canvas_width, canvas_height = canvas_size
    source_mask = Image.new("L", canvas_size, 0)
    left, top, right, bottom = placement
    source_mask.paste(255, (left, top, right, bottom))
    scale = min(contract.width / canvas_width, contract.height / canvas_height)
    resized_width = max(1, round(canvas_width * scale))
    resized_height = max(1, round(canvas_height * scale))
    resized = source_mask.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    output = Image.new("L", (contract.width, contract.height), 0)
    output.paste(
        resized,
        (
            (contract.width - resized_width) // 2,
            (contract.height - resized_height) // 2,
        ),
    )
    return np.asarray(output) > 0


def apply_geometry(
    image: np.ndarray,
    content_bounds: tuple[int, int, int, int],
    *,
    sample: GeometrySample,
    policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY,
) -> PreprocessedImage:
    """Crop, resize, place on white, then apply the frozen preprocessing contract."""
    _validate_cached_input(image, content_bounds)
    crop_top, crop_left, crop_bottom, crop_right = sample.resolve_crop(content_bounds)
    crop = Image.fromarray(
        np.asarray(image[crop_top:crop_bottom, crop_left:crop_right]),
    )
    canvas_width, canvas_height = _canvas_size(sample, policy)
    fit_scale = min(canvas_width / crop.width, canvas_height / crop.height)
    resized_width = max(
        1,
        min(canvas_width, int(round(crop.width * fit_scale * sample.content_scale))),
    )
    resized_height = max(
        1,
        min(canvas_height, int(round(crop.height * fit_scale * sample.content_scale))),
    )
    resized = crop.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)
    available_x = canvas_width - resized_width
    available_y = canvas_height - resized_height
    x = int(round(sample.placement_x_fraction * available_x))
    y = int(round(sample.placement_y_fraction * available_y))
    canvas = Image.new("RGB", (canvas_width, canvas_height), policy.contract.pad_color)
    canvas.paste(resized, (x, y))
    transformed = preprocess_image(canvas, policy.contract)
    content_mask = _preprocess_synthetic_mask(
        canvas_size=(canvas_width, canvas_height),
        placement=(x, y, x + resized_width, y + resized_height),
        contract=policy.contract,
    )
    rows, columns = np.nonzero(content_mask)
    return PreprocessedImage(
        pixels=transformed.pixels,
        content_mask=content_mask,
        content_bounds=(
            int(rows.min()),
            int(columns.min()),
            int(rows.max()) + 1,
            int(columns.max()) + 1,
        ),
    )

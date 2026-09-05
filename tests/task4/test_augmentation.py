from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fashion.task4.augmentation import (
    CANVAS_ASPECTS,
    DEFAULT_GEOMETRY_POLICY,
    apply_geometry,
    sample_geometry,
)
from fashion.task4.preprocessing import PreprocessingContract, normalize_for_model


def _patterned_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((320, 240, 3), 255, dtype=np.uint8)
    rows = np.arange(240, dtype=np.uint8)[:, None]
    columns = np.arange(160, dtype=np.uint8)[None, :]
    image[40:280, 40:200, 0] = rows
    image[40:280, 40:200, 1] = columns
    image[40:280, 40:200, 2] = 37
    return image, (40, 40, 280, 200)


def test_geometry_policy_is_frozen_and_matches_the_approved_values() -> None:
    policy = DEFAULT_GEOMETRY_POLICY

    assert dataclasses.is_dataclass(policy)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.seed = 1  # type: ignore[misc]
    assert policy.seed == 2753
    assert (policy.crop_min, policy.crop_max) == (0.8, 1.0)
    assert (policy.content_scale_min, policy.content_scale_max) == (0.5, 1.0)
    assert policy.canvas_probabilities == (0.50, 0.25, 0.25)
    assert policy.contract == PreprocessingContract(width=240, height=320)


def test_geometry_sampling_uses_only_seed_epoch_and_product_id() -> None:
    np.random.seed(999)
    first = sample_geometry(product_id=42, epoch=3)
    np.random.seed(1)
    second = sample_geometry(product_id=42, epoch=3)

    assert second == first
    assert sample_geometry(product_id=42, epoch=4) != first
    assert first.augmented_source in {"teacher", "v1"}
    assert 0.8 <= first.crop_height_fraction <= 1.0
    assert 0.8 <= first.crop_width_fraction <= 1.0
    assert 0.8 <= first.crop_height_fraction * first.crop_width_fraction <= 1.0
    assert 0.5 <= first.content_scale <= 1.0
    assert 0.0 <= first.crop_top_fraction <= 1.0
    assert 0.0 <= first.crop_left_fraction <= 1.0
    assert 0.0 <= first.placement_x_fraction <= 1.0
    assert 0.0 <= first.placement_y_fraction <= 1.0


def test_geometry_sampling_reaches_all_frozen_canvas_branches() -> None:
    observed = {
        sample_geometry(product_id=product_id, epoch=0).canvas_aspect
        for product_id in range(200)
    }

    assert observed == set(CANVAS_ASPECTS)


@pytest.mark.parametrize("canvas_aspect", CANVAS_ASPECTS)
def test_geometry_outputs_frozen_shape_mask_bounds_and_preserves_rgb(
    canvas_aspect: str,
) -> None:
    image, bounds = _patterned_image()
    sample = next(
        sample_geometry(product_id=product_id, epoch=0)
        for product_id in range(500)
        if sample_geometry(product_id=product_id, epoch=0).canvas_aspect == canvas_aspect
    )

    transformed = apply_geometry(image, bounds, sample=sample)

    assert transformed.pixels.shape == (320, 240, 3)
    assert transformed.pixels.dtype == np.uint8
    assert transformed.content_mask.shape == (320, 240)
    assert transformed.content_mask.dtype == np.bool_
    assert transformed.content_mask.any()
    top, left, bottom, right = transformed.content_bounds
    assert 0 <= top < bottom <= 320
    assert 0 <= left < right <= 240
    assert np.count_nonzero(np.abs(transformed.pixels[..., 2].astype(int) - 37) <= 2) > 100
    assert np.any(transformed.pixels[..., 0] != transformed.pixels[..., 1])


def test_geometry_crop_and_placement_stay_inside_the_known_content_and_canvas() -> None:
    image, bounds = _patterned_image()
    sample = sample_geometry(product_id=84, epoch=2)

    transformed = apply_geometry(image, bounds, sample=sample)

    crop_top, crop_left, crop_bottom, crop_right = sample.resolve_crop(bounds)
    assert bounds[0] <= crop_top < crop_bottom <= bounds[2]
    assert bounds[1] <= crop_left < crop_right <= bounds[3]
    assert (crop_bottom - crop_top) / (bounds[2] - bounds[0]) == pytest.approx(
        sample.crop_height_fraction,
        abs=1 / (bounds[2] - bounds[0]),
    )
    assert (crop_right - crop_left) / (bounds[3] - bounds[1]) == pytest.approx(
        sample.crop_width_fraction,
        abs=1 / (bounds[3] - bounds[1]),
    )
    assert transformed.content_mask.any()
    assert np.all(transformed.pixels[~transformed.content_mask] == 255)


def test_geometry_marks_all_synthetic_padding_for_exact_zero_normalization() -> None:
    image, bounds = _patterned_image()
    sample = next(
        sample_geometry(product_id=product_id, epoch=1)
        for product_id in range(500)
        if sample_geometry(product_id=product_id, epoch=1).canvas_aspect == "normal"
        and sample_geometry(product_id=product_id, epoch=1).content_scale < 0.8
    )

    transformed = apply_geometry(image, bounds, sample=sample)
    normalized = normalize_for_model(
        transformed,
        mean=(0.25, 0.5, 0.75),
        std=(0.25, 0.25, 0.25),
    )

    assert transformed.content_mask.sum() < transformed.content_mask.size
    assert np.all(normalized[~transformed.content_mask] == 0)


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 0, 1, 1),
        (0, 0, 3, 3),
        (2, 4, 9, 8),
        (40, 40, 280, 200),
    ],
)
def test_resolved_crops_retain_eighty_to_one_hundred_percent_content_area(
    bounds: tuple[int, int, int, int],
) -> None:
    content_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    for epoch in range(4):
        for product_id in range(50):
            sample = sample_geometry(product_id=product_id, epoch=epoch)
            top, left, bottom, right = sample.resolve_crop(bounds)
            retained_area = (bottom - top) * (right - left)

            assert 0.8 <= retained_area / content_area <= 1.0


def test_integer_crop_resolution_repairs_the_lower_area_boundary() -> None:
    sample = dataclasses.replace(
        sample_geometry(product_id=1, epoch=0),
        crop_height_fraction=0.8,
        crop_width_fraction=0.8,
    )

    top, left, bottom, right = sample.resolve_crop((0, 0, 3, 3))

    assert (bottom - top) * (right - left) >= 8


@pytest.mark.parametrize(
    ("image_shape", "bounds", "message"),
    [
        ((10, 10, 1), (0, 0, 10, 10), "RGB"),
        ((320, 240, 3), (0, 0, 0, 10), "bounds"),
        ((320, 240, 3), (-1, 0, 10, 10), "bounds"),
    ],
)
def test_geometry_rejects_malformed_cached_inputs(
    image_shape: tuple[int, ...],
    bounds: tuple[int, int, int, int],
    message: str,
) -> None:
    image = np.zeros(image_shape, dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        apply_geometry(image, bounds, sample=sample_geometry(product_id=1, epoch=0))

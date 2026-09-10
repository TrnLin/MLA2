from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fashion.task4_evaluation.conditions import QUERY_CONDITIONS, apply_query_condition


def _image() -> Image.Image:
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[20:60, 15:45] = (120, 60, 200)
    return Image.fromarray(pixels, mode="RGB")


def test_condition_order_is_frozen() -> None:
    assert QUERY_CONDITIONS == (
        "clean",
        "jpeg_quality_85",
        "gaussian_blur_radius_1",
        "brightness_0_85",
        "brightness_1_15",
        "wide_canvas",
        "tall_canvas",
    )


def test_clean_returns_identical_pixels() -> None:
    source = _image()
    result = apply_query_condition(source, "clean")
    assert np.array_equal(np.asarray(result), np.asarray(source))


@pytest.mark.parametrize("condition", QUERY_CONDITIONS)
def test_every_condition_returns_a_deterministic_rgb_image(condition: str) -> None:
    first = np.asarray(apply_query_condition(_image(), condition))
    second = np.asarray(apply_query_condition(_image(), condition))
    assert np.array_equal(first, second)
    assert first.ndim == 3
    assert first.shape[2] == 3
    assert first.dtype == np.uint8


def test_brightness_conditions_move_the_mean_in_opposite_directions() -> None:
    baseline = np.asarray(apply_query_condition(_image(), "clean"), dtype=float).mean()
    darker = np.asarray(apply_query_condition(_image(), "brightness_0_85"), dtype=float).mean()
    brighter = np.asarray(apply_query_condition(_image(), "brightness_1_15"), dtype=float).mean()
    assert darker < baseline < brighter


def test_jpeg_condition_changes_pixels_but_keeps_geometry() -> None:
    source = _image()
    result = apply_query_condition(source, "jpeg_quality_85")
    assert result.size == source.size
    assert not np.array_equal(np.asarray(result), np.asarray(source))


def test_blur_condition_reduces_local_contrast() -> None:
    source = np.asarray(_image(), dtype=float)
    blurred = np.asarray(apply_query_condition(_image(), "gaussian_blur_radius_1"), dtype=float)
    assert blurred.std() < source.std()


def test_canvas_conditions_grow_the_expected_axis_without_cropping() -> None:
    source = _image()
    wide = apply_query_condition(source, "wide_canvas")
    tall = apply_query_condition(source, "tall_canvas")
    assert wide.width == max(source.width, 2 * source.height)
    assert wide.height == source.height
    assert tall.width == source.width
    assert tall.height == max(source.height, 2 * source.width)


def test_unknown_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown query condition"):
        apply_query_condition(_image(), "rotate_90")

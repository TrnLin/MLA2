from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fashion.task1.classical import (
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    Task1HogSpec,
    extract_task1_hog,
)


def test_hog_specs_have_fixed_ids_and_feature_lengths() -> None:
    assert TASK1_HOG_COARSE.hog_id == "task1_gray_hog_ppc16_v1"
    assert TASK1_HOG_COARSE.expected_features == 288
    assert TASK1_HOG_COARSE.pixels_per_cell == (16, 16)
    assert TASK1_HOG_FINE.hog_id == "task1_gray_hog_ppc10_v1"
    assert TASK1_HOG_FINE.expected_features == 1260
    assert TASK1_HOG_FINE.pixels_per_cell == (10, 10)
    for spec in (TASK1_HOG_COARSE, TASK1_HOG_FINE):
        assert spec.orientations == 9
        assert spec.cells_per_block == (2, 2)
        assert spec.block_norm == "L2-Hys"
        assert spec.transform_sqrt is True
        assert spec.image_size == (80, 60)


@pytest.mark.parametrize("spec", [TASK1_HOG_COARSE, TASK1_HOG_FINE])
def test_extract_task1_hog_is_float32_deterministic_and_fixed_width(
    tmp_path: Path, spec: Task1HogSpec
) -> None:
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[20:60, 15:45] = (220, 80, 40)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels, mode="RGB").save(path)

    first = extract_task1_hog(path, spec)
    second = extract_task1_hog(path, spec)

    assert first.shape == (spec.expected_features,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_hog_spec_rejects_unapproved_geometry() -> None:
    with pytest.raises(ValueError, match="expected feature count"):
        Task1HogSpec(
            hog_id="bad",
            pixels_per_cell=(8, 8),
            expected_features=1,
        )

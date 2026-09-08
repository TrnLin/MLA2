from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.task1.image_contract import TASK1_TENSOR_SHAPE
from fashion.task1.preprocessing import (
    TASK1_CONTROL_PREPROCESSING,
    Task1Normalization,
    Task1PreprocessingConfig,
    build_task1_training_transform,
    build_task1_validation_transform,
    fit_task1_normalization,
)


def _save_image(path, mode: str, size: tuple[int, int], value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, value).save(path)


def test_validation_transform_converts_letterboxed_grayscale_to_normalized_chw(tmp_path):
    image_path = tmp_path / "narrow.jpg"
    _save_image(image_path, "L", (30, 80), 0)
    normalization = Task1Normalization(
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        fitted_products=2,
        fitted_ids_sha256="0" * 64,
    )

    transformed = build_task1_validation_transform(normalization)(image_path)

    assert transformed.shape == TASK1_TENSOR_SHAPE
    assert transformed.dtype == np.float32
    assert np.all(transformed[:, :, :15] == 0.0)
    assert np.all(transformed[:, :, 15:45] == -2.0)
    assert np.all(transformed[:, :, 45:] == 0.0)


def test_validation_transform_does_not_apply_training_flip(tmp_path):
    image_path = tmp_path / "asymmetric.png"
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[:, :30] = (255, 0, 0)
    Image.fromarray(pixels, mode="RGB").save(image_path)
    config = Task1PreprocessingConfig(
        horizontal_flip_probability=1.0,
        max_rotation_degrees=0.0,
        max_translation_fraction=0.0,
        scale_range=(1.0, 1.0),
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
    )
    normalization = Task1Normalization(
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        fitted_products=1,
        fitted_ids_sha256="1" * 64,
    )

    transformed = build_task1_validation_transform(normalization, config=config)(image_path)

    assert np.all(transformed[0, :, :30] == 1.0)
    assert np.all(transformed[0, :, 30:] == 0.0)


def test_control_preprocessing_has_no_random_training_changes(tmp_path):
    image_path = tmp_path / "pattern.png"
    pixels = np.arange(80 * 60 * 3, dtype=np.uint16).reshape(80, 60, 3) % 256
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(image_path)
    normalization = Task1Normalization(
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        fitted_products=1,
        fitted_ids_sha256="5" * 64,
    )
    control = build_task1_training_transform(
        normalization,
        seed=2753,
        config=TASK1_CONTROL_PREPROCESSING,
    )(image_path)
    validation = build_task1_validation_transform(
        normalization,
        config=TASK1_CONTROL_PREPROCESSING,
    )(image_path)

    assert TASK1_CONTROL_PREPROCESSING.preprocessing_id != (
        Task1PreprocessingConfig().preprocessing_id
    )
    np.testing.assert_array_equal(control, validation)


def test_training_transform_applies_configured_horizontal_flip(tmp_path):
    image_path = tmp_path / "asymmetric.png"
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[:, :30] = (255, 0, 0)
    Image.fromarray(pixels, mode="RGB").save(image_path)
    config = Task1PreprocessingConfig(
        horizontal_flip_probability=1.0,
        max_rotation_degrees=0.0,
        max_translation_fraction=0.0,
        scale_range=(1.0, 1.0),
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
    )
    normalization = Task1Normalization(
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        fitted_products=1,
        fitted_ids_sha256="2" * 64,
    )

    transformed = build_task1_training_transform(
        normalization,
        seed=2753,
        config=config,
    )(image_path)

    assert np.all(transformed[0, :, :30] == 0.0)
    assert np.all(transformed[0, :, 30:] == 1.0)


def test_training_transform_is_reproducible_for_the_same_seed(tmp_path):
    image_path = tmp_path / "pattern.png"
    pixels = np.arange(80 * 60 * 3, dtype=np.uint16).reshape(80, 60, 3) % 256
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(image_path)
    normalization = Task1Normalization(
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        fitted_products=1,
        fitted_ids_sha256="3" * 64,
    )

    first = build_task1_training_transform(normalization, seed=2753)(image_path)
    second = build_task1_training_transform(normalization, seed=2753)(image_path)

    np.testing.assert_array_equal(first, second)


def test_training_transform_is_independent_of_sample_order(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_pixels = np.arange(80 * 60 * 3, dtype=np.uint16).reshape(80, 60, 3) % 256
    second_pixels = np.flip(first_pixels, axis=1)
    Image.fromarray(first_pixels.astype(np.uint8), mode="RGB").save(first_path)
    Image.fromarray(second_pixels.astype(np.uint8), mode="RGB").save(second_path)
    normalization = Task1Normalization(
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        fitted_products=2,
        fitted_ids_sha256="4" * 64,
    )
    first_order = build_task1_training_transform(normalization, seed=2753, epoch=3)
    second_order = build_task1_training_transform(normalization, seed=2753, epoch=3)

    first_a = first_order(first_path)
    first_b = first_order(second_path)
    second_b = second_order(second_path)
    second_a = second_order(first_path)

    np.testing.assert_array_equal(first_a, second_a)
    np.testing.assert_array_equal(first_b, second_b)


def test_fit_normalization_uses_only_supplied_development_rows(tmp_path):
    first_path = "images/1.png"
    second_path = "images/2.png"
    ignored_path = "images/999.png"
    _save_image(tmp_path / first_path, "RGB", (30, 80), (0, 64, 128))
    _save_image(tmp_path / second_path, "RGB", (30, 80), (255, 192, 64))
    _save_image(tmp_path / ignored_path, "RGB", (60, 80), (255, 255, 255))
    rows = pd.DataFrame(
        [
            {"id": 2, "path": second_path, "partition": "development", "cv_fold": 2},
            {"id": 1, "path": first_path, "partition": "development", "cv_fold": 1},
        ]
    )

    fitted = fit_task1_normalization(rows, validation_fold=0, root=tmp_path)

    assert fitted.mean == pytest.approx((0.5, 128 / 255, 96 / 255), abs=1e-6)
    assert fitted.std == pytest.approx((0.5, 64 / 255, 32 / 255), abs=1e-6)
    assert fitted.fitted_products == 2
    assert len(fitted.fitted_ids_sha256) == 64


def test_fit_normalization_rejects_rows_from_validation_fold(tmp_path):
    image_path = "images/1.png"
    _save_image(tmp_path / image_path, "RGB", (60, 80), (0, 0, 0))
    rows = pd.DataFrame(
        [
            {
                "id": 1,
                "path": image_path,
                "partition": "development",
                "cv_fold": 0,
            }
        ]
    )

    with pytest.raises(ValueError, match="validation fold"):
        fit_task1_normalization(rows, validation_fold=0, root=tmp_path)


@pytest.mark.parametrize("partition", ["holdout", "quarantine"])
def test_fit_normalization_rejects_protected_rows(tmp_path, partition):
    image_path = "images/1.png"
    _save_image(tmp_path / image_path, "RGB", (60, 80), (0, 0, 0))
    rows = pd.DataFrame(
        [{"id": 1, "path": image_path, "partition": partition, "cv_fold": 1}]
    )

    with pytest.raises(ValueError, match="development"):
        fit_task1_normalization(rows, validation_fold=0, root=tmp_path)

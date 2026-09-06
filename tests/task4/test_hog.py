from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fashion.task4.cache import DevelopmentImageCache


def _load_hog():
    path = Path(__file__).resolve().parents[2] / "src/fashion/task4/hog.py"
    assert path.is_file(), "Task 4 HOG production module is missing"
    spec = importlib.util.spec_from_file_location("fashion.task4.hog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _shape_image(*, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.full((320, 240, 3), 255, dtype=np.uint8)
    mask = np.ones((320, 240), dtype=bool)
    if horizontal:
        pixels[:160] = 0
    else:
        pixels[:, :120] = 0
    return pixels, mask


def test_hog_has_exact_dimension_dtype_unit_norm_and_finite_values() -> None:
    hog = _load_hog()
    pixels, mask = _shape_image(horizontal=False)

    descriptor = hog.extract_hog(pixels, mask)

    assert descriptor.shape == (1_728,)
    assert descriptor.dtype == np.float32
    assert np.isfinite(descriptor).all()
    assert np.linalg.norm(descriptor) == pytest.approx(1.0, abs=1e-6)


def test_hog_is_deterministic() -> None:
    hog = _load_hog()
    rng = np.random.default_rng(2753)
    pixels = rng.integers(0, 256, size=(320, 240, 3), dtype=np.uint8)
    mask = np.ones((320, 240), dtype=bool)

    first = hog.extract_hog(pixels, mask)
    second = hog.extract_hog(pixels.copy(), mask.copy())

    assert np.array_equal(first, second)


def test_hog_responds_to_edge_orientation() -> None:
    hog = _load_hog()
    vertical_pixels, mask = _shape_image(horizontal=False)
    horizontal_pixels, _ = _shape_image(horizontal=True)

    vertical = hog.extract_hog(vertical_pixels, mask)
    horizontal = hog.extract_hog(horizontal_pixels, mask)

    assert float(vertical @ horizontal) < 0.25


def test_hog_ignores_pixels_outside_the_content_mask() -> None:
    hog = _load_hog()
    rng = np.random.default_rng(9)
    mask = np.zeros((320, 240), dtype=bool)
    mask[48:272, 32:208] = True
    content = rng.integers(0, 256, size=(224, 176, 3), dtype=np.uint8)
    white_padding = np.full((320, 240, 3), 255, dtype=np.uint8)
    noisy_padding = rng.integers(0, 256, size=(320, 240, 3), dtype=np.uint8)
    white_padding[mask] = content.reshape(-1, 3)
    noisy_padding[mask] = content.reshape(-1, 3)

    expected = hog.extract_hog(white_padding, mask)
    observed = hog.extract_hog(noisy_padding, mask)

    assert np.array_equal(observed, expected)


@pytest.mark.parametrize(
    ("pixels", "mask", "message"),
    [
        (
            np.zeros((319, 240, 3), dtype=np.uint8),
            np.ones((319, 240), dtype=bool),
            "320x240",
        ),
        (
            np.zeros((320, 240, 3), dtype=np.float32),
            np.ones((320, 240), dtype=bool),
            "uint8",
        ),
        (
            np.zeros((320, 240, 3), dtype=np.uint8),
            np.ones((320, 239), dtype=bool),
            "matching",
        ),
        (
            np.zeros((320, 240, 3), dtype=np.uint8),
            np.ones((320, 240), dtype=np.uint8),
            "boolean",
        ),
        (
            np.zeros((320, 240, 3), dtype=np.uint8),
            np.zeros((320, 240), dtype=bool),
            "content pixel",
        ),
    ],
)
def test_hog_rejects_invalid_inputs(
    pixels: np.ndarray,
    mask: np.ndarray,
    message: str,
) -> None:
    hog = _load_hog()

    with pytest.raises(ValueError, match=message):
        hog.extract_hog(pixels, mask)


def test_hog_method_and_fingerprint_encode_the_exact_config() -> None:
    hog = _load_hog()

    assert hog.HOG_DESCRIPTOR_DIM == 1_728
    assert hog.HOG_METHOD == "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1"
    assert hog.HOG_METHOD != "spatial-hsv-edge-4x4-v2"
    assert hog.HOG_CONFIG.to_dict() == {
        "schema_version": "1.0.0",
        "input_height": 320,
        "input_width": 240,
        "grayscale": "rec709-luminance",
        "gaussian_kernel": [1, 4, 6, 4, 1],
        "gaussian_kernel_normalizer": 16,
        "unsigned_orientation_bins": 8,
        "cell_size_pixels": 32,
        "block_size_cells": 2,
        "block_stride_cells": 1,
        "l2_hys_clip": 0.2,
        "mask_policy": "normalized-blur-and-valid-central-differences",
        "zero_gradient_policy": "canonical-unit-sentinel",
        "descriptor_dimension": 1_728,
    }
    assert len(hog.HOG_CONFIG_FINGERPRINT) == 64
    assert hog.HOG_CONFIG.fingerprint() == hog.HOG_CONFIG_FINGERPRINT
    assert (
        replace(hog.HOG_CONFIG, l2_hys_clip=0.1).fingerprint()
        != hog.HOG_CONFIG_FINGERPRINT
    )


def test_hog_maps_a_valid_zero_gradient_image_to_a_unit_sentinel() -> None:
    hog = _load_hog()
    pixels = np.full((320, 240, 3), 127, dtype=np.uint8)
    mask = np.ones((320, 240), dtype=bool)

    descriptor = hog.extract_hog(pixels, mask)

    assert descriptor.dtype == np.float32
    assert np.array_equal(
        descriptor,
        np.pad(np.ones(1, dtype=np.float32), (0, 1_727)),
    )


def _cache(tmp_path: Path) -> DevelopmentImageCache:
    cache_dir = tmp_path / "images"
    cache_dir.mkdir()
    pixels, _ = _shape_image(horizontal=False)
    images = np.stack([pixels, np.flip(pixels, axis=0)]).astype(np.uint8)
    ids = np.array([1, 2], dtype=np.int64)
    bounds = np.array([[0, 0, 320, 240], [0, 0, 320, 240]], dtype=np.int32)
    manifest = {
        "schema_version": "1.0.0",
        "scope": "development",
        "source": "teacher",
        "rows": 2,
        "id_sha256": "a" * 64,
        "source_fingerprint": "b" * 64,
        "contract": {
            "width": 240,
            "height": 320,
            "pad_color": [255, 255, 255],
            "colour_mode": "RGB",
            "resize": "aspect_preserving_letterbox",
            "resample": "LANCZOS",
        },
    }
    return DevelopmentImageCache(
        cache_dir=cache_dir,
        ids=ids,
        images=images,
        content_bounds=bounds,
        manifest=manifest,
    )


def test_hog_feature_cache_identity_changes_with_descriptor_config(
    tmp_path: Path,
) -> None:
    hog = _load_hog()
    cache = _cache(tmp_path)

    first = hog.ensure_hog_feature_index(
        cache,
        cache_root=tmp_path / "features",
        workers=1,
    )
    reused = hog.ensure_hog_feature_index(
        cache,
        cache_root=tmp_path / "features",
        workers=1,
    )
    changed = hog.ensure_hog_feature_index(
        cache,
        cache_root=tmp_path / "features",
        workers=1,
        config=replace(hog.HOG_CONFIG, l2_hys_clip=0.1),
    )

    assert first.cache_dir == reused.cache_dir
    assert first.cache_dir != changed.cache_dir
    assert first.index.checkpoint_fingerprint is None
    assert first.index.config_fingerprint is None
    assert first.index.descriptor_fingerprint == hog.HOG_CONFIG_FINGERPRINT
    assert first.manifest["image_cache_manifest_sha256"]
    assert first.manifest["source_fingerprint"] == "b" * 64
    assert first.manifest["descriptor_fingerprint"] == hog.HOG_CONFIG_FINGERPRINT
    assert first.index.features.shape == (2, 1_728)

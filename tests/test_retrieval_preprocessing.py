from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.retrieval.cache import (
    ensure_development_image_cache,
    fit_cached_fold_rgb_statistics,
    load_development_image_cache,
)
from fashion.retrieval.preprocessing import (
    PreprocessingContract,
    fit_fold_rgb_statistics,
    normalize_for_model,
    preprocess_image,
)


def test_preprocess_composites_transparency_on_white() -> None:
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 0))

    result = preprocess_image(image, PreprocessingContract(width=1, height=1))

    assert result.pixels.tolist() == [[[255, 255, 255]]]
    assert result.content_mask.tolist() == [[True]]


def test_preprocess_corrects_exif_before_letterboxing() -> None:
    image = Image.new("RGB", (2, 3), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    image.getexif()[274] = 6

    result = preprocess_image(image, PreprocessingContract(width=3, height=2))

    assert result.pixels.shape == (2, 3, 3)
    assert result.content_mask.all()
    assert result.pixels[0, 2].tolist() == [255, 0, 0]


def test_preprocess_converts_grayscale_to_rgb_uint8() -> None:
    image = Image.new("L", (3, 4), 64)

    result = preprocess_image(image, PreprocessingContract(width=3, height=4))

    assert result.pixels.shape == (4, 3, 3)
    assert result.pixels.dtype == np.uint8
    assert np.all(result.pixels == 64)


@pytest.mark.parametrize(
    ("source_size", "expected_bounds"),
    [
        ((4, 2), (2, 0, 6, 8)),
        ((2, 4), (0, 2, 8, 6)),
    ],
)
def test_preprocess_preserves_wide_and_tall_geometry_with_masks(
    source_size: tuple[int, int],
    expected_bounds: tuple[int, int, int, int],
) -> None:
    image = Image.new("RGB", source_size, (0, 0, 0))

    result = preprocess_image(image, PreprocessingContract(width=8, height=8))

    assert result.content_bounds == expected_bounds
    top, left, bottom, right = expected_bounds
    expected_mask = np.zeros((8, 8), dtype=bool)
    expected_mask[top:bottom, left:right] = True
    assert np.array_equal(result.content_mask, expected_mask)
    assert np.all(result.pixels[result.content_mask] == 0)
    assert np.all(result.pixels[~result.content_mask] == 255)


def test_fold_statistics_reject_non_development_rows_before_opening_pixels(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["holdout"],
            "cv_fold": [0],
            "source_path": ["does-not-exist.jpg"],
        }
    )

    with pytest.raises(ValueError, match="development"):
        fit_fold_rgb_statistics(
            frame,
            path_column="source_path",
            contract=PreprocessingContract(width=1, height=1),
            validation_fold=1,
            root=tmp_path,
        )


def test_fold_statistics_use_only_training_content_pixels(tmp_path) -> None:
    for name, colour in (
        ("black.png", (0, 0, 0)),
        ("white.png", (255, 255, 255)),
        ("validation.png", (255, 0, 0)),
    ):
        Image.new("RGB", (1, 1), colour).save(tmp_path / name)
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "partition": ["development"] * 3,
            "cv_fold": [0, 2, 1],
            "source_path": ["black.png", "white.png", "validation.png"],
        }
    )

    result = fit_fold_rgb_statistics(
        frame,
        path_column="source_path",
        contract=PreprocessingContract(width=1, height=1),
        validation_fold=1,
        root=tmp_path,
    )

    assert result["validation_fold"] == 1
    assert result["training_rows"] == 2
    assert result["mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["std"] == pytest.approx([0.5, 0.5, 0.5])
    assert isinstance(result["training_id_sha256"], str)
    assert len(result["training_id_sha256"]) == 64


def test_model_normalization_uses_saved_values_and_zeroes_padding() -> None:
    transformed = preprocess_image(
        Image.new("RGB", (1, 2), (64, 96, 128)),
        PreprocessingContract(width=4, height=4),
    )

    normalized = normalize_for_model(
        transformed,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
    )

    assert normalized.dtype == np.float32
    assert np.all(normalized[~transformed.content_mask] == 0)
    assert normalized[transformed.content_mask][0].tolist() == pytest.approx(
        [
            (64 / 255 - 0.5) / 0.25,
            (96 / 255 - 0.5) / 0.25,
            (128 / 255 - 0.5) / 0.25,
        ],
        abs=1e-6,
    )


def _cache_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [2, 1],
            "partition": ["development", "development"],
            "source_path": ["white.png", "black.png"],
            "source_sha256": ["white-sha", "black-sha"],
        }
    )


def test_cache_rejects_holdout_before_opening_pixels(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["holdout"],
            "source_path": ["does-not-exist.png"],
            "source_sha256": ["missing"],
        }
    )

    with pytest.raises(ValueError, match="development"):
        ensure_development_image_cache(
            frame,
            path_column="source_path",
            sha_column="source_sha256",
            source="teacher",
            contract=PreprocessingContract(width=2, height=2),
            cache_root=tmp_path / "cache",
            root=tmp_path,
        )


def test_cache_requires_a_content_hash_column(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["development"],
            "source_path": ["does-not-exist.png"],
        }
    )

    with pytest.raises(ValueError, match="hash"):
        ensure_development_image_cache(
            frame,
            path_column="source_path",
            sha_column="",
            source="teacher",
            contract=PreprocessingContract(width=2, height=2),
            cache_root=tmp_path / "cache",
            root=tmp_path,
        )


def test_cache_sorts_and_round_trips_lossless_pixels_and_bounds(tmp_path) -> None:
    Image.new("RGB", (1, 1), (0, 0, 0)).save(tmp_path / "black.png")
    Image.new("RGB", (1, 1), (255, 255, 255)).save(tmp_path / "white.png")

    cache = ensure_development_image_cache(
        _cache_frame(),
        path_column="source_path",
        sha_column="source_sha256",
        source="teacher",
        contract=PreprocessingContract(width=2, height=2),
        cache_root=tmp_path / "cache",
        root=tmp_path,
    )
    reopened = load_development_image_cache(cache.cache_dir)

    assert reopened.ids.tolist() == [1, 2]
    assert reopened.images.dtype == np.uint8
    assert isinstance(reopened.images, np.memmap)
    assert np.all(reopened.images[0] == 0)
    assert np.all(reopened.images[1] == 255)
    assert reopened.content_bounds.tolist() == [[0, 0, 2, 2], [0, 0, 2, 2]]
    assert reopened.manifest["array_shape"] == [2, 2, 2, 3]


def test_cache_reuses_only_an_exact_manifest_match(tmp_path) -> None:
    Image.new("RGB", (1, 1), (0, 0, 0)).save(tmp_path / "black.png")
    Image.new("RGB", (1, 1), (255, 255, 255)).save(tmp_path / "white.png")
    arguments = {
        "path_column": "source_path",
        "sha_column": "source_sha256",
        "source": "teacher",
        "contract": PreprocessingContract(width=2, height=2),
        "cache_root": tmp_path / "cache",
        "root": tmp_path,
    }

    first = ensure_development_image_cache(_cache_frame(), **arguments)
    manifest_path = first.cache_dir / "manifest.json"
    first_mtime = manifest_path.stat().st_mtime_ns
    second = ensure_development_image_cache(_cache_frame(), **arguments)

    assert second.cache_dir == first.cache_dir
    assert manifest_path.stat().st_mtime_ns == first_mtime


def test_cache_rebuilds_when_source_fingerprint_changes(tmp_path) -> None:
    Image.new("RGB", (1, 1), (0, 0, 0)).save(tmp_path / "black.png")
    Image.new("RGB", (1, 1), (255, 255, 255)).save(tmp_path / "white.png")
    arguments = {
        "path_column": "source_path",
        "sha_column": "source_sha256",
        "source": "teacher",
        "contract": PreprocessingContract(width=2, height=2),
        "cache_root": tmp_path / "cache",
        "root": tmp_path,
    }

    first = ensure_development_image_cache(_cache_frame(), **arguments)
    changed = _cache_frame()
    changed.loc[changed["id"].eq(1), "source_sha256"] = "new-black-sha"
    Image.new("RGB", (1, 1), (255, 0, 0)).save(tmp_path / "black.png")
    second = ensure_development_image_cache(changed, **arguments)

    assert second.manifest["source_fingerprint"] != first.manifest["source_fingerprint"]
    assert second.images[0, 0, 0].tolist() == [255, 0, 0]


def test_cached_statistics_use_only_training_content_pixels(tmp_path) -> None:
    for name, colour in (
        ("black.png", (0, 0, 0)),
        ("white.png", (255, 255, 255)),
        ("validation.png", (255, 0, 0)),
    ):
        Image.new("RGB", (1, 1), colour).save(tmp_path / name)
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "partition": ["development"] * 3,
            "cv_fold": [0, 2, 1],
            "source_path": ["black.png", "white.png", "validation.png"],
            "source_sha256": ["black", "white", "validation"],
        }
    )
    cache = ensure_development_image_cache(
        frame,
        path_column="source_path",
        sha_column="source_sha256",
        source="teacher",
        contract=PreprocessingContract(width=1, height=1),
        cache_root=tmp_path / "cache",
        root=tmp_path,
    )

    result = fit_cached_fold_rgb_statistics(cache, frame, validation_fold=1)

    assert result["training_rows"] == 2
    assert result["mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["std"] == pytest.approx([0.5, 0.5, 0.5])

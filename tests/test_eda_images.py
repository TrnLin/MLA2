from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.eda_images import (
    exact_duplicate_groups,
    measure_image,
    measure_images,
    near_duplicate_candidates,
    paired_image_comparison,
    stratified_sample,
)


def _save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (12, 8), color).save(path)


def _save_edge_heavy(path: Path) -> None:
    pixels = np.indices((8, 12)).sum(axis=0) % 2 * 255
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)


def test_stratified_sample_is_seeded_sorted_and_preserves_classes_when_possible() -> None:
    """Dropping a rare class or using input order would hide reproducibility problems."""
    frame = pd.DataFrame(
        {
            "id": [9, 1, 7, 3, 8, 2, 6, 4, 5],
            "articleType": ["common"] * 6 + ["rare-a", "rare-b", "rare-b"],
        }
    )

    first = stratified_sample(frame, "articleType", limit=5, seed=2753)
    second = stratified_sample(
        frame.sample(frac=1.0, random_state=99), "articleType", limit=5, seed=2753
    )

    assert first["id"].tolist() == sorted(first["id"].tolist())
    assert first["id"].tolist() == second["id"].tolist()
    assert len(first) == 5
    assert set(first["articleType"]) == {"common", "rare-a", "rare-b"}
    assert pd.api.types.is_integer_dtype(first["id"])


def test_stratified_sample_redistributes_unfillable_rare_class_quota() -> None:
    """A class with one product cannot receive two slots when another class can fill them."""
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "articleType": ["rare", "common", "common", "common"],
        }
    )

    sampled = stratified_sample(frame, "articleType", limit=4, seed=2753)

    assert sampled["id"].tolist() == [1, 2, 3, 4]
    assert sampled["articleType"].value_counts().to_dict() == {
        "common": 3,
        "rare": 1,
    }


def test_stratified_sample_redistributes_general_capped_class_allocations() -> None:
    """Every requested slot must move to a class with remaining products, not be lost."""
    frame = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "articleType": ["a"] + ["b"] * 2 + ["c"] * 7,
        }
    )

    sampled = stratified_sample(frame, "articleType", limit=9, seed=2753)

    assert len(sampled) == 9
    assert sampled["articleType"].value_counts().to_dict() == {
        "a": 1,
        "b": 2,
        "c": 6,
    }


@pytest.mark.parametrize("limit", [-1, 1.5])
def test_stratified_sample_rejects_invalid_limits(limit: int | float) -> None:
    """A bad sample limit must fail clearly instead of producing a misleading subset."""
    frame = pd.DataFrame({"id": [1], "articleType": ["tops"]})

    with pytest.raises(ValueError, match="limit"):
        stratified_sample(frame, "articleType", limit=limit, seed=2753)


def test_stratified_sample_handles_empty_frame_and_missing_column() -> None:
    """Empty EDA populations must stay empty while misspelled strata are actionable."""
    empty = pd.DataFrame(
        {"id": pd.Series(dtype="int64"), "articleType": pd.Series(dtype="string")}
    )

    sampled = stratified_sample(empty, "articleType", limit=5, seed=2753)

    assert sampled.empty
    with pytest.raises(ValueError, match="Column not found"):
        stratified_sample(empty, "missing", limit=5, seed=2753)


def test_measure_image_reports_real_rgb_grayscale_flat_and_edge_metrics(
    tmp_path: Path,
) -> None:
    """A wrong conversion or metric calculation would conceal image quality differences."""
    rgb_path = tmp_path / "rgb.png"
    grayscale_path = tmp_path / "gray.png"
    flat_path = tmp_path / "flat.png"
    edge_path = tmp_path / "edge.png"
    _save_rgb(rgb_path, (30, 120, 220))
    Image.new("L", (12, 8), 90).save(grayscale_path)
    _save_rgb(flat_path, (128, 128, 128))
    _save_edge_heavy(edge_path)

    rgb = measure_image(rgb_path)
    grayscale = measure_image(grayscale_path)
    flat = measure_image(flat_path)
    edge = measure_image(edge_path)

    assert rgb["path"] == str(rgb_path)
    assert (rgb["width"], rgb["height"]) == (12, 8)
    assert rgb["aspect_ratio"] == pytest.approx(1.5)
    assert rgb["mode"] == "RGB"
    assert len(rgb["file_sha256"]) == len(rgb["pixel_sha256"]) == 64
    assert rgb["file_bytes"] > 0
    assert len(rgb["dhash"]) == 16
    assert grayscale["mode"] == "L"
    for record in (rgb, grayscale, flat, edge):
        assert record["error"] is None
        assert all(
            np.isfinite(record[name])
            for name in (
                "brightness",
                "contrast",
                "colorfulness",
                "saturation",
                "edge_sharpness",
            )
        )
    assert flat["contrast"] == pytest.approx(0.0)
    assert grayscale["colorfulness"] == pytest.approx(0.0)
    assert edge["edge_sharpness"] > flat["edge_sharpness"]


def test_measure_image_records_corrupt_file_instead_of_crashing(tmp_path: Path) -> None:
    """A corrupt source image must remain auditable rather than aborting the analysis."""
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")

    record = measure_image(corrupt)

    assert record["path"] == str(corrupt)
    assert record["file_bytes"] == len(b"not an image")
    assert len(record["file_sha256"]) == 64
    assert record["error"]
    assert record["pixel_sha256"] is None
    assert record["dhash"] is None


def test_measure_images_preserves_integer_product_ids_and_path_order(tmp_path: Path) -> None:
    """String IDs or input-order output would make later product joins unreliable."""
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    _save_rgb(one, (20, 30, 40))
    _save_rgb(two, (50, 60, 70))
    frame = pd.DataFrame({"id": [20, 10], "image_path": [two, one]})

    measured = measure_images(frame, "image_path")

    assert measured["id"].tolist() == [10, 20]
    assert pd.api.types.is_integer_dtype(measured["id"])
    assert measured["path"].tolist() == [str(one), str(two)]


def test_duplicate_candidates_distinguish_exact_near_and_unrelated_images(
    tmp_path: Path,
) -> None:
    """Byte-identical files are exact duplicates and must not be repeated as near pairs."""
    exact_left = tmp_path / "exact-left.png"
    exact_right = tmp_path / "exact-right.png"
    similar = tmp_path / "similar.png"
    unrelated = tmp_path / "unrelated.png"
    _save_edge_heavy(exact_left)
    shutil.copyfile(exact_left, exact_right)
    similar_pixels = np.indices((8, 12)).sum(axis=0) % 2 * 255
    similar_pixels[0, 0] = 127
    Image.fromarray(similar_pixels.astype(np.uint8), mode="L").save(similar)
    _save_rgb(unrelated, (200, 20, 10))
    frame = pd.DataFrame(
        {
            "id": [40, 10, 30, 20],
            "image_path": [unrelated, exact_right, similar, exact_left],
        }
    )

    measurements = measure_images(frame, "image_path")
    exact = exact_duplicate_groups(measurements)
    near = near_duplicate_candidates(measurements, max_distance=6)

    assert exact["ids"].tolist() == [(10, 20)]
    assert exact.loc[0, "pixel_sha256"] == measurements.loc[
        measurements["id"].eq(10), "pixel_sha256"
    ].item()
    assert [10, 20] not in near[["id_left", "id_right"]].values.tolist()
    assert near.equals(
        near.sort_values(["distance", "id_left", "id_right"], ignore_index=True)
    )
    assert {10, 30}.issubset(set(near[["id_left", "id_right"]].iloc[0]))


def test_pixel_identical_byte_different_images_are_near_not_exact(tmp_path: Path) -> None:
    """Encoding changes must not be called exact content duplicates."""
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    image = Image.new("RGB", (12, 8), (40, 90, 200))
    image.save(left, compress_level=0)
    image.save(right, compress_level=9)
    measurements = measure_images(
        pd.DataFrame({"id": [2, 1], "image_path": [right, left]}), "image_path"
    )

    assert left.read_bytes() != right.read_bytes()
    assert measurements["pixel_sha256"].nunique() == 1
    assert exact_duplicate_groups(measurements).empty
    assert near_duplicate_candidates(measurements, max_distance=0)[
        ["id_left", "id_right"]
    ].values.tolist() == [[1, 2]]


def test_duplicate_helpers_ignore_measurement_errors_and_validate_limits(tmp_path: Path) -> None:
    """Unreadable files and impossible thresholds must not create false duplicate claims."""
    image = tmp_path / "image.png"
    corrupt = tmp_path / "corrupt.jpg"
    _save_rgb(image, (1, 2, 3))
    corrupt.write_bytes(b"corrupt")
    measurements = measure_images(
        pd.DataFrame({"id": [2, 1], "image_path": [corrupt, image]}), "image_path"
    )

    assert exact_duplicate_groups(measurements).empty
    assert near_duplicate_candidates(measurements).empty
    with pytest.raises(ValueError, match="max_distance"):
        near_duplicate_candidates(measurements, max_distance=-1)
    with pytest.raises(ValueError, match="max_distance"):
        near_duplicate_candidates(measurements, max_distance=65)


def test_duplicate_helpers_return_empty_groups_for_empty_measurements() -> None:
    """An empty selected population must produce typed empty duplicate summaries."""
    measurements = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "error": pd.Series(dtype="string"),
            "file_sha256": pd.Series(dtype="string"),
            "pixel_sha256": pd.Series(dtype="string"),
            "dhash": pd.Series(dtype="string"),
        }
    )

    assert exact_duplicate_groups(measurements).empty
    assert near_duplicate_candidates(measurements).empty


def test_paired_image_comparison_joins_each_product_once_and_calculates_deltas(
    tmp_path: Path,
) -> None:
    """An outer join or repeated ID would double-count products in paired quality reports."""
    dark = tmp_path / "dark.png"
    bright = tmp_path / "bright.png"
    _save_rgb(dark, (20, 20, 20))
    _save_rgb(bright, (200, 200, 200))
    low = measure_images(pd.DataFrame({"id": [2, 1], "path": [dark, dark]}), "path")
    high = measure_images(
        pd.DataFrame({"id": [1, 3], "path": [bright, bright]}), "path"
    )

    paired = paired_image_comparison(low, high)

    assert paired["id"].tolist() == [1]
    assert paired.loc[0, "brightness_delta"] > 0
    assert paired.loc[0, "brightness_ratio"] > 1
    assert paired.loc[0, "low_path"] == str(dark)
    assert paired.loc[0, "high_path"] == str(bright)
    assert paired.loc[0, "low_error"] is None
    assert paired.loc[0, "high_error"] is None
    assert paired.loc[0, "valid_pair"]


def test_paired_image_comparison_keeps_corrupt_pair_errors(tmp_path: Path) -> None:
    """A broken view must be visible and invalid rather than producing usable-looking deltas."""
    good = tmp_path / "good.png"
    corrupt = tmp_path / "corrupt.jpg"
    _save_rgb(good, (40, 50, 60))
    corrupt.write_bytes(b"not an image")
    low = measure_images(pd.DataFrame({"id": [1], "path": [corrupt]}), "path")
    high = measure_images(pd.DataFrame({"id": [1], "path": [good]}), "path")

    paired = paired_image_comparison(low, high)

    assert paired.loc[0, "low_error"]
    assert paired.loc[0, "high_error"] is None
    assert not paired.loc[0, "valid_pair"]
    assert np.isnan(paired.loc[0, "brightness_delta"])


def test_paired_image_comparison_handles_empty_and_rejects_malformed_frames() -> None:
    """Empty valid schemas are safe, but missing error evidence is not a usable pair input."""
    columns = ["id", "path", "error", "brightness", "contrast", "colorfulness", "saturation", "edge_sharpness"]
    empty = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})

    assert paired_image_comparison(empty, empty).empty
    with pytest.raises(ValueError, match="error"):
        paired_image_comparison(empty.drop(columns="error"), empty)

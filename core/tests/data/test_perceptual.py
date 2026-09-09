from __future__ import annotations

import inspect
import json

import numpy as np
from PIL import Image

from fashion.data.hashing import compute_sha256
from fashion.data.perceptual import (
    compute_image_hashes,
    compute_pair_pixel_metrics,
    find_hamming_candidates_multi_index,
    hamming_distance,
    run_perceptual_audit,
)


def _pattern(path, vertical: bool) -> None:
    array = np.full((128, 128, 3), 255, dtype=np.uint8)
    if vertical:
        array[:, 50:78] = 0
    else:
        array[50:78, :] = 0
    Image.fromarray(array).save(path, "JPEG", quality=95)


def test_hashes_are_deterministic(tmp_path):
    path = tmp_path / "image.jpg"
    _pattern(path, vertical=True)
    assert compute_image_hashes(path) == compute_image_hashes(path)


def test_recompression_changes_sha_but_keeps_perceptual_hash_close(tmp_path):
    original = tmp_path / "original.jpg"
    recompressed = tmp_path / "recompressed.jpg"
    _pattern(original, vertical=True)
    with Image.open(original) as image:
        image.save(recompressed, "JPEG", quality=75)
    original_hashes = compute_image_hashes(original)
    recompressed_hashes = compute_image_hashes(recompressed)
    assert compute_sha256(original) != compute_sha256(recompressed)
    assert hamming_distance(original_hashes[0], recompressed_hashes[0]) <= 2


def test_distinct_patterns_have_different_hashes(tmp_path):
    vertical = tmp_path / "vertical.jpg"
    horizontal = tmp_path / "horizontal.jpg"
    _pattern(vertical, vertical=True)
    _pattern(horizontal, vertical=False)
    distance = hamming_distance(
        compute_image_hashes(vertical)[0], compute_image_hashes(horizontal)[0]
    )
    assert distance > 8


def test_multi_index_search_has_no_close_pair_false_negatives():
    base = 0x123456789ABCDEF0
    hashes = [
        base,
        base ^ (1 << 2),
        base ^ (1 << 10) ^ (1 << 20),
        base ^ (1 << 5) ^ (1 << 25) ^ (1 << 45),
        base ^ 0xFFFFFFFFFFFFFFFF,
    ]
    candidates = find_hamming_candidates_multi_index(hashes, max_distance=3)
    assert (0, 1) in candidates[1]
    assert (0, 2) in candidates[2]
    assert (0, 3) in candidates[3]
    assert all(4 not in pair for pairs in candidates.values() for pair in pairs)


def test_pixel_metrics_are_zero_for_same_image(tmp_path):
    path = tmp_path / "image.jpg"
    _pattern(path, vertical=True)
    metrics = compute_pair_pixel_metrics(path, path)
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["max_difference"] == 0.0
    assert metrics["crop_mse"] == 0.0
    assert metrics["crop_mae"] == 0.0
    assert metrics["foreground_ratio"] == 1.0


def test_perceptual_audit_records_objective_metrics_before_splitting(prepared_project):
    candidates = np.genfromtxt(
        prepared_project.audit / "near_duplicate_candidates.csv.gz",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert candidates.size > 0
    for field in ("mse", "mae", "crop_mse", "crop_mae", "foreground_ratio"):
        assert field in candidates.dtype.names
    assert "partition_1" not in candidates.dtype.names
    assert "review_notes" not in candidates.dtype.names


def test_perceptual_audit_has_no_human_gate(prepared_project):
    parameters = inspect.signature(run_perceptual_audit).parameters
    assert not {"cross_role_review_csv", "policy_review_csv"}.intersection(parameters)

    summary = json.loads(
        (prepared_project.audit / "near_duplicate_summary.json").read_text(encoding="utf-8")
    )
    assert "cross_role_review" not in summary
    assert "automatic_rule_review" not in summary
    assert "cross_role_automatic_matches" in summary

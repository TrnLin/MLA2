"""Image-only checks for every protected role and external family allocation."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.data.external_usage import assign_external_partitions, prepare_external_image
from fashion.data.external_usage_audit import (
    find_teacher_overlaps,
    fingerprint_image,
    link_external_families,
)
from fashion.data.perceptual import compute_image_hashes


def save_pattern(path, seed=17, size=(45, 90)):
    pixels = np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return fingerprint_image(path)


def candidate(path, fingerprint, name="external", **changes):
    return {
        **fingerprint,
        "external_id": name,
        "original_path": path.name,
        "source": "fixture",
        "source_id": name,
        "source_group_id": name,
        "source_title": name,
        "source_label": "Home",
        "image_url": f"https://example.org/{name}.png",
        **changes,
    }


def test_fingerprint_matches_canonical_perceptual_arithmetic(tmp_path):
    path = tmp_path / "original.png"
    result = save_pattern(path)
    dhash, ahash = compute_image_hashes(path)
    assert result["dhash_hex"] == f"{dhash:016x}"
    assert result["ahash_hex"] == f"{ahash:016x}"
    prepared = tmp_path / "view.png"
    prepare_external_image(path, prepared)
    view_dhash, view_ahash = compute_image_hashes(prepared)
    assert result["view_dhash_hex"] == f"{view_dhash:016x}"
    assert result["view_ahash_hex"] == f"{view_ahash:016x}"
    assert result["view_pixel_sha256"] == fingerprint_image(prepared)["pixel_sha256"]


@pytest.mark.parametrize("role", ["development", "holdout", "quarantine", "prediction"])
def test_reencoded_overlap_is_found_for_every_teacher_role(tmp_path, role):
    original = tmp_path / "original.png"
    fp = save_pattern(original, size=(80, 60))
    teacher = tmp_path / "teacher.bmp"
    with Image.open(original) as image:
        image.save(teacher)
    reference_fp = fingerprint_image(teacher)
    assert fp["file_sha256"] != reference_fp["file_sha256"]
    references = pd.DataFrame(
        [{**reference_fp, "id": 37, "path": teacher.name, "audit_role": role}]
    )  # No target columns are provided, including for holdout.
    overlaps = find_teacher_overlaps(
        pd.DataFrame([candidate(original, fp)]), references, root=tmp_path, cache=tmp_path / "cache"
    )
    assert overlaps.accepted_overlap.any()
    assert set(overlaps.teacher_role) == {role}
    assert overlaps.identity_evidence.str.contains("pixel_sha256=pixel_sha256").any()
    assert set(zip(overlaps.external_view, overlaps.teacher_view)) >= {
        ("original", "original"),
        ("original", "view_"),
        ("view_", "original"),
        ("view_", "view_"),
    }


def test_original_to_normalized_teacher_identity_is_found(tmp_path):
    original = tmp_path / "original.png"
    fp = save_pattern(original)
    teacher = tmp_path / "teacher.png"
    prepare_external_image(original, teacher)
    references = pd.DataFrame(
        [{**fingerprint_image(teacher), "id": 19, "path": teacher.name, "audit_role": "prediction"}]
    )
    overlaps = find_teacher_overlaps(
        pd.DataFrame([candidate(original, fp)]), references, root=tmp_path, cache=tmp_path / "cache"
    )
    assert overlaps.accepted_overlap.any()
    assert overlaps.identity_evidence.str.contains("view_pixel_sha256=pixel_sha256").any()


def test_family_links_are_transitive_stable_and_never_split_views(tmp_path):
    first = tmp_path / "first.png"
    fp = save_pattern(first)
    other = tmp_path / "other.png"
    other_fp = save_pattern(other, seed=88)
    final = tmp_path / "final.png"
    prepare_external_image(first, final)
    rows = [
        candidate(first, fp, "a", source_group_id="family"),
        candidate(other, other_fp, "b", source_group_id="family"),
        candidate(final, fingerprint_image(final), "c"),
    ]
    frame = pd.DataFrame(rows)
    linked, edges = link_external_families(frame, root=tmp_path)
    reordered, _ = link_external_families(frame.iloc[::-1], root=tmp_path)
    pd.testing.assert_series_equal(
        linked.set_index("external_id").external_group_id.sort_index(),
        reordered.set_index("external_id").external_group_id.sort_index(),
    )
    assert linked.external_group_id.nunique() == 1
    assert "source_group" in set(edges.reason)
    split = assign_external_partitions(linked)
    assert split.external_partition.nunique() == 1


def test_final_view_rule_is_checked_after_original_pixel_rejection(tmp_path):
    rows = []
    for name, color in [("red", (255, 0, 0)), ("blue", (0, 0, 255))]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (60, 80), color).save(path)
        rows.append(candidate(path, fingerprint_image(path), name))
    linked, edges = link_external_families(pd.DataFrame(rows), root=tmp_path)
    assert linked.external_group_id.nunique() == 1
    assert "conservative_final_view_hash_group" in set(edges.reason)


def test_white_margins_do_not_hide_same_product_from_teacher_audit(tmp_path):
    # Keep product pixels below the white threshold, with varied edges and texture.
    product = np.random.default_rng(29).integers(10, 220, (64, 48, 3), dtype=np.uint8)
    teacher = tmp_path / "tight_teacher.png"
    Image.fromarray(product).save(teacher)
    canvas = Image.new("RGB", (360, 480), "white")
    canvas.paste(Image.fromarray(product), (151, 173))
    original = tmp_path / "padded_external.png"
    canvas.save(original)
    external_fp = fingerprint_image(original)
    teacher_fp = fingerprint_image(teacher)
    assert external_fp["pixel_sha256"] != teacher_fp["pixel_sha256"]
    assert external_fp["view_pixel_sha256"] != teacher_fp["view_pixel_sha256"]
    assert (int(external_fp["dhash_hex"], 16) ^ int(teacher_fp["dhash_hex"], 16)).bit_count() > 2
    references = pd.DataFrame(
        [{**teacher_fp, "id": 81, "path": teacher.name, "audit_role": "holdout"}]
    )
    matches = find_teacher_overlaps(
        pd.DataFrame([candidate(original, external_fp)]),
        references,
        root=tmp_path,
        cache=tmp_path / "cache",
    )
    assert matches.accepted_overlap.any()
    assert set(matches.teacher_role) == {"holdout"}

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from fashion.retrieval.external import (
    build_external_variant_index,
    ensure_external_image_audit,
    read_external_catalogue,
    reconcile_external_ids,
    select_development_pairs,
)


def _write_jpeg(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (12, 16), colour).save(path, format="JPEG")


def test_reconciliation_proves_variant_union_and_records_missing_file(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_jpeg(image_dir / "1.jpg", (255, 0, 0))
    _write_jpeg(image_dir / "3.jpg", (0, 0, 255))
    catalogue_csv = tmp_path / "images.csv"
    pd.DataFrame(
        {
            "filename": ["1.jpg", "2.jpg", "3.jpg"],
            "link": ["https://example/1", "https://example/2", "https://example/3"],
        }
    ).to_csv(catalogue_csv, index=False)
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(train_csv, index=False)
    pd.DataFrame({"id": [3]}).to_csv(test_csv, index=False)

    summary, missing = reconcile_external_ids(
        catalogue_csv,
        image_dir,
        train_csv,
        test_csv,
        root=tmp_path,
    )

    assert summary["catalogue_equals_teacher_union"] is True
    assert summary["catalogue_train_overlap"] == 2
    assert summary["catalogue_test_overlap"] == 1
    assert summary["catalogue_ids_without_image"] == [2]
    assert missing.to_dict("records") == [
        {"id": 2, "teacher_role": "train", "issue": "catalogue_id_without_image"}
    ]


def test_catalogue_rejects_labels_or_other_columns(tmp_path: Path) -> None:
    path = tmp_path / "images.csv"
    pd.DataFrame(
        {"filename": ["1.jpg"], "link": ["https://example/1"], "articleType": ["Tshirts"]}
    ).to_csv(path, index=False)

    try:
        read_external_catalogue(path)
    except ValueError as error:
        assert "columns must be" in str(error)
    else:
        raise AssertionError("label-bearing external catalogues must be rejected")


def test_external_index_inherits_canonical_split_and_carries_no_targets() -> None:
    splits = pd.DataFrame(
        {
            "id": [1, 2],
            "path": ["teacher/1.jpg", "teacher/2.jpg"],
            "width": [60, 60],
            "height": [80, 80],
            "file_size_bytes": [100, 110],
            "partition": ["development", "holdout"],
            "cv_fold": [0, pd.NA],
            "product_family_group": ["family_1", "family_2"],
            "duplicate_group": ["single_1", "single_2"],
            "articleType": ["Tshirts", ""],
        }
    )
    audit = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "path": ["external/1.jpg", "external/2.jpg", "external/3.jpg"],
            "width": [1080, 1080, 1080],
            "height": [1440, 1440, 1440],
            "aspect_ratio": [0.75, 0.75, 0.75],
            "mode": ["RGB", "RGB", "RGB"],
            "format": ["JPEG", "JPEG", "JPEG"],
            "file_size_bytes": [1000, 1100, 1200],
            "sha256": ["a", "b", "c"],
            "decode_ok": [True, True, True],
        }
    )

    index = build_external_variant_index(splits, audit)

    assert index["id"].tolist() == [1, 2]
    assert index["partition"].tolist() == ["development", "holdout"]
    assert "articleType" not in index
    assert select_development_pairs(index, sample_size=10)["id"].tolist() == [1]


def test_decode_audit_cache_tracks_the_external_inventory(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_jpeg(image_dir / "1.jpg", (255, 0, 0))
    audit_csv = tmp_path / "processed" / "external_image_audit.csv.gz"
    cache_json = tmp_path / "processed" / "external_audit_cache.json"

    first, first_cached = ensure_external_image_audit(
        image_dir,
        audit_csv,
        cache_json,
        root=tmp_path,
        workers=1,
    )
    second, second_cached = ensure_external_image_audit(
        image_dir,
        audit_csv,
        cache_json,
        root=tmp_path,
        workers=1,
    )

    assert first_cached is False
    assert second_cached is True
    assert first["decode_ok"].tolist() == [True]
    assert second["decode_ok"].tolist() == [True]

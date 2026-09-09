from __future__ import annotations

import pandas as pd
import pytest

from fashion.task4.protocol import compute_relevance_coverage
from fashion.task4_evaluation.views import (
    DEVELOPMENT_ROWS_EXPECTED,
    HOLDOUT_ROWS_EXPECTED,
    QUARANTINE_ROWS_EXPECTED,
    build_holdout_views,
)


def _row(
    product_id: int,
    partition: str,
    *,
    cv_fold: int | str,
    family: str,
    article_type: str = "Tshirts",
    colour: str = "Blue",
) -> dict[str, object]:
    return {
        "id": product_id,
        "gender": "Men",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": article_type,
        "baseColour": colour,
        "season": "Summer",
        "usage": "Casual",
        "year": 2011,
        "productDisplayName": f"item {product_id}",
        "product_name_repaired": f"item {product_id}",
        "has_articleType_label": bool(article_type),
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
        "path": f"data/raw/teacher/train/images/{product_id}.jpg",
        "width": 60,
        "height": 80,
        "aspect_ratio": 0.75,
        "mode": "RGB",
        "format": "JPEG",
        "file_size_bytes": 2048,
        "sha256": f"{product_id:064x}",
        "product_name_key": f"key-{product_id}",
        "product_family_group": family,
        "family_group_basis": "product_name_key",
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "pre_quarantine_reason": "",
        "duplicate_group": f"dup-{product_id}",
        "is_cross_role_duplicate": False,
        "quarantine_reason": "approved test quarantine" if partition == "quarantine" else "",
        "partition": partition,
        "cv_fold": cv_fold,
    }


def _split_frame() -> pd.DataFrame:
    rows = [
        _row(1, "development", cv_fold=0, family="dev-a"),
        _row(2, "development", cv_fold=1, family="dev-b"),
        _row(3, "development", cv_fold=2, family="dev-c", colour="Red"),
        _row(4, "development", cv_fold=3, family="dev-d", article_type="Shirts"),
        _row(5, "holdout", cv_fold="", family="hold-a"),
        _row(6, "holdout", cv_fold="", family="hold-a", colour="Red"),
        _row(7, "holdout", cv_fold="", family="hold-b", article_type="Shirts"),
        _row(8, "quarantine", cv_fold="", family="quar-a"),
    ]
    return pd.DataFrame(rows)


def test_expected_partition_counts_match_the_canonical_split() -> None:
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False, low_memory=False)
    counts = splits["partition"].value_counts()
    assert counts["development"] == DEVELOPMENT_ROWS_EXPECTED
    assert counts["holdout"] == HOLDOUT_ROWS_EXPECTED
    assert counts["quarantine"] == QUARANTINE_ROWS_EXPECTED


def test_primary_view_is_holdout_queries_against_all_development() -> None:
    primary, _ = build_holdout_views(_split_frame())
    assert sorted(primary.queries["id"]) == [5, 6, 7]
    assert sorted(primary.gallery["id"]) == [1, 2, 3, 4]


def test_family_view_is_holdout_against_itself() -> None:
    _, family = build_holdout_views(_split_frame())
    assert sorted(family.queries["id"]) == [5, 6, 7]
    assert sorted(family.gallery["id"]) == [5, 6, 7]


def test_quarantine_rows_never_enter_either_view() -> None:
    primary, family = build_holdout_views(_split_frame())
    for frame in (primary.queries, primary.gallery, family.queries, family.gallery):
        assert 8 not in set(frame["id"])


def test_isolation_failure_names_the_crossing_family() -> None:
    frame = _split_frame()
    frame.loc[frame["id"].eq(5), "product_family_group"] = "dev-a"
    with pytest.raises(ValueError, match="product family crosses development and holdout"):
        build_holdout_views(frame)


def test_duplicate_ids_fail_structural_validation() -> None:
    frame = pd.concat([_split_frame(), _split_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="non-null and unique"):
        build_holdout_views(frame)


def test_unlocked_holdout_labels_are_preserved() -> None:
    primary, family = build_holdout_views(_split_frame())
    expected = ["Tshirts", "Tshirts", "Shirts"]
    assert primary.queries["articleType"].tolist() == expected
    assert family.queries["articleType"].tolist() == expected


def test_coverage_runs_on_the_holdout_views() -> None:
    primary, family = build_holdout_views(_split_frame())
    coverage = compute_relevance_coverage(primary, family, k_values=(10,))
    assert sorted(coverage["protocol"].unique()) == ["family", "primary"]
    assert coverage["total_queries"].eq(3).all()

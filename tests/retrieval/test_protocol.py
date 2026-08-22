from __future__ import annotations

import math

import pandas as pd
import pytest

from fashion.data.dataset import load_splits
from fashion.retrieval.protocol import (
    build_development_retrieval_sets,
    build_development_retrieval_variant_sets,
    build_final_application_gallery,
    rank_products_from_variants,
    recall_at_k,
    remove_self_match,
)


def test_development_gallery_is_train_only_and_family_isolated(prepared_project):
    splits = load_splits(prepared_project.splits)
    queries, gallery = build_development_retrieval_sets(splits)
    validation = splits[splits["partition"].eq("val")]
    supported = validation["has_articleType_supported_label"].astype(bool)
    assert queries["partition"].eq("val").all()
    assert len(queries) == int(
        (
            splits["partition"].eq("val") & splits["has_articleType_supported_label"].astype(bool)
        ).sum()
    )
    assert gallery["partition"].eq("train").all()
    assert len(gallery) == int(splits["partition"].eq("train").sum())
    assert set(queries["id"]).isdisjoint(gallery["id"])
    assert set(queries["sha256"]).isdisjoint(gallery["sha256"])
    assert set(queries["product_family_group"]).isdisjoint(gallery["product_family_group"])
    assert len(validation) == len(queries) + int((~supported).sum())
    assert set(validation.loc[~supported, "id"]).isdisjoint(queries["id"])


def test_self_match_is_removed_before_top_k():
    assert remove_self_match(7, [7, 2, 2, 7, 3, 4], top_k=3) == [2, 3, 4]


def test_recall_at_k_uses_grade_two_and_reports_zero_positive_queries():
    assert math.isnan(recall_at_k([1, 0, 1], k=2, total_relevant=0))
    assert recall_at_k([2], k=5, total_relevant=1) == 1.0
    assert recall_at_k([0, 2, 1, 2], k=3, total_relevant=2) == 0.5
    assert recall_at_k([2, 0, 2], k=5, complete_gallery_grades=[2, 0, 2]) == 1.0


def test_recall_at_k_counts_relevant_products_outside_top_k():
    assert recall_at_k([2, 0], k=2, total_relevant=3) == pytest.approx(1 / 3)
    assert recall_at_k(
        [2, 0],
        k=2,
        complete_gallery_grades=[2, 0, 2, 1, 2],
    ) == pytest.approx(1 / 3)


def test_recall_at_k_rejects_a_top_k_only_denominator():
    with pytest.raises(ValueError, match="ranked Top-K grades cannot define"):
        recall_at_k([2, 0, 1], k=3)
    with pytest.raises(ValueError, match="exceed"):
        recall_at_k([2, 2], k=2, total_relevant=1)


def test_paired_variants_rank_each_product_once_even_when_they_disagree():
    scores = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "variant": ["original", "high_resolution"] * 3,
            "score": [0.95, 0.10, 0.80, 0.79, 0.40, 0.85],
        }
    )
    ranked = rank_products_from_variants(scores, top_k=3)
    assert ranked["id"].tolist() == [1, 3, 2]
    assert ranked["variant"].tolist() == ["original", "high_resolution", "original"]
    assert ranked["id"].is_unique


def test_retrieval_variant_sets_keep_both_images_but_one_product_identity(prepared_project):
    splits = load_splits(prepared_project.splits)
    products = splits[splits["partition"].isin({"train", "val", "holdout"})][
        ["id", "partition", "path"]
    ]
    variants = pd.concat(
        [
            products.assign(
                variant="original",
                variant_key=products["id"].astype(str) + ":original",
            ),
            products.assign(
                variant="high_resolution",
                variant_key=products["id"].astype(str) + ":high_resolution",
                path="high/" + products["id"].astype(str) + ".jpg",
            ),
        ],
        ignore_index=True,
    )
    queries, gallery = build_development_retrieval_variant_sets(splits, variants)
    assert queries["partition"].eq("val").all()
    assert gallery["partition"].eq("train").all()
    assert queries.groupby("id")["variant"].agg(set).eq({"original", "high_resolution"}).all()
    assert gallery.groupby("id")["variant"].agg(set).eq({"original", "high_resolution"}).all()
    assert not set(queries["id"]).intersection(gallery["id"])


def test_final_gallery_stays_locked_until_evaluation(prepared_project):
    splits = load_splits(prepared_project.splits)
    with pytest.raises(ValueError, match="locked"):
        build_final_application_gallery(splits, evaluation_complete=False)
    final = build_final_application_gallery(splits, evaluation_complete=True)
    assert set(final["partition"]) <= {"train", "val", "holdout"}

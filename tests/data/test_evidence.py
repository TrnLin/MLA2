from __future__ import annotations

import pandas as pd

from fashion.data.evidence import (
    article_target_heatmap,
    family_profile,
    fold_support_tables,
    near_threshold_review,
    shortcut_benchmarks,
)


def test_shortcut_benchmark_is_calculated_from_rows():
    frame = pd.DataFrame(
        {
            "articleType": ["A", "A", "B", "B"],
            "season": ["S", "S", "W", "W"],
            "usage": ["C", "C", "N", "N"],
            "gender": ["F", "F", "M", "M"],
            "has_articleType_label": [True] * 4,
            "has_season_label": [True] * 4,
            "has_usage_label": [True] * 4,
            "has_gender_label": [True] * 4,
        }
    )
    result = shortcut_benchmarks(frame).set_index("predicted_target")
    assert result.loc["season", "global_majority_accuracy"] == 0.5
    assert result.loc["season", "group_majority_accuracy"] == 1.0
    assert "not model accuracy" in result.loc["season", "interpretation"]
    heatmap = article_target_heatmap(frame, "season", top_article_types=2)
    assert heatmap.loc["A", "S"] == 1.0
    assert heatmap.loc["B", "W"] == 1.0


def test_family_profile_and_fold_support_report_zero_crossings():
    splits = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "partition": ["development"] * 3,
            "cv_fold": [0, 0, 1],
            "product_family_group": ["f1", "f1", "f2"],
            "family_group_basis": ["normalized_product_name"] * 2 + ["singleton"],
        }
    )
    sizes, profile, sources = family_profile(splits)
    assert sizes["family_size"].tolist() == [2, 1]
    assert profile.loc[0, "conservative_split_groups"] == 2
    assert profile.loc[0, "development_fold_crossings"] == 0
    assert (
        sources.set_index("family_source").loc[
            "same normalized product name", "multirow_families"
        ]
        == 1
    )

    summary = pd.DataFrame(
        {
            "target": ["articleType"],
            "class": ["Rare"],
            "development_product_count": [1],
            "development_family_count": [1],
            "rare_warning": ["untrainable_in_one_or_more_folds"],
            "untrainable_fold_count": [1],
            **{f"fold_{fold}_validation_products": [int(fold == 2)] for fold in range(5)},
            **{f"fold_{fold}_training_products": [int(fold != 2)] for fold in range(5)},
        }
    )
    _, validation, untrainable = fold_support_tables(summary)
    assert validation.loc["articleType: Rare", "fold 2"] == 1
    assert bool(untrainable.loc["articleType: Rare", "fold 2"])


def test_near_threshold_review_selects_both_sides_without_changing_decisions():
    candidates = pd.DataFrame(
        {
            "id_1": [1, 3, 5, 7],
            "id_2": [2, 4, 6, 8],
            "is_exact_sha256": [False] * 4,
            "meets_automatic_rule": [True, True, False, False],
            "accepted_near_duplicate": [True, True, False, False],
            "dhash_distance": [2, 0, 2, 8],
            "ahash_distance": [1, 0, 1, 8],
            "mse": [0.00049, 0.0001, 0.00051, 0.1],
            "mae": [0.0099, 0.001, 0.0099, 0.2],
            "crop_mse": [0.0049, 0.001, 0.0049, 0.1],
            "crop_mae": [0.049, 0.01, 0.049, 0.2],
            "foreground_ratio": [0.9, 1.0, 0.9, 0.1],
        }
    )
    review = near_threshold_review(candidates, accepted_pairs=1, rejected_pairs=1)

    assert set(review["review_side"]) == {"accepted side", "rejected side"}
    assert review["pipeline_effect"].eq("none; evidence-only sample").all()
    assert review.loc[review["review_side"].eq("accepted side"), "id_1"].item() == 1
    assert review.loc[review["review_side"].eq("rejected side"), "id_1"].item() == 5

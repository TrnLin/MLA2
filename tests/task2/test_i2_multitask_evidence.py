from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion.data.dataset import load_splits
from fashion.task2.multitask_evidence import (
    I2_REFERENCE_EXPERIMENT_ID,
    apply_i2_selection_rule,
    build_article_type_shortcut_audit,
    fit_article_type_majorities,
    plot_i2_learning_curves,
)


def test_article_type_majority_fit_is_deterministic_and_ignores_masked_rows() -> None:
    training = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "articleType": ["A", "A", "A", "B", "B"],
            "season": ["Spring", "Fall", "Winter", "Summer", "Summer"],
            "has_articleType_label": [True, True, False, True, True],
        }
    )

    mapping = fit_article_type_majorities(training).set_index("articleType")

    assert mapping.loc["A", "shortcut_majority_season"] == "Fall"
    assert int(mapping.loc["A", "majority_count"]) == 1
    assert int(mapping.loc["A", "training_labeled_count"]) == 2
    assert mapping.loc["B", "shortcut_majority_season"] == "Summer"
    assert float(mapping.loc["B", "majority_share"]) == 1.0


def test_repository_shortcut_audit_covers_only_valid_development_season_rows() -> None:
    splits = load_splits()
    mappings, assignments, fold_audit = build_article_type_shortcut_audit(splits)
    expected = splits.loc[
        splits["partition"].eq("development") & splits["has_season_label"]
    ]
    protected_ids = set(
        splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"]
    )

    assert len(assignments) == assignments["id"].nunique() == len(expected) == 32_753
    assert not (set(assignments["id"]) & protected_ids)
    assert set(assignments["fold"]) == set(range(5))
    assert {"aligned", "conflict"} <= set(assignments["shortcut_slice"])
    assert len(fold_audit) == 5
    assert mappings["training_id_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert mappings.groupby("fold")["training_id_sha256"].nunique().eq(1).all()
    assert mappings["fold"].nunique() == 5


def _comparison(*, passing: bool) -> pd.DataFrame:
    conflict_delta = 0.012 if passing else 0.005
    return pd.DataFrame(
        [
            {
                "experiment_id": I2_REFERENCE_EXPERIMENT_ID,
                "auxiliary_weight": 0.0,
                "pooled_macro_f1": 0.738,
                "delta_macro_f1_vs_reference": 0.0,
                "delta_conflict_macro_f1_vs_reference": 0.0,
            },
            {
                "experiment_id": "g4-i2-article-type-lambda-0-1-c1",
                "auxiliary_weight": 0.1,
                "pooled_macro_f1": 0.740 if passing else 0.737,
                "delta_macro_f1_vs_reference": 0.002 if passing else -0.001,
                "delta_conflict_macro_f1_vs_reference": conflict_delta,
            },
            {
                "experiment_id": "g4-i2-article-type-lambda-0-3-c1",
                "auxiliary_weight": 0.3,
                "pooled_macro_f1": 0.739 if passing else 0.736,
                "delta_macro_f1_vs_reference": 0.001 if passing else -0.002,
                "delta_conflict_macro_f1_vs_reference": 0.011 if passing else 0.004,
            },
        ]
    )


def test_i2_selection_accepts_conflict_gain_with_overall_floor() -> None:
    decided, selected = apply_i2_selection_rule(_comparison(passing=True))

    assert selected == "g4-i2-article-type-lambda-0-1-c1"
    selected_row = decided.loc[decided["selected_by_i2_gate"]].iloc[0]
    assert bool(selected_row["passes_conflict_gain"])
    assert not bool(selected_row["passes_overall_gain"])


def test_i2_selection_retains_reference_when_neither_lambda_passes() -> None:
    decided, selected = apply_i2_selection_rule(_comparison(passing=False))

    assert selected == I2_REFERENCE_EXPERIMENT_ID
    assert not decided.loc[decided["auxiliary_weight"].gt(0), "passes_i2_gate"].any()


def test_i2_learning_curve_figure_contains_both_declared_lambdas(tmp_path: Path) -> None:
    rows = []
    for experiment_id, weight in (
        ("g4-i2-article-type-lambda-0-1-c1", 0.1),
        ("g4-i2-article-type-lambda-0-3-c1", 0.3),
    ):
        for epoch in range(1, 4):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "auxiliary_weight": weight,
                    "epoch": epoch,
                    "fold_count": 5,
                    "train_total_loss_mean": 2.0 - 0.2 * epoch,
                    "train_total_loss_sd": 0.05,
                    "validation_total_loss_mean": 2.1 - 0.15 * epoch,
                    "validation_total_loss_sd": 0.06,
                    "validation_accuracy_mean": 0.4 + 0.1 * epoch,
                    "validation_accuracy_sd": 0.02,
                    "validation_macro_f1_mean": 0.35 + 0.1 * epoch,
                    "validation_macro_f1_sd": 0.03,
                    "common_five_fold_horizon": 3,
                }
            )
    output = tmp_path / "i2-learning.png"

    result = plot_i2_learning_curves(pd.DataFrame(rows), output)

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 10_000

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task2.final_evaluation import (
    build_b0_prediction_frame,
    build_holdout_slice_assignments,
    build_official_predictions,
    load_final_evaluation_spec,
    score_prediction_frame,
    summarise_grouped_bootstrap,
    validate_prediction_frame,
)

LABELS = ("Fall", "Spring", "Summer", "Winter")


def _prediction_frame() -> pd.DataFrame:
    rows = [
        (10, "Fall", (0.8, 0.1, 0.05, 0.05)),
        (11, "Spring", (0.1, 0.7, 0.1, 0.1)),
        (12, "Summer", (0.1, 0.1, 0.7, 0.1)),
        (13, "Winter", (0.1, 0.1, 0.1, 0.7)),
    ]
    return pd.DataFrame(
        [
            {
                "id": identifier,
                "y_pred": prediction,
                **{f"prob_{label}": probabilities[index] for index, label in enumerate(LABELS)},
            }
            for identifier, prediction, probabilities in rows
        ]
    )


def test_final_evaluation_spec_is_frozen_to_the_selected_i2_bundle() -> None:
    spec = load_final_evaluation_spec()

    assert spec.evaluation_id == "g9-task2-season-final-evaluation"
    assert spec.labels == LABELS
    assert spec.expected_holdout_rows == 5_778
    assert spec.expected_test_rows == 5_829
    assert spec.bootstrap_replicates == 10_000
    assert spec.bootstrap_seed == 2_753
    assert spec.bundle_sha256 == (
        "5927eff73130acedc8015199e1df5a6c6edf64c0b45023ebd91c48d7ed40f93c"
    )
    assert spec.official_output_path == ROOT / "results/season_test_predictions.csv"
    assert tuple(condition.condition for condition in spec.conditions) == (
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    )


def test_prediction_validation_requires_exact_ids_and_probability_argmax() -> None:
    predictions = _prediction_frame()
    audit = validate_prediction_frame(predictions, expected_ids=[10, 11, 12, 13])

    assert audit["row_count"] == 4
    assert audit["unique_id_count"] == 4
    assert audit["labels"] == list(LABELS)

    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly once"):
        validate_prediction_frame(duplicate, expected_ids=[10, 11, 12, 13])

    mismatched = predictions.copy()
    mismatched.loc[0, "y_pred"] = "Spring"
    with pytest.raises(ValueError, match="argmax"):
        validate_prediction_frame(mismatched, expected_ids=[10, 11, 12, 13])


def test_b0_is_fit_on_development_labels_and_reuses_the_prior_probabilities() -> None:
    development = pd.DataFrame(
        {
            "id": range(1, 9),
            "partition": ["development"] * 8,
            "season": ["Summer", "Summer", "Summer", "Fall", "Fall", "Spring", "Winter", "Summer"],
            "has_season_label": [True] * 8,
        }
    )

    predictions, model = build_b0_prediction_frame(development, expected_ids=[21, 22, 23])

    assert model.majority_label == "Summer"
    assert predictions["y_pred"].tolist() == ["Summer"] * 3
    assert predictions.loc[:, [f"prob_{label}" for label in LABELS]].iloc[0].tolist() == [
        0.25,
        0.125,
        0.5,
        0.125,
    ]


def test_scoring_uses_fixed_label_order_and_does_not_change_predictions() -> None:
    predictions = _prediction_frame()
    truth = pd.DataFrame(
        {"id": [10, 11, 12, 13], "season": ["Fall", "Spring", "Winter", "Winter"]}
    )

    metrics, scored = score_prediction_frame(predictions, truth)

    assert metrics["labels"] == list(LABELS)
    assert metrics["n_samples"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert scored["y_pred"].tolist() == predictions["y_pred"].tolist()
    assert scored["y_true"].tolist() == truth["season"].tolist()


def test_holdout_slices_fit_mapping_and_size_boundaries_on_development_only() -> None:
    development = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "articleType": ["Shirts", "Shirts", "Shirts", "Shoes"] * 2,
            "season": ["Summer", "Summer", "Winter", "Winter"] * 2,
            "file_size_bytes": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    holdout = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "articleType": ["Shirts", "Shirts", "Unknown"],
            "season": ["Summer", "Winter", "Fall"],
            "year": [2011, 2014, ""],
            "file_size_bytes": [15, 55, 90],
            "product_family_group": ["family_a", "family_a", "family_b"],
            "mode": ["RGB", "L", "CMYK"],
        }
    )

    assignments, audit = build_holdout_slice_assignments(development, holdout)

    assert assignments["article_type_shortcut"].tolist() == [
        "aligned",
        "conflict",
        "unseen_article_type",
    ]
    assert assignments["product_family_size"].tolist() == ["multirow", "multirow", "singleton"]
    assert assignments["image_mode"].tolist() == ["rgb", "greyscale", "other_mode"]
    assert audit["mapping_fit_rows"] == 8
    assert audit["file_size_fit_rows"] == 8


def test_grouped_bootstrap_is_deterministic_and_reports_i2_minus_b0() -> None:
    true = np.asarray(["Fall", "Spring", "Summer", "Winter"] * 3, dtype=object)
    b0 = np.asarray(["Summer"] * len(true), dtype=object)
    i2 = true.copy()
    groups = np.asarray([f"family_{index // 2}" for index in range(len(true))], dtype=object)

    first = summarise_grouped_bootstrap(
        true,
        groups,
        b0,
        i2,
        replicates=200,
        random_seed=2753,
    )
    second = summarise_grouped_bootstrap(
        true,
        groups,
        b0,
        i2,
        replicates=200,
        random_seed=2753,
    )

    pd.testing.assert_frame_equal(first, second)
    delta = first.loc[first["metric"].eq("i2_minus_b0_macro_f1")].iloc[0]
    assert delta["median"] > 0
    assert delta["lower_95"] > 0


def test_official_predictions_have_only_id_and_season_in_manifest_order() -> None:
    predictions = _prediction_frame().iloc[[2, 0, 3]].copy()
    official = build_official_predictions(predictions, expected_ids=[10, 13, 12])

    assert official.columns.tolist() == ["id", "season"]
    assert official["id"].tolist() == [10, 13, 12]
    assert official["season"].tolist() == ["Fall", "Winter", "Summer"]

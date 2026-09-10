from __future__ import annotations

import json

import pandas as pd
import pytest

from fashion.config import ROOT, TEST_CSV
from fashion.data.hashing import compute_sha256
from fashion.task2.final_evaluation import load_final_evaluation_spec, validate_prediction_frame
from fashion.task2.final_evaluation_runner import load_verified_final_evaluation
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics

EVIDENCE = ROOT / "results/evidence/task2/final_evaluation"


def _receipt() -> dict[str, object]:
    return json.loads((EVIDENCE / "prediction_receipt.json").read_text(encoding="utf-8"))


def test_blind_receipt_precedes_protected_label_access_and_binds_i2() -> None:
    receipt = _receipt()
    spec = load_final_evaluation_spec()

    assert receipt["phase"] == "blind_prediction_before_holdout_label_access"
    assert receipt["labels_opened"] is False
    assert receipt["teacher_test_scored"] is False
    assert receipt["model_changed"] is False
    assert receipt["retuning_allowed"] is False
    assert receipt["git"]["tracked_files_dirty"] is False
    assert len(receipt["git"]["commit"]) == 40
    assert receipt["model"]["run_id"] == spec.run_id
    assert receipt["model"]["bundle_sha256"] == spec.bundle_sha256
    assert receipt["b0"]["majority_label"] == "Summer"


def test_blind_artifact_ledger_hashes_every_output() -> None:
    receipt = _receipt()

    for record in receipt["artifacts"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert compute_sha256(path) == record["sha256"]


def test_holdout_predictions_cover_every_id_without_target_columns() -> None:
    image_manifest = pd.read_csv(EVIDENCE / "holdout_image_manifest.csv")
    predictions = pd.read_csv(EVIDENCE / "holdout_predictions.csv")
    robustness = pd.read_csv(EVIDENCE / "holdout_robustness_predictions.csv")

    assert len(image_manifest) == 5_778
    assert image_manifest["id"].is_unique
    assert not {"season", "y_true", "articleType", "gender", "usage"} & set(image_manifest)
    assert not {"season", "y_true", "articleType", "gender", "usage"} & set(predictions)
    validate_prediction_frame(predictions, expected_ids=image_manifest["id"].tolist())
    assert len(robustness) == 5 * 5_778
    assert not {"season", "y_true", "articleType", "gender", "usage"} & set(robustness)
    for _, condition in robustness.groupby("condition", observed=True):
        validate_prediction_frame(condition, expected_ids=image_manifest["id"].tolist())


def test_teacher_output_matches_template_order_and_two_column_schema() -> None:
    template = pd.read_csv(TEST_CSV, usecols=["id"])
    predictions = pd.read_csv(ROOT / "results/season_test_predictions.csv")

    assert predictions.columns.tolist() == ["id", "season"]
    assert len(predictions) == 5_829
    assert predictions["id"].is_unique
    assert predictions["id"].tolist() == template["id"].tolist()
    assert predictions["season"].notna().all()
    assert predictions["season"].astype(str).str.strip().ne("").all()
    assert set(predictions["season"]) <= {"Fall", "Spring", "Summer", "Winter"}


def test_scored_evaluation_manifest_hash_verifies_every_artifact() -> None:
    manifest = load_verified_final_evaluation()

    assert manifest["status"] == "complete"
    assert manifest["coverage"] == {
        "holdout_rows": 5_778,
        "valid_season_rows": 5_778,
        "invalid_season_rows": 0,
        "quarantine_rows_excluded": 61,
        "product_family_groups": 4_110,
        "teacher_test_rows": 5_829,
        "prediction_failures": 0,
    }
    assert manifest["model"]["scratch"] is True
    assert manifest["model"]["image_only_inference"] is True
    assert manifest["model"]["model_changed_after_unlock"] is False
    assert manifest["official_test"]["scored"] is False


def test_saved_full_precision_probabilities_reproduce_holdout_metrics() -> None:
    scored = pd.read_csv(EVIDENCE / "holdout_predictions_and_labels.csv")
    recorded = json.loads((EVIDENCE / "holdout_metrics.json").read_text(encoding="utf-8"))
    probability_columns = [f"prob_{label}" for label in SEASON_LABELS]

    recomputed = multiclass_metrics(
        scored["actual_season"].astype(str).to_numpy(),
        probabilities=scored.loc[:, probability_columns].to_numpy(dtype=float),
        labels=SEASON_LABELS,
        y_pred=scored["i2_prediction"].astype(str).to_numpy(),
    )

    assert len(scored) == 5_778
    assert scored["id"].is_unique
    assert recomputed["macro_f1"] == recorded["I2_frozen_temperature"]["macro_f1"]
    assert recomputed["accuracy"] == recorded["I2_frozen_temperature"]["accuracy"]
    assert recomputed["nll"] == pytest.approx(
        recorded["I2_frozen_temperature"]["nll"], abs=1e-12
    )
    assert recomputed["brier"] == pytest.approx(
        recorded["I2_frozen_temperature"]["brier"], abs=1e-12
    )
    assert recomputed["ece"] == pytest.approx(
        recorded["I2_frozen_temperature"]["ece"], abs=1e-12
    )


def test_holdout_scorecard_and_bootstrap_support_the_frozen_judgement() -> None:
    scorecard = pd.read_csv(EVIDENCE / "holdout_scorecard.csv").set_index("model")
    intervals = pd.read_csv(EVIDENCE / "holdout_bootstrap_intervals.csv").set_index("metric")

    assert scorecard.at["I2 frozen", "macro_f1"] == pytest.approx(0.7533847968563716)
    assert scorecard.at["B0 majority", "macro_f1"] == pytest.approx(0.16574106213120443)
    assert scorecard.at[
        "I2 frozen", "holdout_minus_development_macro_f1"
    ] == pytest.approx(0.0006978408982745155)
    assert intervals.at["i2_minus_b0_macro_f1", "replicates"] == 10_000
    assert intervals.at["i2_minus_b0_macro_f1", "sampled_group_count"] == 4_110
    assert intervals.at["i2_minus_b0_macro_f1", "lower_95"] > 0


def test_unlock_receipt_forbids_post_holdout_model_changes() -> None:
    unlock = json.loads((EVIDENCE / "unlock_receipt.json").read_text(encoding="utf-8"))

    assert unlock["holdout_opened"] is True
    assert unlock["valid_season_rows"] == 5_778
    assert unlock["blank_or_invalid_season_rows"] == 0
    assert unlock["quarantine_rows_excluded"] == 61
    assert unlock["model_retrained"] is False
    assert unlock["model_retuned"] is False
    assert unlock["winner_changed"] is False
    assert unlock["temperature_refit"] is False


def test_final_evaluation_covers_slices_robustness_and_tailored_figures() -> None:
    slices = pd.read_csv(EVIDENCE / "holdout_slice_metrics.csv")
    robustness = pd.read_csv(EVIDENCE / "holdout_robustness_metrics.csv")
    expected_families = {
        "article_type_shortcut",
        "acquisition_year",
        "file_size_quartile",
        "product_family_size",
        "image_mode",
    }
    expected_conditions = {
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    }

    assert set(slices["slice_family"]) == expected_families
    assert set(robustness["condition"]) == expected_conditions
    assert robustness.loc[
        robustness["condition"].eq("brightness_0_85"), "delta_macro_f1_vs_clean"
    ].iloc[0] < -0.30
    figure_names = {
        "holdout_scorecard.png",
        "holdout_per_class_confusion.png",
        "holdout_bootstrap_intervals.png",
        "holdout_calibration_risk.png",
        "holdout_slices_robustness.png",
        "holdout_error_examples.png",
    }
    figure_dir = ROOT / "results/figures/task2/final_evaluation"
    for name in figure_names:
        assert (figure_dir / name).stat().st_size > 10_000

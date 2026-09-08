from __future__ import annotations

import json

import pandas as pd

from fashion.config import ROOT, TEST_CSV
from fashion.data.hashing import compute_sha256
from fashion.task2.final_evaluation import load_final_evaluation_spec, validate_prediction_frame

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

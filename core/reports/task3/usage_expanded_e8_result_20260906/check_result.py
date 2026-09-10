"""Verify the downloaded Usage run and compare the fixed teacher validation rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_usage_expanded import (
    CLASSES,
    E8_DIRECTORY,
    EXPERIMENT,
    MAP_SHA256,
    SPLIT_SHA256,
    check_e8_sources,
    expanded_usage_spec,
    read_predictions,
    source_metrics,
    training_scope,
    validate_dataset,
    write_json,
)

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_expanded_e8_20260906"
METRIC_KEYS = ("macro_f1", "accuracy", "nll", "brier", "ece_15")


def assert_metrics(actual, expected):
    for key in METRIC_KEYS:
        assert np.isclose(actual[key], expected[key], atol=1e-7, rtol=0), key


def main():
    rows = pd.read_csv(EVIDENCE / "results/runs.csv", keep_default_na=False)
    rows = rows.sort_values("validation_fold")
    assert len(rows) == 5 and rows.validation_fold.tolist() == list(range(5))
    assert rows.status.eq("complete").all()
    assert rows.experiment_id.eq(EXPERIMENT).all()
    assert rows.last_completed_stage.eq("diagnostic_bundle_complete").all()
    assert rows.split_digest.eq(SPLIT_SHA256).all()
    assert rows.label_map_digest.eq(MAP_SHA256).all()
    splits, contract = validate_dataset(check_images=False)
    recipe = expanded_usage_spec().to_dict()
    base = Task3BaselineConfig(target="usage").to_dict()
    predicted, expected_rows, fold_table, corruptions = [], [], [], []
    hash_checks = []
    for row in rows.itertuples():
        directory = EVIDENCE / row.run_id
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics == json.loads(row.metrics_json)
        config = json.loads((directory / "config.json").read_text())
        assert config["child_experiment"] == recipe
        assert all(config.get(key) == value for key, value in base.items())
        assert metrics["epochs_completed"] == metrics["selected_epoch"] == 30
        assert metrics["expanded_dataset"]["initialization"] == "fresh_scratch_weights"
        hashes = {
            "final_epoch.pt": row.checkpoint_sha256,
            "oof_predictions.csv": row.prediction_sha256,
            **metrics["expanded_artifact_sha256"],
        }
        for name, expected_hash in hashes.items():
            assert compute_sha256(directory / name) == expected_hash, (row.run_id, name)
            hash_checks.append({"run_id": row.run_id, "file": name, "sha256": expected_hash})
        train, expected = training_scope(splits, int(row.validation_fold))
        assert len(train) == row.training_product_count
        assert len(expected) == row.validation_product_count
        validation = validate_oof(
            read_predictions(directory / "oof_predictions.csv"),
            expected,
            target="usage",
            classes=CLASSES,
            run_ids_by_fold={int(row.validation_fold): row.run_id},
        )
        assert_metrics(oof_metrics(validation, CLASSES), metrics)
        scopes = source_metrics(validation)
        for scope, measured in scopes.items():
            recorded = metrics["source_metrics"]["validation"][scope]
            assert measured["rows"] == recorded["rows"]
            assert_metrics(measured["metrics"], recorded["metrics"])
        history = pd.read_csv(directory / "history.csv")
        assert history.epoch.tolist() == list(range(1, 31))
        assert history.loc[history.selected_checkpoint, "epoch"].tolist() == [30]
        predicted.append(validation)
        expected_rows.append(expected)
        robust = pd.read_csv(directory / "robustness.csv")
        assert len(robust) == 5 and robust.run_id.eq(row.run_id).all()
        corruptions.append(robust)
        teacher_train = metrics["source_metrics"]["clean_training"]["teacher"]["metrics"]
        teacher_validation = scopes["teacher"]["metrics"]
        fold_table.append(
            {
                "fold": int(row.validation_fold),
                "epochs": 30,
                "teacher_train_f1": teacher_train["macro_f1"],
                "teacher_validation_f1": teacher_validation["macro_f1"],
                "teacher_gap": teacher_train["macro_f1"] - teacher_validation["macro_f1"],
                "combined_validation_f1": metrics["macro_f1"],
                "train_minutes": float(row.train_seconds) / 60,
            }
        )
    pooled = validate_oof(
        read_predictions(EVIDENCE / "aggregate/oof_predictions.csv"),
        pd.concat(expected_rows, ignore_index=True),
        target="usage",
        classes=CLASSES,
        run_ids_by_fold=dict(zip(rows.validation_fold, rows.run_id, strict=True)),
    )
    pd.testing.assert_frame_equal(
        pooled.sort_values("id").reset_index(drop=True),
        pd.concat(predicted).sort_values("id").reset_index(drop=True),
        check_dtype=False,
    )
    measured_scopes = source_metrics(pooled)
    aggregate = json.loads((EVIDENCE / "aggregate/metrics.json").read_text())
    assert_metrics(measured_scopes["combined"]["metrics"], aggregate)
    comparison = json.loads((EVIDENCE / "aggregate/teacher_comparison.json").read_text())
    for scope, measured in measured_scopes.items():
        assert measured["rows"] == comparison["sources"][scope]["rows"]
        assert_metrics(measured["metrics"], comparison["sources"][scope]["metrics"])
    parents = check_e8_sources(
        directory=ROOT / "results/evidence/task3" / E8_DIRECTORY,
        registry_path=ROOT / "results/runs.csv",
    )
    parent = pd.concat([parents[fold]["predictions"] for fold in range(5)], ignore_index=True)
    teacher = pooled.loc[pooled.source_dataset.eq("teacher")]
    assert set(parent.id) == set(teacher.id)
    parent_metrics = oof_metrics(parent, CLASSES)
    assert_metrics(parent_metrics, comparison["teacher_reference_e8"])
    teacher_metrics = measured_scopes["teacher"]["metrics"]
    change = teacher_metrics["macro_f1"] - parent_metrics["macro_f1"]
    assert np.isclose(change, comparison["teacher_macro_f1_change"], atol=1e-12)
    class_table = pd.DataFrame(comparison["teacher_per_class"])
    pd.testing.assert_frame_equal(
        class_table,
        pd.read_csv(EVIDENCE / "aggregate/teacher_per_class.csv", keep_default_na=False).loc[
            :, class_table.columns
        ],
        check_dtype=False,
        atol=1e-12,
    )
    folds = pd.DataFrame(fold_table)
    folds["e8_teacher_validation_f1"] = [parents[fold]["metrics"]["macro_f1"] for fold in range(5)]
    folds["teacher_validation_change"] = (
        folds.teacher_validation_f1 - folds.e8_teacher_validation_f1
    )
    folds.to_csv(REPORT / "fold_comparison.csv", index=False)
    robust = pd.concat(corruptions, ignore_index=True)
    robust.groupby("corruption")[["macro_f1", "macro_f1_change"]].mean().to_csv(
        REPORT / "robustness_summary.csv"
    )
    summary = {
        "project_goal": "Good predictions on the teacher's unseen test set",
        "model_selection_scope": "Original teacher validation images",
        "completed_folds": 5,
        "epochs_per_fold": 30,
        "verified_artifact_hashes": len(hash_checks),
        "prediction_ids_labels_folds_families_and_probabilities_verified": True,
        "teacher_comparison_uses_identical_ids": True,
        "holdout_and_prediction_rows_in_evaluation": 0,
        "teacher_rows": len(teacher),
        "e8_teacher_macro_f1": parent_metrics["macro_f1"],
        "expanded_teacher_macro_f1": teacher_metrics["macro_f1"],
        "teacher_macro_f1_change": change,
        "combined_macro_f1": aggregate["macro_f1"],
        "teacher_fold_wins": int(folds.teacher_validation_change.gt(0).sum()),
        "mean_teacher_train_f1": float(folds.teacher_train_f1.mean()),
        "mean_teacher_validation_f1": float(folds.teacher_validation_f1.mean()),
        "train_minutes_total": float(folds.train_minutes.sum()),
        "sources": measured_scopes,
        "class_changes": comparison["teacher_per_class"],
        "decision": (
            "Do not replace E8 with this expanded run: teacher-only validation is lower. "
            "The combined-data score is a secondary diagnostic, not evidence of "
            "improved teacher test-set performance."
        ),
        "limits": [
            "One fixed seed; this does not establish statistical significance.",
            "Teacher image counts: Home 1, Party 12, Smart Casual 47, Travel 22.",
            "Added-source class scores have small samples and a different source distribution.",
            "No training or checkpoint promotion was performed during this review.",
        ],
    }
    write_json(summary, REPORT / "verified_summary.json")
    write_json({"hash_checks": hash_checks, "dataset_contract": contract}, REPORT / "checks.json")
    compact = {
        key: value for key, value in summary.items() if key not in {"sources", "class_changes"}
    }
    print(json.dumps(compact, indent=2))
    print(folds.to_string(index=False))
    print(robust.groupby("corruption")[["macro_f1", "macro_f1_change"]].mean().to_string())


if __name__ == "__main__":
    main()

"""Verify saved Usage v2 evidence and compare the unchanged teacher validation rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_samples
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded as previous
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import oof_metrics, paired_family_bootstrap, validate_oof

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_expanded_v2_e8_20260906"
METRICS = ("macro_f1", "accuracy", "nll", "brier", "ece_15")


def same_metrics(measured, recorded):
    for key in METRICS:
        assert np.isclose(measured[key], recorded[key], atol=1e-7, rtol=0), key


def same_sources(measured, recorded):
    assert set(measured) == set(recorded)
    for name, source in measured.items():
        assert source["rows"] == recorded[name]["rows"]
        if source["rows"]:
            same_metrics(source["metrics"], recorded[name]["metrics"])


def source_class_table(predictions):
    records = []
    for cohort, frame in predictions.groupby(v2.COHORT_COLUMN):
        measured = oof_metrics(frame, v2.CLASSES)
        for item in measured["per_class"]:
            label = item["class_name"]
            truth = frame.true_label.eq(label)
            predicted = frame.predicted_label.eq(label)
            records.append(
                {
                    "cohort": cohort,
                    "class_name": label,
                    "support": int(truth.sum()),
                    "correct": int((truth & predicted).sum()),
                    "predicted": int(predicted.sum()),
                    "recall": item["recall"],
                    "precision": item["precision"],
                    "f1": item["f1"],
                }
            )
    return pd.DataFrame(records)


def main():
    rows = RunRegistry(EVIDENCE / "results/runs.csv")._read_rows()
    rows = sorted(rows, key=lambda row: int(row["validation_fold"]))
    assert len(rows) == 5
    assert [int(row["validation_fold"]) for row in rows] == list(range(5))
    assert len({row["run_id"] for row in rows}) == 5
    assert all(row["last_completed_stage"] == "diagnostic_bundle_complete" for row in rows)
    splits, contract = v2.validate_dataset(check_images=False)
    references = v2.check_references(
        e8_directory=ROOT / "results/evidence/task3" / previous.E8_DIRECTORY,
        source_registry_path=ROOT / "results/runs.csv",
        previous_directory=ROOT / "results/evidence/task3/usage_expanded_e8_20260906",
        previous_registry_path=(
            ROOT / "results/evidence/task3/usage_expanded_e8_20260906/results/runs.csv"
        ),
    )
    predictions, folds, robust_frames, hashes = [], [], [], []
    for row in rows:
        fold = int(row["validation_fold"])
        directory = EVIDENCE / row["run_id"]
        result = v2._verified_completed_fold(
            row=row,
            path=directory,
            splits=splits,
            fold=fold,
            spec=v2.expanded_usage_spec(),
            split_digest=v2.SPLIT_SHA256,
        )
        metrics = result["metrics"]
        train, expected = v2.training_scope(splits, fold)
        validation = v2.source_predictions(
            v2.read_predictions(directory / "oof_predictions.csv"), expected
        )
        clean_training = v2.source_predictions(
            v2.read_predictions(directory / "training_predictions.csv"), train
        )
        assert clean_training.run_id.eq(row["run_id"]).all()
        scopes = v2.source_metrics(validation)
        train_scopes = v2.source_metrics(clean_training)
        same_sources(scopes, metrics["source_metrics"]["validation"])
        same_sources(train_scopes, metrics["source_metrics"]["clean_training"])
        source_file = json.loads((directory / "source_metrics.json").read_text())
        assert source_file == metrics["source_metrics"]
        history = pd.read_csv(directory / "history.csv")
        assert history.epoch.tolist() == list(range(1, 31))
        assert history.loc[history.selected_checkpoint, "epoch"].tolist() == [30]
        assert np.isclose(history.iloc[-1].validation_macro_f1, metrics["macro_f1"])
        hashes.extend(
            {"run_id": row["run_id"], "file": name, "sha256": digest}
            for name, digest in {
                "final_epoch.pt": row["checkpoint_sha256"],
                "oof_predictions.csv": row["prediction_sha256"],
                **metrics["expanded_artifact_sha256"],
            }.items()
        )
        teacher_train = train_scopes["teacher"]["metrics"]["macro_f1"]
        teacher_validation = scopes["teacher"]["metrics"]["macro_f1"]
        record = {
            "fold": fold,
            "run_id": row["run_id"],
            "teacher_train_f1": teacher_train,
            "teacher_validation_f1": teacher_validation,
            "teacher_gap": teacher_train - teacher_validation,
            "combined_validation_f1": metrics["macro_f1"],
            "train_minutes": float(row["train_seconds"]) / 60,
        }
        for name, reference in references.items():
            teacher_reference = reference[fold]["predictions"]
            if name == "previous_expansion":
                teacher_reference = teacher_reference.loc[
                    teacher_reference.source_dataset.eq("teacher")
                ]
            record[f"{name}_f1"] = oof_metrics(teacher_reference, v2.CLASSES)["macro_f1"]
            record[f"change_from_{name}"] = teacher_validation - record[f"{name}_f1"]
        folds.append(record)
        predictions.append(validation)
        robust = pd.read_csv(directory / "robustness.csv")
        assert len(robust) == 5 and robust.run_id.eq(row["run_id"]).all()
        robust_frames.append(robust)
        print(f"Verified fold {fold}: files, recipe, validation and clean training", flush=True)

    expected = get_samples(splits.loc[splits.partition.eq("development")], target="usage")
    pooled = validate_oof(
        v2.read_predictions(EVIDENCE / "aggregate/oof_predictions.csv"),
        expected,
        target="usage",
        classes=v2.CLASSES,
        run_ids_by_fold={int(row["validation_fold"]): row["run_id"] for row in rows},
    )
    pd.testing.assert_frame_equal(
        pooled,
        pd.concat(predictions, ignore_index=True).sort_values("id").reset_index(drop=True),
        check_dtype=False,
    )
    same_metrics(
        oof_metrics(pooled, v2.CLASSES),
        json.loads((EVIDENCE / "aggregate/metrics.json").read_text()),
    )
    comparison = v2.build_comparison(
        pooled, splits=splits, folds=tuple(range(5)), references=references, contract=contract
    )
    saved = json.loads((EVIDENCE / "aggregate/teacher_comparison.json").read_text())
    same_sources(comparison["sources"], saved["sources"])
    for name, score in comparison["teacher_references"].items():
        same_metrics(score, saved["teacher_references"][name])
    pd.testing.assert_frame_equal(
        pd.DataFrame(comparison["teacher_per_class"]),
        pd.read_csv(EVIDENCE / "aggregate/teacher_per_class.csv", keep_default_na=False),
        check_dtype=False,
        atol=1e-12,
    )

    teacher = pooled.loc[pooled.source_dataset.eq("teacher")]
    bootstrap = {}
    for name, reference in references.items():
        parent = pd.concat([reference[fold]["predictions"] for fold in range(5)])
        if name == "previous_expansion":
            parent = parent.loc[parent.source_dataset.eq("teacher")]
        print(f"Checking teacher family uncertainty against {name}", flush=True)
        bootstrap[name] = paired_family_bootstrap(teacher, parent, classes=v2.CLASSES)

    fold_table = pd.DataFrame(folds)
    fold_table.to_csv(REPORT / "fold_comparison.csv", index=False)
    class_sources = source_class_table(pooled)
    class_sources.to_csv(REPORT / "source_class_scores.csv", index=False)
    pd.DataFrame(comparison["teacher_per_class"]).to_csv(
        REPORT / "teacher_per_class.csv", index=False
    )
    teacher_confusion = pd.crosstab(teacher.true_label, teacher.predicted_label).reindex(
        index=v2.CLASSES, columns=v2.CLASSES, fill_value=0
    )
    teacher_confusion.to_csv(REPORT / "teacher_confusion.csv")
    robust = pd.concat(robust_frames).groupby("corruption")[["macro_f1", "macro_f1_change"]].mean()
    robust.to_csv(REPORT / "combined_robustness.csv")
    summary = {
        "decision": "Do not replace original E8 on this development evidence",
        "scope": "Same 32,772 teacher validation images and original nine Usage classes",
        "completed_folds": 5,
        "epochs_per_fold": 30,
        "verified_artifact_hashes": len(hashes),
        "teacher_macro_f1": comparison["sources"]["teacher"]["metrics"]["macro_f1"],
        "teacher_macro_f1_changes": comparison["teacher_macro_f1_changes"],
        "teacher_reference_scores": comparison["teacher_references"],
        "combined_macro_f1": comparison["sources"]["combined"]["metrics"]["macro_f1"],
        "teacher_fold_wins": {
            name: int(fold_table[f"change_from_{name}"].gt(0).sum()) for name in references
        },
        "mean_teacher_training_f1": float(fold_table.teacher_train_f1.mean()),
        "mean_teacher_validation_f1": float(fold_table.teacher_validation_f1.mean()),
        "mean_teacher_gap": float(fold_table.teacher_gap.mean()),
        "train_minutes_total": float(fold_table.train_minutes.sum()),
        "new_source_correct": int(
            pooled.loc[pooled[v2.COHORT_COLUMN].eq("new_added")]
            .eval("true_index == predicted_index")
            .sum()
        ),
        "sources": comparison["sources"],
        "paired_family_bootstrap": bootstrap,
        "limits": [
            "One seed; family bootstrap does not measure variation across training seeds.",
            "Prior selection and earlier holdout/test inspection are not erased by resampling.",
            "Teacher supports: Home 1, Party 12, Smart Casual 47, Travel 22.",
            "Source-specific macro-F1 retains all nine classes, including absent classes.",
            "Robustness scores use combined validation rows, not teacher-only rows.",
            "This review used saved development predictions only; no refit or new test evaluation.",
        ],
    }
    v2.write_json(summary, REPORT / "verified_summary.json")
    v2.write_json(
        {
            "download_method": "rclone copy; read-only source access",
            "source": (
                "gdrive:MLA2/task3_usage_expanded_v2_e8/experiments/t3_usage_expanded_v2_e8/usage"
            ),
            "dataset": contract,
            "hash_checks": hashes,
            "registry_sha256": compute_sha256(EVIDENCE / "results/runs.csv"),
            "aggregate_predictions_equal_fold_predictions": True,
            "all_validation_and_clean_training_source_scores_recomputed": True,
            "reference_folds_verified": 10,
            "holdout_or_test_rows_evaluated": 0,
        },
        REPORT / "checks.json",
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"sources", "teacher_reference_scores"}},
            indent=2,
        )
    )
    print(fold_table.to_string(index=False))
    print(robust.to_string())


if __name__ == "__main__":
    main()

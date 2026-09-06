"""Use the authorized holdout Usage labels only to score already frozen predictions."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fashion.config import ROOT, TEACHER_TRAIN_CSV
from fashion.data.hashing import compute_sha256
from fashion.train.metrics import classification_metrics

REPORT = Path(__file__).resolve().parent
TEST = ROOT / "reports/task3/usage_teacher_vs_expanded_test_20260906"
MODELS = ("E1", "E8", "Expanded")


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_predictions():
    freeze = json.loads((REPORT / "prediction_freeze.json").read_text())
    assert freeze["holdout_reference_labels_opened"] is False
    for name, digest in freeze["outputs"].items():
        assert compute_sha256(REPORT / name) == digest, name
    return freeze


def main():
    if (REPORT / "holdout_metrics.json").exists():
        raise RuntimeError("The holdout comparison is saved; preserve its first-access record.")
    freeze = verify_predictions()
    recipe = json.loads((REPORT / "inference_recipe.json").read_text())
    classes = recipe["class_names"]
    manifest = pd.read_csv(REPORT / "holdout_image_manifest.csv", keep_default_na=False)
    assert len(manifest) == freeze["holdout_rows"]
    ids = set(manifest.id)
    reference = {}
    extra_name_fields = []
    opened_at = datetime.now(timezone.utc).isoformat()
    assert opened_at > freeze["frozen_at_utc"]
    with TEACHER_TRAIN_CSV.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert "id" in reader.fieldnames and "usage" in reader.fieldnames
        for row in reader:
            product_id = int(row["id"])
            if product_id not in ids:
                continue
            assert product_id not in reference
            usage = row["usage"].strip()
            assert usage == "" or usage in classes, (product_id, usage)
            reference[product_id] = usage
            if None in row:
                extra_name_fields.append(product_id)
    assert set(reference) == ids
    comparison = manifest[["id", "product_family_group"]].copy()
    comparison["actual_usage"] = comparison.id.map(reference)
    comparison["scorable"] = comparison.actual_usage.isin(classes)
    scored = comparison.loc[comparison.scorable]
    truth = scored.actual_usage.to_numpy()
    labels = scored.actual_usage.map(dict(zip(classes, range(9), strict=True))).to_numpy()
    summary, per_class, models = [], [], {}
    for name in MODELS:
        probability_path = REPORT / name / "usage_holdout_probabilities.csv"
        prediction_path = REPORT / name / "usage_holdout_predictions.csv"
        probabilities = pd.read_csv(probability_path, keep_default_na=False)
        predictions = pd.read_csv(prediction_path, keep_default_na=False)
        assert probabilities.id.tolist() == predictions.id.tolist() == manifest.id.tolist()
        values = probabilities[[f"probability_{c}" for c in classes]].to_numpy()
        assert np.asarray(classes)[values.argmax(axis=1)].tolist() == predictions.usage.tolist()
        comparison[f"{name}_prediction"] = predictions.usage
        comparison[f"{name}_correct"] = (
            predictions.usage.eq(comparison.actual_usage) & comparison.scorable
        )
        metrics = classification_metrics(labels, values[comparison.scorable], classes)
        expected_predictions = predictions.loc[comparison.scorable, "usage"]
        np.testing.assert_allclose(metrics["accuracy"], accuracy_score(truth, expected_predictions))
        np.testing.assert_allclose(
            metrics["macro_f1"],
            f1_score(truth, expected_predictions, labels=classes, average="macro", zero_division=0),
        )
        metrics["correct"] = int(comparison[f"{name}_correct"].sum())
        metrics["incorrect"] = len(scored) - metrics["correct"]
        metrics["macro_f1_classes_present"] = float(
            np.mean([row["f1"] for row in metrics["per_class"] if row["support"] > 0])
        )
        models[name] = metrics
        summary.append(
            {
                "model": name,
                "holdout_images": len(manifest),
                "scored_images": len(scored),
                "correct": metrics["correct"],
                "incorrect": metrics["incorrect"],
                "accuracy": metrics["accuracy"],
                "macro_f1_nine_classes": metrics["macro_f1"],
                "macro_f1_classes_present": metrics["macro_f1_classes_present"],
            }
        )
        matrix = np.asarray(metrics["confusion_matrix"])
        assert int(np.trace(matrix)) == metrics["correct"]
        for index, row in enumerate(metrics["per_class"]):
            per_class.append({"model": name, **row, "correct": int(matrix[index, index])})
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
            REPORT / f"{name}_confusion_matrix.csv", index_label="actual_usage"
        )

    paired = {}
    for source, target in (("E1", "E8"), ("E1", "Expanded"), ("E8", "Expanded")):
        before = comparison.loc[comparison.scorable, f"{source}_correct"]
        after = comparison.loc[comparison.scorable, f"{target}_correct"]
        fixed = int((~before & after).sum())
        broken = int((before & ~after).sum())
        assert fixed - broken == models[target]["correct"] - models[source]["correct"]
        paired[f"{source}_to_{target}"] = {
            "fixed_errors": fixed,
            "new_errors": broken,
            "net_additional_correct": fixed - broken,
            "both_correct": int((before & after).sum()),
            "both_wrong": int((~before & ~after).sum()),
            "accuracy_change_percentage_points": 100 * (fixed - broken) / len(scored),
            "macro_f1_change": models[target]["macro_f1"] - models[source]["macro_f1"],
        }
    summary_frame = pd.DataFrame(summary)
    class_frame = pd.DataFrame(per_class)
    summary_frame.to_csv(REPORT / "holdout_summary.csv", index=False)
    class_frame.to_csv(REPORT / "holdout_per_class.csv", index=False)
    comparison.to_csv(REPORT / "holdout_predictions_and_labels.csv", index=False)
    comparison.loc[~comparison.scorable].to_csv(REPORT / "unscored_holdout_rows.csv", index=False)
    test_metrics = json.loads((TEST / "comparison_metrics.json").read_text())
    population_rows = []
    score_rows = []
    for population, model_results in (
        ("holdout", models),
        ("teacher_test", test_metrics["models"]),
    ):
        for name, metrics in model_results.items():
            score_rows.append(
                {
                    "population": population,
                    "model": name,
                    "images": metrics["support"],
                    "correct": metrics["correct"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
        for row in model_results["E1"]["per_class"]:
            population_rows.append(
                {
                    "population": population,
                    "usage": row["class_name"],
                    "images": row["support"],
                    "share": row["support"] / model_results["E1"]["support"],
                }
            )
    pd.DataFrame(population_rows).to_csv(REPORT / "holdout_and_test_class_mix.csv", index=False)
    pd.DataFrame(score_rows).to_csv(REPORT / "holdout_and_test_scores.csv", index=False)
    result = {
        "user_authorization": recipe["user_authorization"],
        "holdout_rows": len(manifest),
        "matched_reference_ids": len(reference),
        "scored_rows": len(scored),
        "blank_usage_labels": len(manifest) - len(scored),
        "source": str(TEACHER_TRAIN_CSV.relative_to(ROOT)),
        "source_sha256": compute_sha256(TEACHER_TRAIN_CSV),
        "predictions_frozen_at_utc": freeze["frozen_at_utc"],
        "holdout_usage_labels_opened_at_utc": opened_at,
        "label_scope": (
            "Matching holdout IDs and Usage only; no other targets or quarantine scoring"
        ),
        "training_tuning_or_model_selection_after_holdout_access": False,
        "class_names": classes,
        "models": models,
        "paired_changes": paired,
        "matching_source_rows_with_extra_csv_fields": extra_name_fields,
        "source_hashes": {
            str(p.relative_to(ROOT)): compute_sha256(p)
            for p in (
                Path(__file__),
                REPORT / "prediction_freeze.json",
                TEST / "comparison_metrics.json",
                ROOT / "src/fashion/train/metrics.py",
            )
        },
    }
    save_json(result, REPORT / "holdout_metrics.json")
    verify_predictions()
    print(summary_frame.to_string(index=False))
    print(json.dumps(paired, indent=2))
    print(class_frame[["model", "class_name", "support", "correct", "f1"]].to_string(index=False))
    print("Scoring coverage:", len(scored), "/", len(manifest))


if __name__ == "__main__":
    main()

"""Compare already frozen Usage predictions with matching high-resolution labels."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256

REPORT = Path(__file__).resolve().parent
REFERENCE = (
    ROOT / "data/raw/external/fashion_product_images_v1_legacy_extras/fashion-dataset/styles.csv"
)
FIGURE = ROOT / "results/figures/task3/usage_expanded_e8_test_20260906.png"


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    if (REPORT / "test_metrics.json").exists():
        raise RuntimeError("The original test comparison is saved; keep its first-access record.")
    freeze_path = REPORT / "prediction_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    assert freeze["reference_labels_opened"] is False
    for name, digest in freeze["outputs"].items():
        assert compute_sha256(REPORT / name) == digest, name
    recipe = json.loads((REPORT / "inference_recipe.json").read_text())
    classes = recipe["class_names"]
    predictions = pd.read_csv(REPORT / "usage_test_predictions.csv", keep_default_na=False)
    assert predictions.id.is_unique
    assert len(predictions) == freeze["test_rows"]
    assert set(predictions.usage).issubset(classes)
    test_ids = set(predictions.id)

    opened_at = datetime.now(timezone.utc).isoformat()
    assert opened_at > freeze["frozen_at_utc"]
    matches = {}
    duplicates = []
    nonstandard_matching_rows = []
    with REFERENCE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header[:10] == [
            "id",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
            "productDisplayName",
        ], header
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            product_id = int(row[0])
            if product_id not in test_ids:
                continue
            assert len(row) >= 10, (product_id, line_number)
            # Additional unquoted commas belong to the final product-name field.
            # Only matching IDs and the Usage field are used for this comparison.
            usage = row[8].strip()
            if len(row) != len(header):
                nonstandard_matching_rows.append(product_id)
            if product_id in matches:
                duplicates.append(product_id)
                assert matches[product_id] == usage, product_id
            matches[product_id] = usage

    comparison = predictions.rename(columns={"usage": "predicted_usage"}).copy()
    comparison["reference_found"] = comparison.id.isin(matches)
    comparison["actual_usage"] = comparison.id.map(matches).fillna("")
    comparison["scorable"] = comparison.actual_usage.isin(classes)
    comparison["correct"] = comparison.scorable & comparison.predicted_usage.eq(
        comparison.actual_usage
    )
    comparison.to_csv(REPORT / "usage_test_comparison.csv", index=False)
    comparison.loc[~comparison.correct].to_csv(
        REPORT / "usage_test_errors_or_unscored.csv", index=False
    )
    scored = comparison.loc[comparison.scorable]
    assert len(scored) > 0
    truth = scored.actual_usage
    predicted = scored.predicted_usage
    matrix = confusion_matrix(truth, predicted, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=classes, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "usage": classes,
            "test_images": support,
            "correct": np.diag(matrix),
            "predicted_count": matrix.sum(axis=0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    per_class.to_csv(REPORT / "usage_test_per_class.csv", index=False)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        REPORT / "usage_test_confusion_matrix.csv", index_label="actual_usage"
    )
    correct = int(scored.correct.sum())
    accuracy = float(accuracy_score(truth, predicted))
    macro_f1 = float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0))
    assert int(np.trace(matrix)) == correct
    assert accuracy == correct / len(scored)
    assert int(matrix.sum()) == len(scored)
    metrics = {
        "model": "Usage E8 trained from scratch on teacher plus 120 admitted external images",
        "prediction_rule": recipe["aggregation"],
        "test_rows": len(predictions),
        "matched_reference_ids": int(comparison.reference_found.sum()),
        "scored_rows": len(scored),
        "correct": correct,
        "incorrect": len(scored) - correct,
        "accuracy": accuracy,
        "macro_f1_all_nine_classes": macro_f1,
        "macro_f1_classes_present_in_reference": float(f1[support > 0].mean()),
        "balanced_accuracy_classes_present": float(recall[support > 0].mean()),
        "class_names": classes,
        "reference_class_counts": {str(k): int(v) for k, v in Counter(truth).items()},
        "per_class": per_class.to_dict("records"),
        "missing_reference_ids": comparison.loc[~comparison.reference_found, "id"].tolist(),
        "unscorable_reference_values": comparison.loc[
            comparison.reference_found & ~comparison.scorable, ["id", "actual_usage"]
        ].to_dict("records"),
        "literal_NA_is_a_class": True,
        "blank_reference_values_are_not_imputed": True,
        "reference_path": str(REFERENCE.relative_to(ROOT)),
        "reference_sha256": compute_sha256(REFERENCE),
        "reference_labels_opened_at_utc": opened_at,
        "predictions_frozen_at_utc": freeze["frozen_at_utc"],
        "prediction_freeze_sha256": compute_sha256(freeze_path),
        "matching_ids_with_extra_product_name_columns": nonstandard_matching_rows,
        "duplicate_matching_reference_ids": duplicates,
        "training_tuning_or_checkpoint_selection_after_label_access": False,
        "evaluation_scope": (
            "Official teacher test IDs only; high-resolution metadata supplies Usage labels only"
        ),
    }
    save_json(metrics, REPORT / "test_metrics.json")

    fig, ax = plt.subplots(figsize=(11, 9))
    plotted = ax.imshow(
        np.maximum(matrix, 0.6), cmap="Blues", norm=LogNorm(vmin=0.6, vmax=max(1, matrix.max()))
    )
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted Usage", labelpad=12)
    ax.set_ylabel("Actual Usage from matching high-resolution metadata", labelpad=12)
    for row in range(len(classes)):
        for column in range(len(classes)):
            count = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(count) if count else "·",
                ha="center",
                va="center",
                color="white" if plotted.norm(max(count, 0.6)) > 0.58 else "#22384b",
                fontsize=10,
            )
    fig.suptitle("Expanded Usage model: teacher test results", fontsize=17, y=0.98)
    ax.set_title(
        f"Accuracy {accuracy:.2%} ({correct:,}/{len(scored):,})  ·  "
        f"Nine-class macro-F1 {macro_f1:.4f}\n"
        "Equal probability average of the five saved fold models",
        fontsize=11,
        pad=18,
    )
    colorbar = fig.colorbar(plotted, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Image count (log colour scale)")
    fig.text(
        0.5,
        0.01,
        "Predictions were saved before reference labels were opened. "
        "No fitting or tuning used those labels.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170, facecolor="white")
    plt.close(fig)

    for name, digest in freeze["outputs"].items():
        assert compute_sha256(REPORT / name) == digest, name
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "test_rows",
                    "matched_reference_ids",
                    "scored_rows",
                    "correct",
                    "incorrect",
                    "accuracy",
                    "macro_f1_all_nine_classes",
                    "reference_class_counts",
                    "unscorable_reference_values",
                    "missing_reference_ids",
                    "matching_ids_with_extra_product_name_columns",
                )
            },
            indent=2,
        )
    )
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()

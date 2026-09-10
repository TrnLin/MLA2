"""Score the frozen original-model predictions against the same test reference."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train.metrics import classification_metrics

REPORT = Path(__file__).resolve().parent
PRIOR = ROOT / "reports/task3/usage_expanded_e8_test_20260906"
REFERENCE = (
    ROOT / "data/raw/external/fashion_product_images_v1_legacy_extras/fashion-dataset/styles.csv"
)
FIGURE = ROOT / "results/figures/task3/usage_teacher_vs_expanded_test_20260906.png"
NAMES = {
    "E1": "Original baseline (E1)",
    "E8": "Original translation model (E8)",
    "Expanded": "E8 with 120 added images",
}


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_outputs(directory):
    freeze = json.loads((directory / "prediction_freeze.json").read_text())
    for name, digest in freeze["outputs"].items():
        assert compute_sha256(directory / name) == digest, name
    return freeze


def main():
    if (REPORT / "comparison_metrics.json").exists():
        raise RuntimeError("This saved test comparison is already complete.")
    freeze = verify_outputs(REPORT)
    verify_outputs(PRIOR)
    recipe = json.loads((REPORT / "inference_recipe.json").read_text())
    classes = recipe["class_names"]
    old_metrics = json.loads((PRIOR / "test_metrics.json").read_text())
    assert compute_sha256(REFERENCE) == old_metrics["reference_sha256"]
    template = pd.read_csv(
        ROOT / "data/raw/teacher/test/styles_prediction.csv", usecols=["id"], keep_default_na=False
    )
    ids = set(template.id)
    reference_opened_at = datetime.now(timezone.utc).isoformat()
    assert reference_opened_at > freeze["frozen_at_utc"]
    reference = {}
    extra_name_fields = []
    with REFERENCE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header.index("id") == 0 and header.index("usage") == 8
        assert header[9] == "productDisplayName" and len(header) == 10
        for row in reader:
            if not row or int(row[0]) not in ids:
                continue
            product_id = int(row[0])
            assert len(row) >= len(header)
            assert product_id not in reference
            usage = row[8].strip()
            assert usage in classes
            reference[product_id] = usage
            if len(row) > len(header):
                extra_name_fields.append(product_id)
    assert set(reference) == ids
    comparison = template.copy()
    comparison["actual_usage"] = comparison.id.map(reference)
    truth = comparison.actual_usage.to_numpy()
    labels = comparison.actual_usage.map(dict(zip(classes, range(9), strict=True))).to_numpy()
    old_comparison = pd.read_csv(PRIOR / "usage_test_comparison.csv", keep_default_na=False)
    assert old_comparison.id.tolist() == template.id.tolist()
    assert old_comparison.actual_usage.tolist() == comparison.actual_usage.tolist()

    models, summary_rows, class_rows = {}, [], []
    for key, name in NAMES.items():
        directory = PRIOR if key == "Expanded" else REPORT / key
        probabilities = pd.read_csv(
            directory / "usage_test_probabilities.csv", keep_default_na=False
        )
        predictions = pd.read_csv(directory / "usage_test_predictions.csv", keep_default_na=False)
        assert predictions.id.tolist() == probabilities.id.tolist() == template.id.tolist()
        values = probabilities[[f"probability_{c}" for c in classes]].to_numpy()
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-6)
        assert predictions.usage.tolist() == np.asarray(classes)[values.argmax(axis=1)].tolist()
        metrics = classification_metrics(labels, values, classes)
        np.testing.assert_allclose(metrics["accuracy"], accuracy_score(truth, predictions.usage))
        np.testing.assert_allclose(
            metrics["macro_f1"],
            f1_score(truth, predictions.usage, labels=classes, average="macro", zero_division=0),
        )
        comparison[f"{key}_prediction"] = predictions.usage
        comparison[f"{key}_correct"] = predictions.usage.eq(comparison.actual_usage)
        metrics["correct"] = int(comparison[f"{key}_correct"].sum())
        metrics["incorrect"] = len(comparison) - metrics["correct"]
        present_class_f1 = [row["f1"] for row in metrics["per_class"] if row["support"] > 0]
        metrics["macro_f1_classes_present"] = float(np.mean(present_class_f1))
        models[key] = metrics
        summary_rows.append(
            {
                "model": key,
                "description": name,
                "test_images": len(comparison),
                "correct": metrics["correct"],
                "incorrect": metrics["incorrect"],
                "accuracy": metrics["accuracy"],
                "macro_f1_nine_classes": metrics["macro_f1"],
                "macro_f1_eight_present_classes": metrics["macro_f1_classes_present"],
            }
        )
        matrix = np.asarray(metrics["confusion_matrix"])
        for index, row in enumerate(metrics["per_class"]):
            class_rows.append({"model": key, **row, "correct": int(matrix[index, index])})
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
            REPORT / f"{key}_confusion_matrix.csv", index_label="actual_usage"
        )
    assert models["Expanded"]["correct"] == old_metrics["correct"] == 5000
    np.testing.assert_allclose(
        models["Expanded"]["macro_f1"], old_metrics["macro_f1_all_nine_classes"]
    )

    paired = {}
    for source, target in (("E1", "E8"), ("E1", "Expanded"), ("E8", "Expanded")):
        before = comparison[f"{source}_correct"]
        after = comparison[f"{target}_correct"]
        fixed = int((~before & after).sum())
        broken = int((before & ~after).sum())
        paired[f"{source}_to_{target}"] = {
            "fixed_errors": fixed,
            "new_errors": broken,
            "both_correct": int((before & after).sum()),
            "both_wrong": int((~before & ~after).sum()),
            "net_additional_correct": fixed - broken,
            "accuracy_change_percentage_points": 100 * (fixed - broken) / len(comparison),
            "macro_f1_change": models[target]["macro_f1"] - models[source]["macro_f1"],
        }
        assert fixed - broken == models[target]["correct"] - models[source]["correct"]
    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(class_rows)
    summary.to_csv(REPORT / "test_summary.csv", index=False)
    per_class.to_csv(REPORT / "test_per_class.csv", index=False)
    comparison.to_csv(REPORT / "test_predictions_and_labels.csv", index=False)
    comparison.loc[comparison.E8_prediction.ne(comparison.Expanded_prediction)].to_csv(
        REPORT / "e8_vs_expanded_changed_predictions.csv", index=False
    )
    result = {
        "test_rows": len(comparison),
        "reference_matches": len(reference),
        "reference_path": str(REFERENCE.relative_to(ROOT)),
        "reference_sha256": compute_sha256(REFERENCE),
        "reference_labels_opened_at_utc": reference_opened_at,
        "new_predictions_frozen_at_utc": freeze["frozen_at_utc"],
        "historical_test_label_access": "Expanded-model test labels were already inspected earlier",
        "test_labels_used_for_model_selection_or_training": False,
        "selection": json.loads((REPORT / "selection_contract.json").read_text()),
        "models": models,
        "paired_changes": paired,
        "matching_csv_rows_with_extra_name_fields": extra_name_fields,
        "input_hashes": {
            str(path.relative_to(ROOT)): compute_sha256(path)
            for path in (
                Path(__file__),
                REPORT / "prediction_freeze.json",
                PRIOR / "prediction_freeze.json",
                PRIOR / "test_metrics.json",
                PRIOR / "usage_test_comparison.csv",
                ROOT / "src/fashion/train/metrics.py",
            )
        },
    }
    save_json(result, REPORT / "comparison_metrics.json")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1, 2.5]}
    )
    colours = ["#738393", "#2367a4", "#db8735"]
    display_names = ["E1: original baseline", "E8: original", "E8: added images"]
    top.barh(display_names, summary.accuracy * 100, color=colours, height=0.58)
    top.invert_yaxis()
    top.set_xlim(0, 105)
    top.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    for index, row in enumerate(summary.itertuples()):
        top.text(
            row.accuracy * 100 + 1,
            index,
            f"{row.accuracy:.2%}\n{row.correct:,}/{row.test_images:,}",
            va="center",
            fontsize=10,
        )
    top.set_title("Overall test accuracy", loc="left", weight="bold", pad=12)
    top.spines[["top", "right"]].set_visible(False)
    bottom.axis("off")
    table_rows = []
    for usage in classes:
        row = per_class.loc[per_class.class_name.eq(usage)].set_index("model")
        table_rows.append(
            [
                usage,
                *[
                    f"{int(row.loc[key, 'correct'])} / {int(row.loc[key, 'support'])}"
                    if int(row.loc[key, "support"])
                    else "No test images"
                    for key in NAMES
                ],
            ]
        )
    table_rows.append(["Nine-class macro-F1", *[f"{models[key]['macro_f1']:.4f}" for key in NAMES]])
    table = bottom.table(
        cellText=table_rows,
        colLabels=["Actual Usage", "E1 original", "E8 original", "E8 added images"],
        cellLoc="center",
        bbox=[0, 0.03, 1, 0.90],
        colWidths=[0.31, 0.23, 0.23, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e1e7")
        if row == 0:
            cell.set_facecolor("#20384b")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4f7")
        if column == 0 and row > 0:
            cell.set_text_props(ha="left")
    bottom.set_title("Correct predictions / actual images in each class", loc="left", weight="bold")
    fig.suptitle("Usage: original teacher data vs added images", fontsize=17, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.015,
        "Same 5,829 test images and reference labels. Each result averages five saved models.\n"
        "E1 chosen by validation accuracy; E8 chosen by nine-class validation macro-F1.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=2.0)
    fig.savefig(FIGURE, dpi=170, facecolor="white")
    plt.close(fig)
    verify_outputs(REPORT)
    verify_outputs(PRIOR)
    print(summary.to_string(index=False))
    print(json.dumps(paired, indent=2))
    print(per_class[["model", "class_name", "support", "correct", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()

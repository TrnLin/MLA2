"""Score the frozen 687-addition E8 test predictions against the existing reference."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

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
EARLIER = ROOT / "reports/task3/usage_teacher_vs_expanded_test_20260906"
FIRST_EXPANSION = ROOT / "reports/task3/usage_expanded_e8_test_20260906"
REFERENCE = (
    ROOT / "data/raw/external/fashion_product_images_v1_legacy_extras/fashion-dataset/styles.csv"
)
FIGURE = ROOT / "results/figures/task3/usage_expanded_v2_e8_test_20260907.png"


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_frozen(directory):
    freeze = json.loads((directory / "prediction_freeze.json").read_text())
    for name, digest in freeze["outputs"].items():
        assert compute_sha256(directory / name) == digest, name
    return freeze


def main():
    if (REPORT / "comparison_metrics.json").exists():
        raise RuntimeError("The comparison is already saved; preserve this evaluation.")
    freeze = verify_frozen(REPORT)
    assert not freeze["reference_labels_opened_in_this_inference"]
    verify_frozen(EARLIER)
    verify_frozen(FIRST_EXPANSION)
    recipe = json.loads((REPORT / "inference_recipe.json").read_text())
    classes = recipe["class_names"]
    previous_metrics = json.loads((FIRST_EXPANSION / "test_metrics.json").read_text())
    assert compute_sha256(REFERENCE) == previous_metrics["reference_sha256"]
    template = pd.read_csv(
        ROOT / "data/raw/teacher/test/styles_prediction.csv", usecols=["id"], keep_default_na=False
    )
    assert template.id.is_unique and len(template) == 5829
    ids = set(template.id)
    opened_at = datetime.now(timezone.utc).isoformat()
    assert opened_at > freeze["frozen_at_utc"]
    reference, extra_name_fields = {}, []
    with REFERENCE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert (
            len(header) == 10
            and header[0] == "id"
            and header[8] == "usage"
            and header[9] == "productDisplayName"
        )
        for row in reader:
            if not row or int(row[0]) not in ids:
                continue
            product_id = int(row[0])
            assert len(row) >= 10 and product_id not in reference
            assert row[8].strip() in classes
            reference[product_id] = row[8].strip()
            if len(row) > 10:
                extra_name_fields.append(product_id)
    assert set(reference) == ids
    comparison = template.copy()
    comparison["actual_usage"] = comparison.id.map(reference)
    old_comparison = pd.read_csv(EARLIER / "test_predictions_and_labels.csv", keep_default_na=False)
    assert old_comparison.id.tolist() == comparison.id.tolist()
    assert old_comparison.actual_usage.tolist() == comparison.actual_usage.tolist()
    labels = comparison.actual_usage.map({name: i for i, name in enumerate(classes)}).to_numpy()
    locations = {
        "E1": EARLIER / "E1",
        "E8": EARLIER / "E8",
        "E8_120": FIRST_EXPANSION,
        "E8_687": REPORT,
    }
    summaries, class_rows, models = [], [], {}
    for name, directory in locations.items():
        probabilities = pd.read_csv(
            directory / "usage_test_probabilities.csv", keep_default_na=False
        )
        predictions = pd.read_csv(directory / "usage_test_predictions.csv", keep_default_na=False)
        assert predictions.id.tolist() == probabilities.id.tolist() == template.id.tolist()
        values = probabilities[[f"probability_{c}" for c in classes]].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)
        assert predictions.usage.tolist() == np.asarray(classes)[values.argmax(axis=1)].tolist()
        metrics = classification_metrics(labels, values, classes)
        np.testing.assert_allclose(
            metrics["accuracy"], accuracy_score(comparison.actual_usage, predictions.usage)
        )
        np.testing.assert_allclose(
            metrics["macro_f1"],
            f1_score(
                comparison.actual_usage,
                predictions.usage,
                labels=classes,
                average="macro",
                zero_division=0,
            ),
        )
        comparison[name + "_prediction"] = predictions.usage
        comparison[name + "_correct"] = predictions.usage.eq(comparison.actual_usage)
        correct = int(comparison[name + "_correct"].sum())
        matrix = np.asarray(metrics["confusion_matrix"])
        assert np.trace(matrix) == correct and matrix.sum() == len(comparison)
        metrics["correct"] = correct
        metrics["macro_f1_present_classes"] = float(
            np.mean([r["f1"] for r in metrics["per_class"] if r["support"] > 0])
        )
        models[name] = metrics
        summaries.append(
            dict(
                model=name,
                test_rows=len(comparison),
                correct=correct,
                incorrect=len(comparison) - correct,
                accuracy=metrics["accuracy"],
                macro_f1_nine_classes=metrics["macro_f1"],
                macro_f1_present_classes=metrics["macro_f1_present_classes"],
            )
        )
        for index, item in enumerate(metrics["per_class"]):
            class_rows.append(dict(model=name, **item, correct=int(matrix[index, index])))
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
            REPORT / f"{name}_confusion_matrix.csv", index_label="actual_usage"
        )
    assert (
        models["E1"]["correct"] == 5123
        and models["E8"]["correct"] == 4989
        and models["E8_120"]["correct"] == 5000
    )
    changes = {}
    for baseline in ("E1", "E8", "E8_120"):
        before, after = comparison[baseline + "_correct"], comparison.E8_687_correct
        fixed, broken = int((~before & after).sum()), int((before & ~after).sum())
        changes[baseline] = dict(
            fixed_errors=fixed,
            new_errors=broken,
            net_additional_correct=fixed - broken,
            accuracy_change_percentage_points=100 * (fixed - broken) / len(comparison),
            macro_f1_change=models["E8_687"]["macro_f1"] - models[baseline]["macro_f1"],
        )
        assert fixed - broken == models["E8_687"]["correct"] - models[baseline]["correct"]
    summary, per_class = pd.DataFrame(summaries), pd.DataFrame(class_rows)
    summary.to_csv(REPORT / "test_summary.csv", index=False)
    per_class.to_csv(REPORT / "test_per_class.csv", index=False)
    comparison.to_csv(REPORT / "test_predictions_and_labels.csv", index=False)
    comparison.loc[~comparison.E8_687_correct].to_csv(REPORT / "usage_test_errors.csv", index=False)
    details = dict(
        reference_labels_opened_at_utc=opened_at,
        reference_sha256=compute_sha256(REFERENCE),
        prediction_freeze_sha256=compute_sha256(REPORT / "prediction_freeze.json"),
        test_rows=len(comparison),
        matched_reference_ids=len(reference),
        missing_reference_ids=[],
        duplicate_reference_ids=[],
        extra_product_name_fields=extra_name_fields,
        test_labels_previously_seen=True,
        training_tuning_or_selection_performed=False,
        comparison_is_new_blind_test=False,
        models=models,
        changes_against_previous_models=changes,
    )
    save_json(details, REPORT / "comparison_metrics.json")

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.45]}
    )
    names = ["E1\nteacher only", "E8\nteacher only", "E8\n+120 images", "E8\n+687 images"]
    bars = axes[0].bar(
        np.arange(4), 100 * summary.accuracy, color=["#57758f", "#95abbc", "#b5c7d0", "#168475"]
    )
    axes[0].bar_label(
        bars,
        labels=[f"{a:.2%}\n{n:,} correct" for a, n in zip(summary.accuracy, summary.correct)],
        padding=5,
        fontsize=9,
    )
    axes[0].set(
        xticks=np.arange(4),
        xticklabels=names,
        ylim=(0, 100),
        ylabel="Test accuracy (%)",
        title="Same 5,829 official test images",
    )
    x = np.arange(len(classes))
    for offset, model, color in ((-0.19, "E1", "#57758f"), (0.19, "E8_687", "#168475")):
        frame = per_class.loc[per_class.model.eq(model)].set_index("class_name").loc[classes]
        axes[1].bar(
            x + offset,
            100 * frame.recall,
            0.36,
            color=color,
            label="E1" if model == "E1" else "E8 +687",
        )
    support = (
        per_class.loc[per_class.model.eq("E8_687")].set_index("class_name").loc[classes, "support"]
    )
    axes[1].set(
        xticks=x,
        xticklabels=[f"{name}\n(n={n:,})" for name, n in zip(classes, support)],
        ylim=(0, 105),
        ylabel="Correct within each class (%)",
        title="Per-class recall; Home has no test examples",
    )
    axes[1].tick_params(axis="x", labelrotation=40, labelsize=9)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("E8 trained with 687 added images: official test comparison", fontsize=15)
    fig.savefig(FIGURE, dpi=160, facecolor="white")
    plt.close(fig)
    for directory in (REPORT, EARLIER, FIRST_EXPANSION):
        verify_frozen(directory)
    for path, digest in recipe["inputs"].items():
        assert compute_sha256(resolve_task3_path(path, root=ROOT)) == digest
    print(summary.to_string(index=False))
    print(
        per_class.loc[
            per_class.model.eq("E8_687"),
            ["class_name", "support", "correct", "precision", "recall", "f1"],
        ].to_string(index=False)
    )
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    main()

"""Check saved SAM25 receipts, rules and comparisons without training."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.train.task3_gender_sam25 import verify_sam25_evidence
from fashion.train.task3_gender_stronger_mixup import apply_improvement_rules

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "task3_gender_sam_result_20260906"


def read(path):
    return json.loads(path.read_text())


decision = read(HERE / "screen_decision.json")
comparison = read(HERE / "incremental_comparison.json")
assert decision["incremental_comparison"] == comparison
original = dict(
    decision,
    checks=[c for c in decision["checks"] if not c["gate"].startswith("vs_mixup20.")]
    + decision["diagnostic_checks"],
    status=decision["historical_screen_status"],
)
recomputed = apply_improvement_rules(original)
assert recomputed["checks"] == decision["checks"]
assert recomputed["status"] == decision["status"] == "pass"
for check in decision["checks"]:
    value, rule = check["value"], check["rule"]
    if rule == "0 < value < 3000000000":
        passed = 0 < value < 3_000_000_000
    elif rule.startswith(">= "):
        passed = value + 1e-12 >= float(rule.split()[1])
    elif rule.startswith("<= "):
        passed = value <= float(rule.split()[1]) + 1e-12
    elif rule.startswith("> "):
        passed = value > float(rule.split()[1])
    elif rule.startswith("== "):
        passed = value == float(rule.split()[1])
    else:
        raise AssertionError(rule)
    assert passed and check["status"] == "pass"
splits = load_gender_name_truth_variant()
prefix = {}
fold_metrics = []
for fold in (0, 4):
    directory = HERE / f"fold{fold}"
    run = {"config": read(directory / "config.json"), "metrics": read(directory / "metrics.json")}
    verify_sam25_evidence(run, fold=fold, splits=splits, directory=directory)
    assert run["metrics"]["run_id"] == decision["run_ids"][str(fold)]
    history = pd.read_csv(directory / "history.csv")
    old_history = pd.read_csv(OLD / f"fold{fold}/history.csv").iloc[:25]
    columns = [
        "learning_rate",
        "train_loss",
        "validation_loss",
        "validation_macro_f1",
        "sam_second_loss",
    ]
    prefix[str(fold)] = {c: bool(np.array_equal(history[c], old_history[c])) for c in columns}
    assert all(prefix[str(fold)].values())
    fold_metrics.append(run["metrics"])
cm = sum(np.asarray(m["confusion_matrix"]) for m in fold_metrics)
assert np.array_equal(cm, comparison["candidate"]["confusion_matrix"])
for group in ("candidate", "dropout"):
    matrix = np.asarray(comparison[group]["confusion_matrix"])
    assert np.isclose(
        np.mean(2 * np.diag(matrix) / (matrix.sum(0) + matrix.sum(1))),
        comparison[group]["macro_f1"],
    )
old = read(OLD / "verified_summary.json")
rows = []
for label, metrics, gap, train in (
    (
        "MixUp 0.2",
        comparison["dropout"],
        np.mean([f["dropout_gap"] for f in comparison["folds"]]),
        np.mean([f["dropout_train_f1"] for f in comparison["folds"]]),
    ),
    (
        "SAM 30",
        read(OLD / "incremental.json")["candidate"],
        old["mean_sam_gap"],
        old["mean_sam_training_f1"],
    ),
    (
        "SAM 25",
        comparison["candidate"],
        np.mean([f["candidate_gap"] for f in comparison["folds"]]),
        np.mean([f["candidate_train_f1"] for f in comparison["folds"]]),
    ),
):
    matrix = np.asarray(metrics["confusion_matrix"])
    rows.append(
        dict(
            model=label,
            validation_f1=metrics["macro_f1"],
            mean_gap=float(gap),
            mean_clean_train_f1=float(train),
            unisex_correct=int(matrix[3, 3]),
            unisex_support=int(matrix[3].sum()),
            errors=int(matrix.sum() - np.trace(matrix)),
            nll=metrics["nll"],
            ece_15=metrics["ece_15"],
        )
    )
summary = dict(
    status=decision["status"],
    passed_checks=sum(c["status"] == "pass" for c in decision["checks"]),
    total_checks=len(decision["checks"]),
    comparisons=rows,
    mean_gap_reduction=float(np.mean([f["gap_reduction"] for f in comparison["folds"]])),
    class_f1_delta=comparison["class_f1_delta"],
    validation_interval=comparison["validation_interval"],
    corruption_delta=comparison["mean_induced_change_delta"],
    prefix_matches=prefix,
    training_minutes=sum(f["train_seconds"] for f in decision["folds"]) / 60,
    peak_gpu_memory_bytes=max(f["peak_memory_bytes"] for f in decision["folds"]),
    limits=(
        "Saved evidence checked. No checkpoint download, independent model evaluation "
        "or full registry audit. Two selected folds; not independent test evidence."
    ),
)
(HERE / "verified_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
pd.DataFrame(rows).to_csv(HERE / "comparison.csv", index=False)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
x = np.arange(3)
colors = ["#667780", "#c18449", "#377e91"]
for ax, field, scale, title, ylim in zip(
    axes,
    ("validation_f1", "mean_gap", "unisex_correct"),
    (100, 100, 1),
    ("Validation F1 (%)", "Train–validation gap (points)", "Unisex found correctly / 707"),
    ((0, 100), (0, 16), (0, 420)),
    strict=True,
):
    values = [r[field] * scale for r in rows]
    bars = ax.bar(x, values, color=colors, width=0.6)
    ax.bar_label(
        bars, labels=[f"{v:.2f}" if scale == 100 else str(int(v)) for v in values], padding=4
    )
    ax.set(title=title, ylim=ylim, xticks=x, xticklabels=[r["model"] for r in rows])
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
fig.suptitle("Fixed epoch-25 SAM: all 19 screen checks pass", fontsize=16)
fig.text(
    0.5,
    0.01,
    "Same corrected labels and folds 0 and 4. Matched IEEE evaluation. "
    "Selected validation evidence.",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
fig.savefig(HERE / "comparison.png", dpi=170)
print(json.dumps(summary, indent=2))

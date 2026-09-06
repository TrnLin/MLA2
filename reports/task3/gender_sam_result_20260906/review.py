"""Verify and summarize the saved two-fold SAM screen without training."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.train.task3_gender_sam import verify_sam_evidence
from fashion.train.task3_gender_stronger_mixup import apply_improvement_rules

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def pooled(rows):
    cm = sum(np.asarray(r["validation"]["confusion_matrix"]) for r in rows)
    f1 = np.mean(2 * np.diag(cm) / (cm.sum(0) + cm.sum(1)))
    return float(f1), int(cm[3, 3]), int(cm[3].sum())


decision, comparison = read("decision.json"), read("incremental.json")
assert decision["incremental_comparison"] == comparison
original = dict(
    decision,
    checks=[c for c in decision["checks"] if not c["gate"].startswith("vs_mixup20.")]
    + decision["diagnostic_checks"],
    status=decision["historical_screen_status"],
)
recomputed = apply_improvement_rules(original)
assert recomputed["checks"] == decision["checks"]
assert recomputed["status"] == decision["status"]
for group in ("candidate", "dropout"):
    cm = np.asarray(comparison[group]["confusion_matrix"])
    assert np.isclose(
        np.mean(2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))),
        comparison[group]["macro_f1"],
        rtol=0,
        atol=1e-12,
    )
splits = load_gender_name_truth_variant()
for fold in (0, 4):
    run = {"config": read(f"fold{fold}/config.json"), "metrics": read(f"metrics{fold}.json")}
    verify_sam_evidence(run, fold=fold, splits=splits, directory=HERE / f"fold{fold}")
diagnostics = {f: read(f"diag{f}.json") for f in (0, 4)}
epochs = []
for i, epoch in enumerate((10, 15, 20, 25, 30)):
    rows = [diagnostics[f][i] for f in (0, 4)]
    assert all(r["epoch"] == epoch and r["checkpoint_selection"] == "diagnostic_only" for r in rows)
    for row in rows:
        assert np.isclose(
            row["gap"], row["clean_training"]["macro_f1"] - row["validation"]["macro_f1"]
        )
    f1, correct, support = pooled(rows)
    epochs.append(
        dict(
            epoch=epoch,
            mean_train_f1=np.mean([r["clean_training"]["macro_f1"] for r in rows]),
            mean_validation_f1=np.mean([r["validation"]["macro_f1"] for r in rows]),
            mean_gap=np.mean([r["gap"] for r in rows]),
            pooled_validation_f1=f1,
            unisex_correct=correct,
            unisex_support=support,
            unisex_recall=correct / support,
        )
    )
frame = pd.DataFrame(epochs)
frame.to_csv(HERE / "epoch_summary.csv", index=False)
folds = comparison["folds"]
summary = {
    "status": decision["status"],
    "passed_checks": sum(c["status"] == "pass" for c in decision["checks"]),
    "total_checks": len(decision["checks"]),
    "failed_checks": [c for c in decision["checks"] if c["status"] != "pass"],
    "mean_parent_gap": float(np.mean([r["dropout_gap"] for r in folds])),
    "mean_sam_gap": float(np.mean([r["candidate_gap"] for r in folds])),
    "gap_reduction": float(np.mean([r["gap_reduction"] for r in folds])),
    "mean_parent_training_f1": float(np.mean([r["dropout_train_f1"] for r in folds])),
    "mean_sam_training_f1": float(np.mean([r["candidate_train_f1"] for r in folds])),
    "parent_pooled_validation_f1": comparison["dropout"]["macro_f1"],
    "sam_pooled_validation_f1": comparison["candidate"]["macro_f1"],
    "validation_f1_delta": comparison["validation_delta"],
    "validation_f1_interval": comparison["validation_interval"],
    "class_f1_delta": comparison["class_f1_delta"],
    "epoch25_diagnostic": epochs[3],
    "sum_training_seconds": sum(r["train_seconds"] for r in decision["folds"]),
    "peak_gpu_memory_bytes": max(r["peak_memory_bytes"] for r in decision["folds"]),
    "verification": [
        "Recomputed all 19 screen checks with the saved rule implementation.",
        "Recomputed parent and candidate pooled F1 from their confusion matrices.",
        "Verified both folds' SAM/MixUp receipts, diagnostic hashes, and training-row contracts.",
        "No independent checkpoint download, model re-evaluation, or full registry audit.",
    ],
    "checkpoint25_limit": "Diagnostics only; no saved weights or full IEEE/corruption checks.",
}
(HERE / "verified_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
x = frame.epoch
axes[0].plot(x, 100 * frame.mean_train_f1, "o-", color="#277da1", label="Clean training")
axes[0].plot(x, 100 * frame.mean_validation_f1, "o-", color="#e07a35", label="Validation")
axes[0].set(title="Mean fold F1", ylabel="F1 (%)", ylim=(70, 95))
axes[0].legend(frameon=False, loc="upper left")
axes[1].plot(x, 100 * frame.mean_gap, "o-", color="#277da1")
limit = 100 * (summary["mean_parent_gap"] - 0.02)
axes[1].axhline(limit, color="#a74450", ls="--", label=f"Gap ceiling {limit:.2f}")
axes[1].set(title="Mean clean gap", ylabel="Percentage points", ylim=(0, 14))
axes[1].legend(frameon=False, loc="lower right")
axes[2].plot(x, frame.unisex_correct, "o-", color="#7159a6")
axes[2].axhline(361, color="#a74450", ls="--", label="Parent: 361 / 707")
axes[2].annotate("361", (25, 361), xytext=(23, 376), fontsize=10)
axes[2].annotate("356", (30, 356), xytext=(28, 340), fontsize=10)
axes[2].set(title="Unisex items found correctly", ylabel="Count out of 707", ylim=(235, 400))
axes[2].legend(frameon=False, loc="lower right")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.18)
    ax.axvline(25, color="#888888", lw=1, ls=":", alpha=0.6)
fig.suptitle("SAM: five saved diagnostic checkpoints", fontsize=16, x=0.5, y=0.98)
fig.text(
    0.5,
    0.01,
    "Training-runtime diagnostics. Epoch 30 is the fixed final model; "
    "epoch 25 has not had the full evaluation.",
    ha="center",
    fontsize=10,
    color="#555555",
)
fig.tight_layout(rect=(0, 0.07, 1, 0.92))
fig.savefig(HERE / "epoch_diagnostics.png", dpi=170)
print(json.dumps(summary, indent=2))

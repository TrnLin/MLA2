"""Render the measured class changes, saved learning curves and fixed BN probe."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
HERE = Path(__file__).resolve().parent
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
old = pd.read_csv(HERE / "mixup20_class_gaps.csv").set_index("class_name").loc[CLASSES]
new = pd.read_csv(HERE / "mixup40_class_gaps.csv").set_index("class_name").loc[CLASSES]
history = pd.read_csv(HERE / "notebook_epoch_history.csv")
bn = pd.read_csv(HERE / "bn_probe_scores.csv")
pooled = json.loads((HERE / "bn_probe_pooled.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
fig.suptitle("Find a different way to reduce the Gender gap", fontsize=17, weight="bold")
x = np.arange(5)
ax = axes[0, 0]
for offset, frame, label, color in [
    (-0.19, old, "MixUp 0.2", "#177e89"),
    (0.19, new, "MixUp 0.4", "#dc8150"),
]:
    bars = ax.bar(x + offset, frame.gap * 100, 0.36, label=label, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
ax.set(xticks=x, xticklabels=CLASSES, ylabel="Clean F1 gap (percentage points)", ylim=(0, 27))
ax.set_title("Stronger mixing worsened the two child-class gaps", pad=12)
ax.legend(frameon=False)

ax = axes[0, 1]
delta = (new - old) * 100
for offset, column, label, color in [
    (-0.19, "train_f1", "Clean training F1", "#6c7a89"),
    (0.19, "validation_f1", "Validation F1", "#a94939"),
]:
    bars = ax.bar(x + offset, delta[column], 0.36, label=label, color=color)
    ax.bar_label(bars, fmt="%+.1f", padding=3, fontsize=9)
ax.axhline(0, color="#888888", linewidth=0.8)
ax.set(xticks=x, xticklabels=CLASSES, ylabel="0.4 minus 0.2 (percentage points)", ylim=(-4.7, 1.4))
ax.set_title("Unisex: less training fit, little validation gain", pad=12)
ax.legend(loc="lower right", frameon=False)

ax = axes[1, 0]
for fold, color in [(0, "#177e89"), (4, "#755391")]:
    h = history.loc[history.alpha.eq(0.2) & history.fold.eq(fold)]
    ax.plot(h.epoch, h.validation_macro_f1 * 100, label=f"Fold {fold}", color=color, linewidth=2)
ax.axhline(74, color="#888888", linestyle="--", label="74% reference")
ax.axvline(20, color="#bbbbbb", linestyle=":")
ax.set(
    xlabel="Training epoch", ylabel="Logged validation macro-F1 (%)", ylim=(45, 85), xlim=(1, 30)
)
ax.set_title("MixUp 0.2 validation F1 still improved late", pad=12)
ax.legend(loc="lower right", frameon=False)
ax.text(
    0.03,
    0.04,
    "No clean training F1 was saved at earlier epochs.\n"
    "Logged scores use training-runtime precision.",
    transform=ax.transAxes,
    fontsize=9,
)

ax = axes[1, 1]
means = bn.groupby(["model", "scope"]).macro_f1.mean().unstack()
labels = ["Original 0.2", "Clean BN probe"]
keys = ["original_cpu", "clean_bn_cpu"]
metrics = [
    ("Train F1", [means.loc[k, "train"] * 100 for k in keys]),
    ("Validation F1", [pooled[k]["macro_f1"] * 100 for k in keys]),
    ("Clean gap", [(means.loc[k, "train"] - means.loc[k, "validation"]) * 100 for k in keys]),
]
x = np.arange(3)
for j, (label, color) in enumerate(zip(labels, ["#177e89", "#8c9096"])):
    values = [v[j] for _, v in metrics]
    bars = ax.bar(x + (j - 0.5) * 0.36, values, 0.34, label=label, color=color)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.set(
    xticks=x,
    xticklabels=[m for m, _ in metrics],
    ylabel="Percent / percentage points",
    ylim=(0, 107),
)
ax.set_title("Training-only BN recalibration did not reduce the gap", pad=12)
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.supxlabel(
    "Folds 0 and 4; same corrected labels. Gaps and class scores are fold means; "
    "overall validation F1 is pooled.",
    fontsize=10,
)
destination = ROOT / "results/figures/task3/gender_overfitting_research_20260906.png"
fig.savefig(destination, dpi=160)
fig.savefig(HERE / "findings.png", dpi=160)
print(destination)

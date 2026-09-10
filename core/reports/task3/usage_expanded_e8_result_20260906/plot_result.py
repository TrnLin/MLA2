"""Plot the verified teacher-only comparison for the report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fashion.config import ROOT

REPORT = Path(__file__).resolve().parent
summary = json.loads((REPORT / "verified_summary.json").read_text())
figure_path = ROOT / "results/figures/task3/usage_expanded_e8_comparison_20260906.png"
figure_path.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, (overall, rare) = plt.subplots(
    1, 2, figsize=(12.8, 6.6), gridspec_kw={"width_ratios": [0.95, 1.3]}
)
fig.subplots_adjust(left=0.12, right=0.98, top=0.76, bottom=0.22, wspace=0.47)
fig.suptitle(
    "Added images did not improve the teacher-only score",
    x=0.04,
    y=0.965,
    ha="left",
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.04,
    0.90,
    "All five folds completed · 30 epochs each · same 32,772 teacher validation images",
    color="#52606D",
    fontsize=11,
)
colors = ["#2E5266", "#D96C3B"]
scores = [summary["e8_teacher_macro_f1"], summary["expanded_teacher_macro_f1"]]
overall.barh([1, 0], scores, height=0.43, color=colors)
overall.set_yticks([1, 0], ["Original E8", "Added images"])
overall.set_xlim(0, 0.56)
overall.set_ylim(-0.8, 1.8)
overall.set_xlabel("Macro-F1 (equal weight for each of 9 classes)", fontsize=10)
overall.set_title("Fair overall comparison", loc="left", pad=17, fontweight="bold")
for y, score in zip([1, 0], scores, strict=True):
    overall.text(score + 0.01, y, f"{score:.4f}", va="center", fontweight="bold")
overall.text(
    0.02,
    -0.20,
    f"Change: {summary['teacher_macro_f1_change']:+.4f} · lower in 4/5 folds",
    transform=overall.transAxes,
    fontsize=10,
    color="#A34323",
    linespacing=1.6,
)
selected = [
    row
    for row in summary["class_changes"]
    if row["class_name"] in {"Home", "NA", "Party", "Smart Casual", "Travel"}
]
positions = np.arange(len(selected))
for offset, key, color, label in (
    (-0.17, "e8_f1", colors[0], "Original E8"),
    (0.17, "expanded_f1", colors[1], "Added images"),
):
    rare.barh(
        positions + offset,
        [row[key] for row in selected],
        height=0.29,
        color=color,
        label=label,
    )
    for position, row in zip(positions + offset, selected, strict=True):
        rare.text(row[key] + 0.006, position, f"{row[key]:.3f}", va="center", fontsize=9)
rare.set_yticks(positions, [f"{row['class_name']}  (n={row['support']})" for row in selected])
rare.invert_yaxis()
rare.set_xlim(0, 0.29)
rare.set_xlabel("Class F1 on teacher images")
rare.set_title("Rare classes: no gain for Party or Smart Casual", loc="left", pad=17, fontsize=11)
rare.legend(loc="upper center", bbox_to_anchor=(0.48, -0.16), ncol=2, frameon=False, fontsize=10)
for axis in (overall, rare):
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E2E7EB", linewidth=0.7)
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
fig.text(
    0.04,
    0.04,
    "The combined score (0.5548) includes added-source images and cannot replace this comparison.\n"
    "One seed; very small rare-class samples. "
    "This result alone does not establish statistical significance.",
    color="#52606D",
    fontsize=10,
    linespacing=1.5,
)
fig.savefig(figure_path, dpi=180, facecolor="white")
plt.close(fig)
print(figure_path)

"""Plot the verified teacher comparison and source-specific rare-class recall."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from fashion.config import ROOT

REPORT = Path(__file__).resolve().parent


def main():
    summary = json.loads((REPORT / "verified_summary.json").read_text())
    classes = pd.read_csv(REPORT / "source_class_scores.csv", keep_default_na=False)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.2, top=0.75, wspace=0.43)
    labels = ["Original E8", "E8 + 120 images", "E8 + 687 images"]
    scores = [
        summary["teacher_reference_scores"]["teacher_e8"]["macro_f1"],
        summary["teacher_reference_scores"]["previous_expansion"]["macro_f1"],
        summary["teacher_macro_f1"],
    ]
    bars = axes[0].barh(labels, scores, color=["#334155", "#aab3c0", "#0e7490"], height=0.56)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.5)
    axes[0].set_xlabel("Nine-class macro-F1")
    axes[0].set_title("Same 32,772 teacher validation images", loc="left", fontsize=12)
    axes[0].bar_label(bars, labels=[f"{x:.4f}" for x in scores], padding=5, fontsize=11)
    axes[0].grid(axis="x", alpha=0.15)
    axes[0].set_axisbelow(True)

    rare = ["Home", "Party", "Smart Casual", "Travel"]
    x = np.arange(len(rare))
    for cohort, offset, label, color in (
        ("teacher", -0.19, "Teacher", "#334155"),
        ("new_added", 0.19, "New 567 images", "#0e7490"),
    ):
        frame = classes.loc[classes.cohort.eq(cohort)].set_index("class_name").loc[rare]
        recall = frame.correct / frame.support
        bars = axes[1].bar(x + offset, recall, width=0.36, label=label, color=color)
        axes[1].bar_label(
            bars,
            labels=[f"{row.correct}/{row.support}" for row in frame.itertuples()],
            padding=3,
            fontsize=8,
        )
    axes[1].set_xticks(x, ["Home", "Party", "Smart\nCasual", "Travel"])
    axes[1].set_ylim(0, 1.19)
    axes[1].set_yticks(np.arange(0, 1.01, 0.25))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Correct / images in that class")
    axes[1].set_title("Rare-class recall for the new model", loc="left", fontsize=12)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    axes[1].grid(axis="y", alpha=0.15)
    axes[1].set_axisbelow(True)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Usage E8 with 687 added images", fontsize=18, y=0.95, fontweight="bold")
    figure.text(
        0.5,
        0.85,
        "Teacher macro-F1 fell by 0.0101; "
        "the mixed-data score of 0.7567 uses a different class mix.",
        ha="center",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.06,
        "Five saved folds, 30 epochs, one seed. Source comparisons are diagnostic. "
        "Teacher rare-class samples are small.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    destination = ROOT / "results/figures/task3/usage_expanded_v2_e8_comparison_20260906.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()

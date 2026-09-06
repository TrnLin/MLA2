"""Render the saved holdout scores without reopening any source labels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fashion.config import ROOT

REPORT = Path(__file__).resolve().parent
FIGURE = ROOT / "results/figures/task3/usage_holdout_20260906.png"
MODELS = ("E1", "E8", "Expanded")


def main():
    metrics = json.loads((REPORT / "holdout_metrics.json").read_text())
    summary = pd.read_csv(REPORT / "holdout_summary.csv", keep_default_na=False)
    per_class = pd.read_csv(REPORT / "holdout_per_class.csv", keep_default_na=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1, 2.5]}
    )
    names = ["E1: original baseline", "E8: original", "E8: added images"]
    top.barh(names, summary.accuracy * 100, color=["#738393", "#2367a4", "#db8735"], height=0.58)
    top.invert_yaxis()
    top.set_xlim(0, 107)
    top.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    for index, row in enumerate(summary.itertuples()):
        top.text(
            row.accuracy * 100 + 1,
            index,
            f"{row.accuracy:.2%}\n{row.correct:,}/{row.scored_images:,}",
            va="center",
            fontsize=10,
        )
    top.set_title("Holdout accuracy", loc="left", weight="bold", pad=12)
    top.spines[["top", "right"]].set_visible(False)
    bottom.axis("off")
    rows = []
    for usage in metrics["class_names"]:
        selected = per_class.loc[per_class.class_name.eq(usage)].set_index("model")
        rows.append(
            [
                usage,
                *[
                    f"{int(selected.loc[key, 'correct'])} / {int(selected.loc[key, 'support'])}"
                    if int(selected.loc[key, "support"])
                    else "No holdout images"
                    for key in MODELS
                ],
            ]
        )
    rows.append(
        ["Nine-class macro-F1", *[f"{metrics['models'][key]['macro_f1']:.4f}" for key in MODELS]]
    )
    table = bottom.table(
        cellText=rows,
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
    fig.suptitle("Usage: holdout results", fontsize=18, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.015,
        f"Same {metrics['scored_rows']:,} holdout images for all three models. "
        "Each result averages five saved models.\n"
        "Predictions were saved before holdout Usage labels were opened. No fitting or tuning.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=2.0)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170, facecolor="white")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()

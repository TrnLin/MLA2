"""Figures for the one-shot Task 4 holdout evaluation."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402


def _save_figure(figure: Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def build_holdout_scorecard_figure(scorecard: pd.DataFrame, path: str | Path) -> Path:
    labels = scorecard["method"].astype(str) + "\n" + scorecard["direction"].astype(str)
    values = scorecard["ndcg_at_10_query_mean"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(np.arange(len(values)), values, color="#457b9d")
    axis.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Protocol A query-mean nDCG@10")
    axis.set_title("Frozen holdout scorecard")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def build_holdout_bootstrap_figure(intervals: pd.DataFrame, path: str | Path) -> Path:
    ordered = intervals.reset_index(drop=True)
    y = np.arange(len(ordered))
    medians = ordered["median"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            medians - ordered["lower_95"].to_numpy(dtype=float),
            ordered["upper_95"].to_numpy(dtype=float) - medians,
        ]
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.errorbar(medians, y, xerr=errors, fmt="o", capsize=4, color="#264653")
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, ordered["metric"])
    axis.set_xlabel("Family-blocked bootstrap estimate (95% interval)")
    axis.set_title("R5 holdout uncertainty and paired effects")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def build_holdout_selective_retrieval_figure(
    selective: pd.DataFrame,
    path: str | Path,
) -> Path:
    ordered = selective.sort_values("actual_coverage")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        ordered["actual_coverage"],
        ordered["selective_ndcg_at_10"],
        marker="o",
        label="nDCG@10",
    )
    axis.plot(
        ordered["actual_coverage"],
        ordered["selective_recall_at_10"],
        marker="s",
        label="family recall@10",
    )
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Retained query coverage")
    axis.set_ylabel("Mean score")
    axis.set_title("R5 selective retrieval by rank-1 distance")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def build_holdout_slices_robustness_figure(
    slices: pd.DataFrame,
    robustness: pd.DataFrame,
    path: str | Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(slices["slice"], slices["value"], color="#6d597a")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Slice score")
    axes[0].set_title("R5 teacher-clean failure slices")
    axes[0].grid(axis="x", alpha=0.2)

    for method, rows in robustness.groupby("method", sort=False):
        axes[1].plot(
            rows["condition"],
            rows["ndcg_at_10"],
            marker="o",
            label=str(method),
        )
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("Protocol A nDCG@10")
    axes[1].set_title("Frozen query-condition robustness")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def build_holdout_error_examples_figure(
    examples: pd.DataFrame,
    *,
    project_root: str | Path,
    path: str | Path,
) -> Path:
    root = Path(project_root)
    shown = examples.head(6)
    rows = max(1, len(shown))
    figure, axes = plt.subplots(rows, 2, figsize=(7, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for row_index, row in enumerate(shown.itertuples(index=False)):
        for column, (path_field, title) in enumerate(
            (
                ("query_path", f"Query {row.query_id}: {row.query_articleType}"),
                (
                    "candidate_path",
                    f"Top-1 {row.candidate_id}: {row.candidate_articleType}",
                ),
            )
        ):
            image_path = (root / str(getattr(row, path_field))).resolve()
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                axes[row_index, column].imshow(image)
            axes[row_index, column].set_title(title, fontsize=9)
            axes[row_index, column].axis("off")
    figure.suptitle("Deterministic lowest-scoring R5 holdout examples", fontsize=12)
    figure.tight_layout()
    return _save_figure(figure, path)


def build_holdout_source_robustness_figure(
    source_robustness: pd.DataFrame,
    path: str | Path,
) -> Path:
    measured = source_robustness.loc[
        source_robustness["holdout_source_robustness_ratio"].notna()
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        measured["method"],
        measured["holdout_source_robustness_ratio"],
        color="#2a9d8f",
    )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_ylabel("V1→teacher / teacher→teacher nDCG@10")
    axis.set_title("Frozen holdout source robustness")
    axis.tick_params(axis="x", rotation=35)
    finite = measured["holdout_source_robustness_ratio"].to_numpy(dtype=float)
    if len(finite):
        lower = max(0.0, float(np.nanmin(finite)) - 0.1)
        upper = max(1.05, float(np.nanmax(finite)) + 0.1)
        if math.isfinite(lower) and math.isfinite(upper):
            axis.set_ylim(lower, upper)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


__all__ = [
    "build_holdout_bootstrap_figure",
    "build_holdout_error_examples_figure",
    "build_holdout_scorecard_figure",
    "build_holdout_selective_retrieval_figure",
    "build_holdout_slices_robustness_figure",
    "build_holdout_source_robustness_figure",
]

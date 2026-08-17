"""Matplotlib evidence figures for the reproducible fashion EDA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image


MAX_REVIEW_EXAMPLES = 24
COMPARABLE_IMAGE_METRICS = ("brightness", "contrast", "colorfulness", "saturation")
TARGET_HEAD_SIZE = 15
ARTICLE_HEAD_SIZE = 20


def _new_figure(rows: int = 1, columns: int = 1, *, figsize: tuple[float, float] = (9, 5)):
    return plt.subplots(rows, columns, figsize=figsize, squeeze=False)


def _readable_labels(values: pd.Series) -> pd.Series:
    return values.astype(str).replace("", "<blank>")


def _horizontal_distribution(
    axis,
    table: pd.DataFrame,
    *,
    title: str,
    log_scale: bool = False,
) -> None:
    ordered = table.sort_values(["count", "label"], ascending=[True, True]).copy()
    labels = _readable_labels(ordered["label"])
    bars = axis.barh(labels, ordered["count"], color="#4178a8")
    shares = (
        ordered["share"].to_numpy(dtype=float)
        if "share" in ordered
        else ordered["count"].to_numpy(dtype=float) / max(float(ordered["count"].sum()), 1.0)
    )
    axis.bar_label(
        bars,
        labels=[
            f"{int(count):,} ({share:.1%})"
            for count, share in zip(ordered["count"], shares)
        ],
        padding=3,
        fontsize=8,
    )
    if log_scale:
        axis.set_xscale("log")
    else:
        axis.margins(x=0.22)
    axis.set(title=title, xlabel="Products" + (" (log scale)" if log_scale else ""))
    axis.grid(axis="x", alpha=0.2)


def plot_target_distributions(distributions: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot readable target heads while leaving the full long tail to its own figure."""
    names = list(distributions)
    visible_counts = [
        min(len(distributions[name]), TARGET_HEAD_SIZE)
        if name == "articleType"
        else len(distributions[name])
        for name in names
    ]
    height_ratios = [max(2.4, 0.34 * count + 0.8) for count in visible_counts]
    figure, axes = plt.subplots(
        max(1, len(names)),
        1,
        figsize=(12, sum(height_ratios)),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    for axis, name in zip(axes.ravel(), names):
        complete = distributions[name].sort_values(
            ["count", "label"], ascending=[False, True], ignore_index=True
        )
        if name == "articleType":
            visible = complete.head(TARGET_HEAD_SIZE)
            title = (
                f"articleType — Top {len(visible)} of {len(complete)} classes"
            )
        else:
            visible = complete
            title = f"{name} — all {len(complete)} classes"
        positive = visible.loc[visible["count"].gt(0), "count"]
        imbalance = (
            float(positive.max() / positive.min()) if not positive.empty else 1.0
        )
        _horizontal_distribution(
            axis,
            visible,
            title=title,
            log_scale=name != "articleType" and imbalance >= 100,
        )
    for axis in axes.ravel()[len(names):]:
        axis.remove()
    figure.tight_layout()
    return figure


def plot_article_type_support(distribution: pd.DataFrame, support: pd.DataFrame) -> Figure:
    """Name the learnable head and summarize every class in tail-safe views."""
    figure, axes = _new_figure(2, 2, figsize=(15, 10))
    ordered = distribution.sort_values(["count", "label"], ascending=[False, True]).reset_index(drop=True)
    ranks = np.arange(1, len(ordered) + 1)
    head = ordered.head(ARTICLE_HEAD_SIZE)
    _horizontal_distribution(
        axes[0, 0],
        head,
        title=f"articleType — {len(head)} largest classes",
    )
    axes[0, 1].plot(ranks, ordered["count"], color="#4178a8")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(
        title=f"All {len(ordered)} classes ranked by support",
        xlabel="Class rank",
        ylabel="Products (log scale)",
    )
    axes[0, 1].grid(alpha=0.2)
    total = float(ordered["count"].sum())
    cumulative = ordered["count"].cumsum() / total if total else np.zeros(len(ordered))
    axes[1, 0].plot(ranks, cumulative, color="#54a24b")
    axes[1, 0].axhline(0.8, color="#777777", linestyle="--", linewidth=1)
    if len(cumulative):
        threshold = int(np.searchsorted(cumulative.to_numpy(), 0.8) + 1)
        axes[1, 0].axvline(threshold, color="#777777", linestyle=":", linewidth=1)
        axes[1, 0].annotate(
            f"{threshold} classes cover 80%",
            xy=(threshold, 0.8),
            xytext=(8, -22),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1, 0].set(
        title="How quickly common classes cover products",
        xlabel="Class rank",
        ylabel="Cumulative product share",
        ylim=(0, 1),
    )
    axes[1, 0].grid(alpha=0.2)
    support_bars = axes[1, 1].bar(
        support["band"].astype(str),
        support["class_count"],
        color="#e17c05",
    )
    axes[1, 1].bar_label(
        support_bars,
        labels=[f"{int(value)} classes" for value in support["class_count"]],
        padding=3,
        fontsize=8,
    )
    axes[1, 1].set(title="Class support bands", xlabel="Examples per class", ylabel="Classes")
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def _annotated_matrix(axis, matrix: pd.DataFrame, title: str) -> None:
    values = matrix.to_numpy(dtype=float)
    axis.imshow(values, vmin=0, vmax=max(1.0, float(np.nanmax(values, initial=0.0))), cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title(title)
    for row, column in np.ndindex(values.shape):
        axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)


def plot_association_matrix(matrix: pd.DataFrame) -> Figure:
    """Plot a categorical Cramér's V association matrix without a colorbar axis."""
    figure, axes = _new_figure(figsize=(8, 7))
    _annotated_matrix(axes[0, 0], matrix, "Categorical association (Cramér's V)")
    figure.tight_layout()
    return figure


def plot_relationship_heatmap(table: pd.DataFrame, row_name: str, column_name: str) -> Figure:
    """Plot a row-normalized categorical relationship heatmap."""
    figure, axes = _new_figure(figsize=(8, 6))
    axis = axes[0, 0]
    _annotated_matrix(axis, table, f"{row_name} × {column_name} (row-normalized)")
    axis.set_xlabel(column_name)
    axis.set_ylabel(row_name)
    figure.tight_layout()
    return figure


def plot_drift(table: pd.DataFrame, group_column: str) -> Figure:
    """Plot total-variation drift across year or deterministic ID ranges."""
    figure, axes = _new_figure(figsize=(9, 4))
    axis = axes[0, 0]
    ordered = table
    if group_column == "id_bin":
        lower_bounds = table[group_column].astype(str).str.split("–", n=1).str[0].astype(int)
        ordered = table.assign(_lower_bound=lower_bounds).sort_values("_lower_bound", kind="stable")
    axis.plot(ordered[group_column].astype(str), ordered["total_variation"], marker="o", color="#e17c05")
    axis.set(title=f"Distribution drift by {group_column}", xlabel=group_column, ylabel="Total variation")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    return figure


def plot_image_profiles(paired: pd.DataFrame) -> Figure:
    """Compare low- and high-resolution image quality metrics once per product."""
    metrics = [
        metric
        for metric in COMPARABLE_IMAGE_METRICS
        if f"{metric}_low" in paired and f"{metric}_high" in paired
    ]
    figure, axes = _new_figure(1, 2, figsize=(12, 4.5))
    low = [float(paired[f"{metric}_low"].mean()) for metric in metrics]
    high = [float(paired[f"{metric}_high"].mean()) for metric in metrics]
    positions = np.arange(len(metrics))
    axes[0, 0].bar(positions - 0.2, low, width=0.4, label="60×80")
    axes[0, 0].bar(positions + 0.2, high, width=0.4, label="original")
    axes[0, 0].set(
        title="Mean image properties (native sharpness excluded: resolution-dependent)",
        xticks=positions,
        xticklabels=metrics,
    )
    axes[0, 0].tick_params(axis="x", labelrotation=45)
    axes[0, 0].legend()
    deltas = [float(paired.get(f"{metric}_delta", pd.Series(dtype=float)).mean()) for metric in metrics]
    axes[0, 1].bar(metrics, deltas, color="#54a24b")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set(
        title="Original minus 60×80 (native sharpness excluded: resolution-dependent)",
        ylabel="Mean delta",
    )
    axes[0, 1].tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    return figure


def plot_duplicate_summary(summary: pd.DataFrame) -> Figure:
    """Plot exact and perceptual review-candidate counts."""
    figure, axes = _new_figure(figsize=(6, 4))
    axes[0, 0].bar(summary["kind"].astype(str), summary["count"], color=["#e45756", "#72b7b2"][: len(summary)])
    axes[0, 0].set(title="Duplicate review summary", ylabel="Groups or pairs")
    figure.tight_layout()
    return figure


def plot_review_grid(
    examples: Sequence[tuple[int, Image.Image, str]], title: str, columns: int = 4
) -> Figure:
    """Render a deterministic product-image review grid or visible empty state."""
    examples = list(examples)[:MAX_REVIEW_EXAMPLES]
    count = max(1, len(examples))
    rows = int(np.ceil(count / columns))
    figure, axes = _new_figure(rows, columns, figsize=(3 * columns, 2.8 * rows))
    for axis in axes.ravel():
        axis.axis("off")
    if not examples:
        axis = axes.ravel()[0]
        axis.text(
            0.5,
            0.5,
            "none found",
            ha="center",
            va="center",
            fontsize=16,
            transform=axis.transAxes,
        )
    for axis, (product_id, image, label) in zip(axes.ravel(), examples):
        axis.imshow(image)
        axis.set_title(f"{product_id}: {label}", fontsize=8)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def build_report_summary(
    distributions: Mapping[str, pd.DataFrame],
    associations: pd.DataFrame,
    paired: pd.DataFrame,
) -> Figure:
    """Build the fixed four-axis report figure with no extra colorbar axes."""
    figure, axes = _new_figure(2, 2, figsize=(15, 10))
    skew_axis, long_tail_axis, association_axis, quality_axis = axes.ravel()
    target_names = list(distributions)
    normalized_skew = []
    for name in target_names:
        distribution = distributions[name]
        learned = distribution.loc[~distribution.get("is_blank", pd.Series(False, index=distribution.index))]
        counts = learned["count"].to_numpy(dtype=float)
        probabilities = counts / counts.sum() if counts.sum() else np.array([], dtype=float)
        entropy = float(-(probabilities * np.log2(probabilities)).sum()) if len(probabilities) else 0.0
        normalized_entropy = entropy / np.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
        normalized_skew.append(1 - normalized_entropy)
    skew_axis.bar(target_names, normalized_skew, color="#e45756")
    skew_axis.set(
        title="Target normalized skew",
        ylabel="Normalized skew (0 balanced, 1 concentrated)",
        ylim=(0, 1),
    )
    skew_axis.tick_params(axis="x", labelrotation=35)

    article = distributions.get("articleType", pd.DataFrame({"count": []}))
    long_tail_axis.plot(np.arange(1, len(article) + 1), article["count"], color="#4178a8")
    long_tail_axis.set_yscale("log")
    long_tail_axis.set(title="articleType support (log)", xlabel="Class rank", ylabel="Products")

    _annotated_matrix(association_axis, associations, "Categorical association (Cramér's V)")

    metrics = [
        metric
        for metric in COMPARABLE_IMAGE_METRICS
        if f"{metric}_low" in paired and f"{metric}_high" in paired
    ]
    low_medians: list[float] = []
    high_medians: list[float] = []
    for metric in metrics:
        low = float(pd.to_numeric(paired[f"{metric}_low"], errors="coerce").median())
        high = float(pd.to_numeric(paired[f"{metric}_high"], errors="coerce").median())
        low = low if np.isfinite(low) else 0.0
        high = high if np.isfinite(high) else 0.0
        scale = max(low, high, 0.0)
        low_medians.append(np.clip(low / scale, 0.0, 1.0) if scale else 0.0)
        high_medians.append(np.clip(high / scale, 0.0, 1.0) if scale else 0.0)
    positions = np.arange(len(metrics))
    quality_axis.bar(
        positions - 0.2,
        low_medians,
        width=0.4,
        label="60×80",
    )
    quality_axis.bar(
        positions + 0.2,
        high_medians,
        width=0.4,
        label="original",
    )
    quality_axis.set(
        title="Low/high image quality (native sharpness excluded: resolution-dependent)",
        ylabel="Normalized median (within metric, 0–1)",
        ylim=(0, 1.05),
        xticks=positions,
        xticklabels=metrics,
    )
    quality_axis.tick_params(axis="x", labelrotation=40)
    quality_axis.legend()
    figure.tight_layout()
    return figure

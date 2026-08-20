"""Report-ready EDA figures built from protected inputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import TARGET_COLUMNS
from fashion.eda.diagnostics import build_split_balance_inputs
from fashion.eda.scope import bool_mask

PALETTE = ("#245B78", "#D97732", "#4C956C", "#8C6BB1", "#C44E52", "#6B7280")


def configure_style() -> None:
    """Apply one restrained style to every report figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAF8",
            "axes.edgecolor": "#C7C7C7",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E4E4E1",
            "grid.linestyle": "--",
            "grid.alpha": 0.75,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 14,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_article_type_long_tail(train: pd.DataFrame, path: Path) -> Path:
    """Show both the leading article types and the complete long tail."""
    counts = train.loc[bool_mask(train["has_articleType_label"]), "articleType"].value_counts()
    top = counts.head(25).sort_values()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    bars = left.barh(top.index, top.values, color=PALETTE[0])
    left.set_title("Most common training article types")
    left.set_xlabel("Training images")
    left.grid(axis="y", visible=False)
    left.bar_label(bars, labels=[f"{value:,}" for value in top.values], padding=3, fontsize=8)
    left.set_xlim(0, top.max() * 1.15)

    right.plot(
        np.arange(1, len(counts) + 1),
        counts.values,
        color=PALETTE[0],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    right.axhline(10, color=PALETTE[1], linestyle="--", label="Fewer than 10 images")
    right.set_yscale("log")
    right.set_title(f"All {len(counts)} training classes")
    right.set_xlabel("Class rank")
    right.set_ylabel("Training images (log scale)")
    right.legend(frameon=False)
    fig.suptitle("Article type has a strong long tail (training partition only)")
    fig.tight_layout()
    return _save(fig, path)


def plot_target_distributions(train: pd.DataFrame, path: Path) -> Path:
    """Compare the three compact supervised target distributions."""
    targets = ("gender", "season", "usage")
    titles = ("Gender", "Season", "Usage")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, target, title in zip(axes, targets, titles):
        valid = bool_mask(train[f"has_{target}_label"])
        counts = train.loc[valid, target].value_counts()
        missing = int((~valid).sum())
        labels = [*map(str, counts.index)]
        values = counts.astype(int).tolist()
        colors = [PALETTE[0]] * len(values)
        if missing:
            labels.append("Missing")
            values.append(missing)
            colors.append(PALETTE[1])
        bars = axis.bar(labels, values, color=colors)
        if target == "usage":
            axis.set_yscale("log")
            axis.set_ylabel("Training images (log scale)")
        else:
            axis.set_ylabel("Training images")
        axis.set_title(f"{title} (valid n={int(valid.sum()):,})")
        axis.tick_params(axis="x", rotation=35)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=2, fontsize=8)
        axis.grid(axis="x", visible=False)
    fig.suptitle("Training target distributions and unfilled labels")
    fig.tight_layout()
    return _save(fig, path)


def plot_image_profile(images: pd.DataFrame, path: Path) -> Path:
    """Show objective training-source geometry, encoding, and size evidence."""
    resolutions = images.groupby(["width", "height"]).size().sort_values()
    resolution_labels = [f"{int(width)}x{int(height)}" for width, height in resolutions.index]
    modes = images["mode"].value_counts()
    file_sizes = images["file_size_bytes"] / 1024
    aspect = images["aspect_ratio"].value_counts().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    resolution_bars = axes[0, 0].barh(resolution_labels, resolutions.values, color=PALETTE[0])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Native resolutions")
    axes[0, 0].set_xlabel("Training images (log scale)")
    axes[0, 0].bar_label(
        resolution_bars,
        labels=[f"{value:,}" for value in resolutions.values],
        padding=3,
        fontsize=8,
    )

    axes[0, 1].hist(file_sizes, bins=40, color=PALETTE[0], edgecolor="white")
    median = float(file_sizes.median())
    percentile_95 = float(file_sizes.quantile(0.95))
    axes[0, 1].axvline(median, color=PALETTE[1], linestyle="--", label=f"median {median:.1f} KiB")
    axes[0, 1].axvline(
        percentile_95, color=PALETTE[2], linestyle=":", label=f"95th {percentile_95:.1f} KiB"
    )
    axes[0, 1].set_title("Source file sizes")
    axes[0, 1].set_xlabel("KiB")
    axes[0, 1].set_ylabel("Training images")
    axes[0, 1].legend(frameon=False)

    mode_bars = axes[1, 0].bar(modes.index.astype(str), modes.values, color=PALETTE[: len(modes)])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Source colour modes")
    axes[1, 0].set_ylabel("Training images (log scale)")
    axes[1, 0].bar_label(mode_bars, labels=[f"{value:,}" for value in modes.values], padding=3)

    aspect_labels = [f"{float(value):.3f}" for value in aspect.index]
    aspect_bars = axes[1, 1].bar(aspect_labels, aspect.values, color=PALETTE[0])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Native aspect ratios")
    axes[1, 1].set_xlabel("Width / height")
    axes[1, 1].set_ylabel("Training images (log scale)")
    axes[1, 1].bar_label(
        aspect_bars, labels=[f"{value:,}" for value in aspect.values], padding=3, fontsize=8
    )
    fig.suptitle(f"Objective image profile (training only, n={len(images):,})")
    fig.tight_layout()
    return _save(fig, path)


def plot_development_balance(
    splits: pd.DataFrame,
    diagnostics: dict[str, dict[str, object]],
    path: Path,
) -> Path:
    """Show normalized Train/Validation balance without opening protected targets."""
    balance, denominators = build_split_balance_inputs(splits)
    positions = np.arange(len(balance))
    width = 0.38
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 1), hspace=0.55, wspace=0.35)
    top = fig.add_subplot(grid[0, :])
    gap_axis = fig.add_subplot(grid[1, 0])
    coverage_axis = fig.add_subplot(grid[1, 1])
    train_bars = top.bar(
        positions - width / 2,
        balance["train"],
        width,
        color=PALETTE[0],
        label=f"Train (n={denominators['train']:,})",
    )
    validation_bars = top.bar(
        positions + width / 2,
        balance["val"],
        width,
        color=PALETTE[1],
        label=f"Validation (n={denominators['val']:,})",
    )
    top.set_xticks(positions, balance.index, rotation=32, ha="right")
    top.set_ylabel("Within-partition valid labels (%)")
    top.set_title("Top 10 article types selected by training frequency")
    top.legend(frameon=False)
    top.bar_label(train_bars, fmt="%.1f", padding=2, fontsize=7)
    top.bar_label(validation_bars, fmt="%.1f", padding=2, fontsize=7)

    labels = ("Article type", "Season", "Gender", "Usage")
    gaps = [
        diagnostics[target]["distribution_gap_summary"]["max_absolute_percentage_point_gap"]
        for target in TARGET_COLUMNS
    ]
    gap_bars = gap_axis.barh(labels, gaps, color=PALETTE[:4])
    gap_axis.invert_yaxis()
    gap_axis.set_title("Largest class-share gap")
    gap_axis.set_xlabel("Percentage points")
    gap_axis.bar_label(gap_bars, fmt="%.2f pp", padding=3, fontsize=8)

    coverage = [diagnostics[target]["training_class_coverage_percent"] for target in TARGET_COLUMNS]
    coverage_bars = coverage_axis.barh(labels, coverage, color=PALETTE[:4])
    coverage_axis.invert_yaxis()
    coverage_axis.set_xlim(0, 108)
    coverage_axis.set_title("Training classes seen in validation")
    coverage_axis.set_xlabel("Coverage (%)")
    coverage_axis.bar_label(coverage_bars, fmt="%.1f%%", padding=3, fontsize=8)
    fig.suptitle("Development balance uses local percentages; holdout stays closed")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.96)
    return _save(fig, path)

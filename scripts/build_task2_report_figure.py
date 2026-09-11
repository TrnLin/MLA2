"""Build the compact Task 2 report figure from recorded evidence artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fashion.train.metrics import paired_group_bootstrap  # noqa: E402

MODEL_TABLE = (
    ROOT
    / "results/evidence/task2/development_model_comparison/all_model_oof_metrics.csv"
)
HOLDOUT_ROWS = (
    ROOT / "results/evidence/task2/final_evaluation/holdout_predictions_and_labels.csv"
)
BOOTSTRAP_INTERVALS = (
    ROOT / "results/evidence/task2/final_evaluation/holdout_bootstrap_intervals.csv"
)
OUTPUT = ROOT / "results/figures/task2/task2_report_selection_uncertainty.png"
LABELS = ("Fall", "Spring", "Summer", "Winter")


SHORT_LABELS = {
    "b0-majority": "B0 Majority",
    "b1-hog-hsv-svm": "B1 HOG+HSV SVM",
    "g1-c1-smallcnn": "G1 C1 SmallCNN",
    "g1-c2-resnet18": "G1 C2 ResNet18",
    "g1-c3-mobilenetv3": "G1 C3 MobileNetV3",
    "g2-p1-c2-resnet18": "G2 C2 P1 size",
    "g2-a1-c2-resnet18": "G2 C2 A1 augment",
    "g2-t1-c1-smallcnn": "G2 C1 T1",
    "g2-t1-c2-resnet18": "G2 C2 T1",
    "g2-t2-c1-smallcnn": "G2 C1 T2",
    "g2-t2-c2-resnet18": "G2 C2 T2",
    "g3-c1-t1-smallcnn": "G3 C1-T1 full",
    "g3-c2-t0-resnet18": "G3 C2-T0 full",
    "g4-i1-effective-number-c1": "G4 I1 balanced",
    "g4-i2-article-type-lambda-0-1-c1": "G4 I2 lambda=.1",
    "g4-i2-article-type-lambda-0-3-c1": "G4 I2 lambda=.3 [selected]",
    "g4-p0s-resnet18-standard-scratch": "G4 P0S scratch [bench.]",
    "g4-pstar-resnet18-standard-pretrained": "G4 P* pretrained [bench.]",
    "g5-c2-t0-resnet18-s2026": "G5 C2-T0 seed 2026",
    "g5-i2-article-type-lambda-0-3-c1-s2026": "G5 I2 lambda=.3 seed 2026",
}


def _load_bootstrap_draws() -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = pd.read_csv(HOLDOUT_ROWS)
    intervals = pd.read_csv(BOOTSTRAP_INTERVALS)
    draws = paired_group_bootstrap(
        scored["actual_season"].astype(str).to_numpy(),
        scored["product_family_group"].astype(str).to_numpy(),
        {
            "I2_minus_B0": (
                scored["b0_prediction"].astype(str).to_numpy(),
                scored["i2_prediction"].astype(str).to_numpy(),
            )
        },
        labels=LABELS,
        replicates=10_000,
        random_seed=2_753,
    )
    for metric, column in (
        ("i2_macro_f1", "model_b_macro_f1"),
        ("i2_minus_b0_macro_f1", "b_minus_a_macro_f1"),
    ):
        expected = intervals.loc[
            intervals["metric"].eq(metric), ["lower_95", "median", "upper_95"]
        ].iloc[0]
        actual = np.quantile(draws[column].to_numpy(dtype=float), [0.025, 0.5, 0.975])
        np.testing.assert_allclose(actual, expected.to_numpy(dtype=float), atol=1e-12, rtol=0)
    return draws, intervals


def _plot_models(axis: plt.Axes, catalog: pd.DataFrame) -> None:
    ordered = catalog.sort_values("workflow_order", ascending=False).reset_index(drop=True)
    ordered["short_label"] = ordered["experiment_id"].map(SHORT_LABELS)
    if ordered["short_label"].isna().any():
        missing = ordered.loc[ordered["short_label"].isna(), "experiment_id"].tolist()
        raise ValueError(f"missing report labels for: {missing}")

    y = np.arange(len(ordered))
    phases = ordered["phase"].astype(str).to_numpy()
    for index in range(len(phases) - 1):
        if phases[index] != phases[index + 1]:
            axis.axhline(index + 0.5, color="#D1D5DB", linewidth=0.75, zorder=0)
    axis.hlines(
        y,
        ordered[["macro_f1", "accuracy", "balanced_accuracy"]].min(axis=1),
        ordered[["macro_f1", "accuracy", "balanced_accuracy"]].max(axis=1),
        color="#D1D5DB",
        linewidth=1.0,
        zorder=1,
    )
    axis.scatter(
        ordered["macro_f1"], y, color="#111827", marker="o", s=27, zorder=3,
        label="Macro-F1",
    )
    axis.scatter(
        ordered["accuracy"], y, color="#2563EB", marker="s", s=24, zorder=3,
        label="Accuracy",
    )
    axis.scatter(
        ordered["balanced_accuracy"], y, color="#D97706", marker="^", s=29,
        zorder=3, label="Balanced accuracy",
    )

    selected = ordered["selected_for_freeze"].astype(bool).to_numpy()
    axis.scatter(
        ordered.loc[selected, "macro_f1"],
        y[selected],
        facecolors="none",
        edgecolors="#DC2626",
        linewidths=1.7,
        marker="o",
        s=74,
        zorder=4,
    )
    for index, row in ordered.iterrows():
        axis.text(
            min(float(row["macro_f1"]) + 0.012, 0.785),
            index,
            f"{float(row['macro_f1']):.3f}",
            va="center",
            fontsize=7.8,
            color="#111827",
        )

    axis.set_yticks(y, ordered["short_label"], fontsize=8.2)
    axis.set_xlim(0.12, 0.81)
    axis.set_xlabel("Pooled five-fold OOF score", fontsize=9)
    axis.set_title(
        "A. Development selection: all 20 registered configurations",
        loc="left",
        fontsize=10.4,
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.tick_params(axis="x", labelsize=8)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        fontsize=7.5,
        frameon=False,
        ncol=3,
        borderpad=0.2,
        handletextpad=0.4,
        columnspacing=0.9,
    )


def _plot_distribution(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    show_ylabel: bool,
) -> tuple[float, float, float]:
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    axis.hist(
        values,
        bins=38,
        density=True,
        color="#8ECAE6",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.72,
    )
    x_grid = np.linspace(values.min(), values.max(), 350)
    axis.plot(x_grid, gaussian_kde(values)(x_grid), color="#023047", linewidth=1.7)
    axis.axvspan(lower, upper, color="#FFB703", alpha=0.18)
    axis.axvline(lower, color="#FB8500", linestyle=":", linewidth=1.3)
    axis.axvline(median, color="#D00000", linestyle="--", linewidth=1.5)
    axis.axvline(upper, color="#FB8500", linestyle=":", linewidth=1.3)
    axis.set_title(title, fontsize=9.2, fontweight="bold", loc="left")
    axis.set_xlabel("Bootstrap statistic", fontsize=8)
    axis.set_ylabel("Density" if show_ylabel else "", fontsize=8)
    axis.tick_params(labelsize=7.5)
    axis.grid(axis="y", alpha=0.18)
    axis.text(
        0.02,
        0.95,
        f"95%: [{lower:.4f}, {upper:.4f}]\nmedian: {median:.4f}",
        transform=axis.transAxes,
        va="top",
        fontsize=7.6,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#D1D5DB"},
    )
    return float(lower), float(median), float(upper)


def build_report_figure() -> Path:
    """Write one page-efficient figure without dropping either evidence view."""
    catalog = pd.read_csv(MODEL_TABLE)
    if len(catalog) != 20:
        raise ValueError(f"expected 20 development configurations, found {len(catalog)}")
    draws, _ = _load_bootstrap_draws()

    figure = plt.figure(figsize=(12.6, 5.9))
    outer = figure.add_gridspec(1, 2, width_ratios=(1.08, 0.92), wspace=0.25)
    model_axis = figure.add_subplot(outer[0, 0])
    _plot_models(model_axis, catalog)

    right = outer[0, 1].subgridspec(3, 1, height_ratios=(1, 1, 0.33), hspace=0.52)
    absolute_axis = figure.add_subplot(right[0, 0])
    paired_axis = figure.add_subplot(right[1, 0])
    effect_axis = figure.add_subplot(right[2, 0])

    _plot_distribution(
        absolute_axis,
        draws["model_b_macro_f1"].to_numpy(dtype=float),
        title="B1. Holdout I2 macro-F1",
        show_ylabel=True,
    )
    lower, median, upper = _plot_distribution(
        paired_axis,
        draws["b_minus_a_macro_f1"].to_numpy(dtype=float),
        title="B2. Paired holdout improvement: I2 - B0",
        show_ylabel=True,
    )
    effect_axis.errorbar(
        median,
        0,
        xerr=np.array([[median - lower], [upper - median]]),
        fmt="o",
        color="#023047",
        ecolor="#FB8500",
        capsize=4,
        linewidth=1.8,
    )
    effect_axis.axvline(0, color="#D00000", linestyle="--", linewidth=1.3)
    effect_axis.set_xlim(-0.02, 0.64)
    effect_axis.set_yticks([])
    effect_axis.set_xlabel("Paired effect; red line = no effect (0)", fontsize=8)
    effect_axis.tick_params(axis="x", labelsize=7.5)
    effect_axis.grid(axis="x", alpha=0.18)

    distribution_legend = (
        Line2D([0], [0], color="#8ECAE6", linewidth=5, alpha=0.72, label="10,000 draws"),
        Line2D([0], [0], color="#023047", linewidth=1.7, label="KDE guide"),
        Line2D([0], [0], color="#FB8500", linestyle=":", label="95% endpoints"),
        Line2D([0], [0], color="#D00000", linestyle="--", label="Median"),
    )
    absolute_axis.legend(
        handles=distribution_legend,
        loc="upper right",
        fontsize=7.1,
        frameon=True,
        borderpad=0.35,
        handlelength=1.8,
    )

    figure.suptitle(
        "Task 2: model-selection evidence and family-blocked holdout uncertainty",
        fontsize=12,
        fontweight="bold",
        y=0.992,
    )
    figure.subplots_adjust(left=0.16, right=0.99, top=0.93, bottom=0.14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "MLA2 Task 2 report figure builder"},
    )
    plt.close(figure)
    return OUTPUT


if __name__ == "__main__":
    print(build_report_figure())

"""Rebuild Task 3's complete development charts from saved comparison evidence.

Registry-derived tables retain historical scores. Checked prediction audits supply
the final Gender recipes and teacher-only Usage expansion scores. No training,
inference, or final-evaluation data is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

SELECTION = "reports/task3/gender_mixup_selection_20260911"
USAGE = "reports/task3/usage_deep_investigation_20260905"
COLOURS = {"candidate": "#45749c", "selected": "#087f68", "diagnostic": "#b67519"}


def comparison_rows(root: Path, target: str) -> pd.DataFrame:
    """Return labelled points with their exact source and comparison population."""
    rows = []

    def add(panel, label, score, source, role="candidate"):
        rows.append(dict(panel=panel, label=label, macro_f1=score, source=source, role=role))

    def read(relative):
        return json.loads((root / relative).read_text())

    if target == "gender":
        source = f"{SELECTION}/all_gender_development_stages.csv"
        stages = pd.read_csv(root / source)
        for row in stages.itertuples():
            # The checked five-fold audit below replaces this historical readout.
            if row.stage == "final":
                continue
            folds = "5 folds" if row.rows == 32773 else "Folds 0 + 4"
            labels = "original labels" if row.label_basis == "Original" else "corrected labels"
            label = row.experiment.replace(" · ", " ")
            if row.stage == "dark":
                label = "Dropout + darkening"
            if row.stage == "gray":
                label = "Dropout + dark + gray 10%"
            add(
                f"{folds} | {labels}",
                label,
                row.macro_f1,
                source,
                "selected" if row.stage == "mixup" else "candidate",
            )
        source = f"{SELECTION}/summary.json"
        summary = read(source)
        for labels, key in [("original", "original_teacher_metrics"), ("corrected", "metrics")]:
            for model, metrics in summary[key].items():
                label = "MixUp 0.20 / 30 [audit]" if model == "MixUp" else "MixUp + SAM25 [audit]"
                add(
                    f"5 folds | {labels} labels",
                    label,
                    metrics["macro_f1"],
                    source,
                    "selected" if model == "MixUp" else "candidate",
                )
        source = "reports/task3/gender_overfitting_research_20260906/bn_probe_pooled.json"
        add(
            "Folds 0 + 4 | corrected labels",
            "BatchNorm recalibration",
            read(source)["clean_bn_cpu"]["macro_f1"],
            source,
            "diagnostic",
        )
    elif target == "usage":
        names = {
            "E1": "E1 SmallCNN",
            "E2": "E2 Class weights",
            "E3": "E3 Dropout",
            "E4": "E4 TinyResNet",
            "E5": "E5 Label smoothing",
            "E6": "E6 Focal loss",
            "E7": "E7 TinyConvNeXt",
            "E8": "E8 Translation",
            "E9": "E9 Exception balance",
            "S1": "S1 Type cascade",
            "S2": "S2 Micro-Swin",
            "U1": "U1 Component weights",
            "U2": "U2 Calibrated HOG-SVM",
        }
        source = f"{USAGE}/model_comparison.csv"
        models = pd.read_csv(root / source)
        for row in models.loc[models.scope.eq("all_available_folds")].itertuples():
            folds = "5 folds" if row.support == 32772 else "Folds 0 + 4"
            add(
                folds,
                names[row.model],
                row.macro_f1,
                source,
                "selected" if row.model == "E8" else "candidate",
            )
        for row in models.loc[
            models.scope.eq("folds_0_4") & models.model.isin(["E2", "E8"])
        ].itertuples():
            add(
                "Folds 0 + 4",
                names[row.model] + " (reference)",
                row.macro_f1,
                source,
                "selected" if row.model == "E8" else "candidate",
            )
        for folder, label in [
            ("usage_expanded_e8_result_20260906", "E8 +120 images"),
            ("usage_expanded_v2_e8_result_20260906", "E8 +687 images"),
        ]:
            source = f"reports/task3/{folder}/verified_summary.json"
            add("5 folds", label, read(source)["sources"]["teacher"]["metrics"]["macro_f1"], source)
        source = "reports/task3/usage_mixup_sam_result_20260906/verified_summary.json"
        mix = read(source)
        add("Folds 0 + 4", "E8 +687 (reference)", mix["baseline_teacher_f1"], source)
        add("Folds 0 + 4", "v2 +687, MixUp + SAM", mix["candidate_teacher_f1"], source)
        source = "reports/task3/usage_replaced_v3_mixup_sam_result_20260907/verified_summary.json"
        add("Folds 0 + 4", "v3 130 replacements", read(source)["candidate_teacher_f1"], source)
        source = "reports/task3/usage_u3_review_20260905/summary.csv"
        for row in pd.read_csv(root / source).itertuples():
            if row.model.startswith("U3"):
                add("Folds 0 + 4", row.model + " [artifact]", row.macro_f1, source)
        source = f"{USAGE}/u2_mechanism_summary.json"
        add(
            "Folds 0 + 4",
            "U2 Raw-margin diagnostic",
            read(source)["pooled_raw_mean_margin_f1"],
            source,
            "diagnostic",
        )
        for filename in ["fixed_blend_diagnostics.csv", "fixed_prior_diagnostics.csv"]:
            source = f"{USAGE}/{filename}"
            for row in pd.read_csv(root / source).itertuples():
                panel = "5 folds" if row.support == 32772 else "Folds 0 + 4"
                label = row.model + (" average" if "blend" in filename else "")
                add(panel, label, row.macro_f1, source, "diagnostic")
    else:
        raise ValueError(f"Unknown target: {target}")
    frame = pd.DataFrame(rows)
    assert frame.macro_f1.between(0, 1).all()
    assert not frame.duplicated(["panel", "label"]).any()
    return frame


def plot_model_comparison(root: Path, target: str, output_dir: Path | None = None):
    """Save the chart and its source table, and return the figure for notebooks."""
    data = comparison_rows(root, target)
    if target == "gender":
        panels = [
            "5 folds | original labels",
            "Folds 0 + 4 | original labels",
            "Folds 0 + 4 | corrected labels",
            "5 folds | corrected labels",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={"height_ratios": [1.6, 1]})
        limits = (64, 84)
        footer = (
            "Compare within panels. Original and corrected labels answer different questions.\n"
            "[audit] = checked IEEE predictions; "
            "other model points use registered confusion counts.\n"
            "Refits, sampled feature probes and incomplete attempts have no comparable OOF score."
        )
    else:
        panels = ["5 folds", "Folds 0 + 4"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        limits = (29, 46)
        footer = (
            "Teacher-only scoring rows, including for expanded-data models. "
            "Compare within panels.\n"
            "[artifact] = U3 audit; fold 4 lacks a completed registry receipt. "
            "Orange points reuse predictions.\n"
            "Refits, sampled feature probes and incomplete attempts have no comparable OOF score."
        )
    for ax, panel in zip(axes.flat, panels, strict=True):
        part = data.loc[data.panel.eq(panel)].sort_values("macro_f1", ascending=False)
        for y, row in enumerate(part.itertuples()):
            score = row.macro_f1 * 100
            ax.scatter(
                score,
                y,
                s=78 if row.role == "selected" else 48,
                marker="D" if row.role == "diagnostic" else "o",
                color=COLOURS[row.role],
                zorder=3,
            )
            ax.text(
                score + 0.35,
                y,
                f"{score:.2f}",
                fontsize=11,
                va="center",
                weight="bold" if row.role == "selected" else "normal",
            )
        ax.set_yticks(range(len(part)), part.label, fontsize=11)
        ax.set_ylim(len(part) - 0.5, -0.8)
        ax.set_xlim(*limits)
        ax.set_title(panel, loc="left", fontsize=14, weight="bold", pad=14)
        ax.set_xlabel("Pooled macro-F1 (%) · higher is better", fontsize=11)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"{target.title()} · full development comparison", fontsize=22, weight="bold", y=0.98
    )
    legend = [
        Line2D([], [], marker="o", linestyle="", color=COLOURS["candidate"], label="Other model"),
        Line2D(
            [], [], marker="o", linestyle="", color=COLOURS["selected"], label="Selected recipe"
        ),
        Line2D([], [], marker="D", linestyle="", color=COLOURS["diagnostic"], label="Diagnostic"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.text(0.04, 0.018, footer, fontsize=11, color="#46515b", linespacing=1.6)
    fig.tight_layout(rect=(0.015, 0.10, 0.995, 0.90), h_pad=2.7, w_pad=3)
    output_dir = output_dir or root / "results/figures/task3"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{target}_development_model_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=160, facecolor="white")
    data.to_csv(stem.with_suffix(".csv"), index=False)
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    for target in ["gender", "usage"]:
        plt.close(plot_model_comparison(root, target))

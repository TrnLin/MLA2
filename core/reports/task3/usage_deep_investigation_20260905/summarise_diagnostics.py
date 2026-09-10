"""Make report figures and paired sensitivity intervals without fitting models."""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyse_evidence import CLASSES, EXPERIMENTS, OUT, PROBS, ROOT, csv, save_json, score

from fashion.train.task3_decisions import paired_family_bootstrap

FIG = ROOT / "results/figures/task3/usage_investigation"


def main():
    frames = {
        k: csv(EXPERIMENTS[k] + "/oof_predictions.csv").set_index("id").sort_index()
        for k in ["E2", "E3", "E8", "U2"]
    }
    baseline = frames["E2"]
    candidate = baseline.copy()
    p = sum(frames[k][PROBS].to_numpy() for k in ["E2", "E3", "E8"]) / 3
    candidate[PROBS] = p
    candidate["predicted_index"] = p.argmax(axis=1)
    candidate["predicted_label"] = np.asarray(CLASSES)[p.argmax(axis=1)]
    summary = []
    for label, mask in [
        ("all_five_folds", np.ones(len(candidate), dtype=bool)),
        ("folds_0_4", candidate.cv_fold.isin([0, 4]).to_numpy()),
    ]:
        child, parent = candidate.loc[mask].copy(), baseline.loc[mask].copy()
        cm, pm = score(child[PROBS], child.true_index), score(parent[PROBS], parent.true_index)
        changes = [
            {"class_name": c["class_name"], "delta": c["f1"] - b["f1"]}
            for c, b in zip(cm["per_class"], pm["per_class"], strict=True)
        ]
        interval = paired_family_bootstrap(
            child.reset_index(), parent.reset_index(), classes=CLASSES
        )
        summary.append(
            {
                "model": "E2+E3+E8 equal probabilities",
                "scope": label,
                "metrics": cm,
                "paired_interval_vs_E2": interval,
                "nll_relative_improvement": (pm["nll"] - cm["nll"]) / pm["nll"],
                "brier_relative_improvement": (pm["brier"] - cm["brier"]) / pm["brier"],
                "non_home_deltas": [r for r in changes if r["class_name"] != "Home"],
                "posthoc_only": True,
                "robustness": "requires saved-model inference; not inferable from mean metrics",
            }
        )
    raw = csv(OUT / "u2_raw_margin_predictions.csv.gz").set_index("id").sort_index()
    parent = baseline.loc[raw.index].copy()
    rawchild = parent.copy()
    rawchild["predicted_index"] = raw.raw_margin_prediction.map(dict(zip(CLASSES, range(9))))
    assert rawchild.predicted_index.notna().all()
    save_json(
        "paired_diagnostic_intervals.json",
        {
            "fixed_blend": summary,
            "u2_raw_vs_E2": paired_family_bootstrap(
                rawchild.reset_index(), parent.reset_index(), classes=CLASSES
            ),
            "warning": (
                "All development folds have already been used in research. "
                "These intervals do not remove selection bias or prove a final test gain."
            ),
            "new_fits": 0,
        },
    )

    ledger = csv(OUT / "run_ledger.csv")
    model = csv(OUT / "model_comparison.csv")
    model = model.loc[model.scope.eq("all_available_folds")].set_index("model")
    classes = csv(OUT / "class_comparison.csv")
    classes = classes.loc[classes.scope.eq("all_available_folds")]
    order = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "S1", "S2", "U1", "U2"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    y = np.arange(len(order))
    vals = model.loc[order, "macro_f1"].astype(float)
    colors = ["#326e96" if len(model.loc[k, "folds"].split(",")) == 5 else "#bd7741" for k in order]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(
        y,
        [
            k + (" · 0/4 only" if colors[i] == "#bd7741" else " · 5 folds")
            for i, k in enumerate(order)
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 0.5)
    for i, value in enumerate(vals):
        ax.text(value + 0.006, i, f"{value:.4f}", va="center")
    ax.set_xlabel("Pooled macro-F1 · all nine labels retained")
    ax.set_title("Usage experiments: small gains and weak two-fold alternatives", pad=16)
    ax.text(
        0.99,
        0.01,
        "Different fold scopes are labelled.\n"
        "This is an evidence inventory, not a ranking of accepted models.",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.savefig(FIG / "experiment_summary.png", dpi=160)
    plt.close(fig)

    selected = ["E2", "E3", "E8", "E9", "U2"]
    table = (
        classes.pivot(index="model", columns="class_name", values="f1")
        .loc[selected, CLASSES]
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(12, 4.7), layout="constrained")
    im = ax.imshow(table, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(9),
        ["Casual", "Ethnic", "Formal", "Home", "NA", "Party", "Smart\nCasual", "Sports", "Travel"],
    )
    ax.set_yticks(
        range(5), ["E2 · 5 folds", "E3 · 5 folds", "E8 · 5 folds", "E9 · 5 folds", "U2 · 0/4"]
    )
    for i in range(5):
        for j in range(9):
            value = table.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#172633",
            )
    ax.set_title("Class F1: rare labels remain the central weakness", pad=18)
    fig.colorbar(im, ax=ax, label="F1", fraction=0.03, pad=0.025)
    fig.savefig(FIG / "class_f1_summary.png", dpi=160)
    plt.close(fig)

    robust = csv(OUT / "robustness_ledger.csv")
    robust["macro_f1_change"] = pd.to_numeric(robust.macro_f1_change)
    pooled = robust.groupby(["model", "corruption"]).macro_f1_change.mean().unstack()
    pooled.to_csv(OUT / "robustness_means.csv")
    matched = (
        robust.loc[robust.validation_fold.isin([0, 4])]
        .groupby(["model", "corruption"])
        .macro_f1_change.mean()
        .unstack()
    )
    (matched - matched.loc["E2"]).to_csv(OUT / "robustness_delta_vs_e2_folds_0_4.csv")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    corruptions = ["brightness_085", "brightness_115", "grayscale", "jpeg_75", "translation_003"]
    for i, label in enumerate(["E2", "E8", "U2"]):
        ax.bar(
            np.arange(5) + (i - 1) * 0.25, matched.loc[label, corruptions], width=0.25, label=label
        )
    ax.axhline(0, color="#55616e", linewidth=0.8)
    ax.set_xticks(range(5), ["Darker\n×0.85", "Brighter\n×1.15", "Gray", "JPEG 75", "Shift\n3%"])
    ax.set_ylabel("Mean fold macro-F1 change from clean images")
    ax.set_title("Robustness on matched folds 0 and 4", pad=15)
    ax.legend(ncol=3)
    fig.savefig(FIG / "robustness_summary.png", dpi=160)
    plt.close(fig)
    # Report numerical readback; pooled F1 above differs from mean fold F1 here.
    for column in ["macro_f1", "train_macro_f1", "gap", "train_seconds", "peak_memory_bytes"]:
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    ledger.groupby(["model", "train_source"])[
        ["macro_f1", "train_macro_f1", "gap", "train_seconds", "peak_memory_bytes"]
    ].mean().to_csv(OUT / "fit_summary.csv")
    print(
        json.dumps(
            [
                {
                    k: row[k]
                    for k in [
                        "scope",
                        "paired_interval_vs_E2",
                        "nll_relative_improvement",
                        "non_home_deltas",
                    ]
                }
                for row in summary
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

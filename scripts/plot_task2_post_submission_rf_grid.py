"""Render a zero-centred RF-grid comparison from verified development evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import atomic_write_json, verify_artifact

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE_DIR = ROOT / "results/evidence/task2/post_submission/rf_grid"
FIGURE_DIR = ROOT / "results/figures/task2/post_submission/rf_grid"


def main() -> Path:
    summary_path = EVIDENCE_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["study_id"] != "postsubmit-i2-rf-grid":
        raise ValueError("unexpected RF grid study")
    for name in ("model_comparison", "paired_bootstrap_intervals"):
        artifact = summary["artifacts"][name]
        verify_artifact(ROOT / artifact["path"], artifact["sha256"])
    models = pd.read_csv(EVIDENCE_DIR / "model_comparison.csv")
    intervals = pd.read_csv(EVIDENCE_DIR / "paired_bootstrap_intervals.csv")
    merged = models.merge(intervals, on="candidate", how="inner", validate="one_to_one")
    merged = merged.sort_values("delta_vs_r0")
    if len(merged) != 4 or (merged["delta_vs_r0"] >= 0.003).any():
        raise ValueError("figure protocol expects four non-qualifying variants")

    figure, axis = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    positions = np.arange(len(merged))
    estimates = merged["delta_vs_r0"].to_numpy(float)
    lower = merged["ci95_lower_y"].to_numpy(float)
    upper = merged["ci95_upper"].to_numpy(float)
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        color="#22577a",
        ecolor="#73a9ad",
        capsize=5,
        markersize=7,
        linewidth=2,
    )
    axis.axvline(0.0, color="#374151", linestyle="--", label="RF baseline: no change")
    axis.axvline(0.003, color="#b45309", linestyle=":", label="Predeclared +0.003 gate")
    axis.set_yticks(positions, merged["candidate"])
    axis.set_xlim(-0.0045, 0.0045)
    axis.set_xlabel("Pooled OOF macro-F1 change vs. original RF")
    axis.set_title("No RF-grid variant clears the predeclared improvement gate")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right", fontsize=9)
    axis.text(
        0.01,
        -0.22,
        "95% family-bootstrap intervals are descriptive; the same folds were "
        "used to compare variants.",
        transform=axis.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "paired_delta.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    atomic_write_json(
        EVIDENCE_DIR / "visual_manifest.json",
        {
            "schema_version": "1.0.0",
            "source_summary_sha256": compute_sha256(summary_path),
            "figure_path": path.relative_to(ROOT).as_posix(),
            "figure_sha256": compute_sha256(path),
        },
    )
    return path


if __name__ == "__main__":
    print(main())

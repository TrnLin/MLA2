"""Task 1 report figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from fashion.config import TASK1_FIGURE_DIR


def write_task1_comparison_figure(
    fold_metrics: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_preprocessing_macro_f1.png",
) -> Path:
    """Write a fold-level macro-F1 comparison with sample-standard-deviation bars."""
    required = {"preprocessing_id", "fold", "macro_f1"}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold_metrics are missing columns: {sorted(missing)}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped = list(fold_metrics.groupby("preprocessing_id", sort=False))
    for position, (preprocessing_id, candidate) in enumerate(grouped):
        values = candidate["macro_f1"].to_numpy(dtype=float)
        if len(values) != 5:
            raise ValueError("comparison figure requires exactly five folds per preprocessing ID")
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            alpha=0.8,
            label=preprocessing_id,
        )
        axis.errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1),
            color="black",
            capsize=6,
            fmt="D",
            zorder=3,
        )
    axis.set_xticks(range(len(grouped)), [name for name, _ in grouped], rotation=15, ha="right")
    axis.set_xlabel("Preprocessing candidate")
    axis.set_ylabel("Validation macro-F1 (124 classes)")
    axis.set_title("Task 1 five-fold preprocessing comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_task1_confusion_figure(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_oof_confusion_matrix.png",
) -> Path:
    """Write a normalized 124-class OOF confusion matrix after full CV is available."""
    if len(class_names) != 124:
        raise ValueError("Task 1 confusion evidence requires exactly 124 class names")
    required = {"true_index", "predicted_index"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    matrix = confusion_matrix(
        predictions["true_index"],
        predictions["predicted_index"],
        labels=np.arange(124),
        normalize="true",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(22, 20))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Within-class proportion")
    ticks = np.arange(124)
    axis.set_xticks(ticks, class_names, rotation=90, fontsize=4)
    axis.set_yticks(ticks, class_names, fontsize=4)
    axis.set_xlabel("Predicted article type")
    axis.set_ylabel("True article type")
    axis.set_title("Task 1 out-of-fold normalized confusion matrix")
    figure.text(0.5, 0.01, "Rows with no true examples are shown as zeros.", ha="center")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return output_path

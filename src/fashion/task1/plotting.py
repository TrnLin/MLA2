"""Task 1 report figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from fashion.config import TASK1_FIGURE_DIR


def write_task1_comparison_figure(
    fold_metrics: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_candidate_macro_f1.png",
) -> Path:
    """Write a fold-level macro-F1 comparison with sample-standard-deviation bars."""
    required = {"candidate_id", "fold", "macro_f1"}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold_metrics are missing columns: {sorted(missing)}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped = list(fold_metrics.groupby("candidate_id", sort=False))
    for position, (candidate_id, candidate) in enumerate(grouped):
        values = candidate["macro_f1"].to_numpy(dtype=float)
        if len(values) != 5:
            raise ValueError("comparison figure requires exactly five folds per candidate ID")
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            alpha=0.8,
            label=candidate_id,
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
    axis.set_xlabel("CNN candidate")
    axis.set_ylabel("Validation macro-F1 (124 classes)")
    axis.set_title("Task 1 five-fold CNN candidate comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _validated_learning_curve_data(
    histories: Mapping[str, Sequence[pd.DataFrame]],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return finite, aligned history arrays without creating output artifacts."""
    if not histories:
        raise ValueError("learning curve figure requires at least one candidate")
    required = {"epoch", "train_loss", "validation_loss", "macro_f1"}
    validated: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for candidate_id, candidate_histories in histories.items():
        if not candidate_histories:
            raise ValueError(f"candidate {candidate_id!r} has no histories")
        epoch_grid: np.ndarray | None = None
        fold_train_loss: list[np.ndarray] = []
        fold_validation_loss: list[np.ndarray] = []
        fold_macro_f1: list[np.ndarray] = []
        for history in candidate_histories:
            missing = required.difference(history.columns)
            if missing:
                raise ValueError(
                    f"history for {candidate_id!r} is missing columns: {sorted(missing)}"
                )
            try:
                epochs = history["epoch"].to_numpy(dtype=float)
                train_loss = history["train_loss"].to_numpy(dtype=float)
                validation_loss = history["validation_loss"].to_numpy(dtype=float)
                macro_f1 = history["macro_f1"].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"history for {candidate_id!r} must contain numeric values"
                ) from error
            if not len(epochs):
                raise ValueError(f"history for {candidate_id!r} has no epochs")
            values = (epochs, train_loss, validation_loss, macro_f1)
            if not all(np.isfinite(series).all() for series in values):
                raise ValueError(f"history for {candidate_id!r} must contain finite values")
            if epoch_grid is None:
                epoch_grid = epochs
            elif not np.array_equal(epoch_grid, epochs):
                raise ValueError(f"histories for {candidate_id!r} must use equal epoch grids")
            fold_train_loss.append(train_loss)
            fold_validation_loss.append(validation_loss)
            fold_macro_f1.append(macro_f1)
        if epoch_grid is None:
            raise RuntimeError("history validation did not establish an epoch grid")
        validated.append(
            (
                str(candidate_id),
                epoch_grid,
                np.vstack(fold_train_loss),
                np.vstack(fold_validation_loss),
                np.vstack(fold_macro_f1),
            )
        )
    return validated


def write_task1_learning_curve_figure(
    histories: Mapping[str, Sequence[pd.DataFrame]],
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_learning_curves.png",
) -> Path:
    """Write mean CNN learning curves from candidate-keyed fold histories."""
    candidates = _validated_learning_curve_data(histories)
    output_path = Path(output)
    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(15, 5.5))
    for position, candidate in enumerate(candidates):
        candidate_id, epochs, train_loss, validation_loss, macro_f1 = candidate
        color = f"C{position % 10}"
        mean_train_loss = train_loss.mean(axis=0)
        mean_validation_loss = validation_loss.mean(axis=0)
        mean_macro_f1 = macro_f1.mean(axis=0)
        loss_axis.plot(
            epochs,
            mean_train_loss,
            color=color,
            linestyle="--",
            label=f"{candidate_id} train",
        )
        loss_axis.plot(
            epochs,
            mean_validation_loss,
            color=color,
            label=f"{candidate_id} validation",
        )
        f1_axis.plot(epochs, mean_macro_f1, color=color, label=candidate_id)
        if len(train_loss) == 5:
            loss_axis.fill_between(
                epochs,
                mean_train_loss - train_loss.std(axis=0, ddof=1),
                mean_train_loss + train_loss.std(axis=0, ddof=1),
                color=color,
                alpha=0.12,
            )
            loss_axis.fill_between(
                epochs,
                mean_validation_loss - validation_loss.std(axis=0, ddof=1),
                mean_validation_loss + validation_loss.std(axis=0, ddof=1),
                color=color,
                alpha=0.12,
            )
            f1_axis.fill_between(
                epochs,
                mean_macro_f1 - macro_f1.std(axis=0, ddof=1),
                mean_macro_f1 + macro_f1.std(axis=0, ddof=1),
                color=color,
                alpha=0.12,
            )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Training and validation loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(fontsize=8)
    f1_axis.set_xlabel("Epoch")
    f1_axis.set_ylabel("Validation macro-F1 (124 classes)")
    f1_axis.set_title("Validation macro-F1")
    f1_axis.grid(alpha=0.25)
    f1_axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

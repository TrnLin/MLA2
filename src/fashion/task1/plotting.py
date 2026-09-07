"""Task 1 report figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix

from fashion.config import ROOT, TASK1_FIGURE_DIR


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


def write_task1_confusion_pair_figure(
    confusion_detail: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "top_confusion_pairs.png",
) -> Path:
    """Write a readable bar chart of the largest directed OOF errors."""
    required = {"true_label", "predicted_label", "error_count"}
    if missing := required.difference(confusion_detail.columns):
        raise ValueError(f"confusion detail is missing columns: {sorted(missing)}")
    if confusion_detail.empty:
        raise ValueError("confusion detail must contain at least one pair")
    detail = confusion_detail.copy()
    detail["error_count"] = pd.to_numeric(detail["error_count"], errors="raise")
    detail = detail.sort_values(
        ["error_count", "true_label", "predicted_label"],
        ascending=[True, False, False],
        kind="stable",
    )
    labels = detail["true_label"].astype(str) + " → " + detail["predicted_label"].astype(str)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(4.8, 0.5 * len(detail) + 1.8)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    bars = axis.barh(labels, detail["error_count"], color="#4C78A8")
    axis.bar_label(bars, labels=[str(int(value)) for value in detail["error_count"]], padding=4)
    axis.set_xlabel("OOF errors")
    axis.set_ylabel("True label → predicted label")
    axis.set_title("Most common article-type confusion pairs")
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlim(0, float(detail["error_count"].max()) * 1.14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _focus_labels(confusion_detail: pd.DataFrame, max_classes: int) -> list[str]:
    if max_classes < 2:
        raise ValueError("max_classes must be at least 2")
    required = {"true_label", "predicted_label"}
    if missing := required.difference(confusion_detail.columns):
        raise ValueError(f"confusion detail is missing columns: {sorted(missing)}")
    labels: list[str] = []
    for row in confusion_detail.itertuples(index=False):
        for label in (str(row.true_label), str(row.predicted_label)):
            if label not in labels and len(labels) < max_classes:
                labels.append(label)
    if len(labels) < 2:
        raise ValueError("focused confusion figure requires at least two labels")
    return labels


def write_task1_focused_confusion_figure(
    predictions: pd.DataFrame,
    confusion_detail: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "focused_confusion_matrix.png",
    max_classes: int = 8,
) -> Path:
    """Write an annotated OOF matrix for the classes in the largest errors."""
    required = {"true_label", "predicted_label"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    labels = _focus_labels(confusion_detail, max_classes)
    columns = [*labels, "Other"]
    counts = pd.DataFrame(0, index=labels, columns=columns, dtype=int)
    for row in predictions.loc[predictions["true_label"].astype(str).isin(labels)].itertuples(
        index=False
    ):
        true_label = str(row.true_label)
        predicted_label = str(row.predicted_label)
        column = predicted_label if predicted_label in labels else "Other"
        counts.loc[true_label, column] += 1
    support = counts.sum(axis=1)
    proportions = counts.div(support.replace(0, np.nan), axis=0).fillna(0.0)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, max(6, 0.8 * len(labels) + 2)))
    image = axis.imshow(proportions.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03, label="Within-class proportion")
    axis.set_xticks(np.arange(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Predicted article type")
    axis.set_ylabel("True article type")
    axis.set_title("Closer look at the largest OOF confusions")
    for row_index in range(len(labels)):
        for column_index in range(len(columns)):
            count = int(counts.iloc[row_index, column_index])
            proportion = float(proportions.iloc[row_index, column_index])
            color = "white" if proportion >= 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                f"{count}\n{proportion:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    figure.text(
        0.5,
        0.01,
        (
            "Rows use all OOF examples for that true class; Other contains "
            "predictions outside this view."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_task1_confusion_example_figure(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    confusion_detail: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "confusion_examples.png",
    pair_limit: int = 5,
    examples_per_pair: int = 2,
    root: str | Path = ROOT,
) -> Path:
    """Write representative source images for the largest directed OOF errors."""
    if pair_limit <= 0 or examples_per_pair <= 0:
        raise ValueError("pair_limit and examples_per_pair must be positive")
    prediction_required = {"id", "true_label", "predicted_label"}
    if missing := prediction_required.difference(predictions.columns):
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    split_required = {"id", "partition", "path"}
    if missing := split_required.difference(splits.columns):
        raise ValueError(f"splits are missing columns: {sorted(missing)}")
    detail_required = {"true_label", "predicted_label", "example_ids"}
    if missing := detail_required.difference(confusion_detail.columns):
        raise ValueError(f"confusion detail is missing columns: {sorted(missing)}")
    if confusion_detail.empty:
        raise ValueError("confusion detail must contain at least one pair")
    if predictions["id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate product IDs")
    if splits["id"].duplicated().any():
        raise ValueError("splits contain duplicate product IDs")

    prediction_by_id = predictions.assign(id=predictions["id"].astype(int)).set_index("id")
    split_by_id = splits.assign(id=splits["id"].astype(int)).set_index("id")
    examples: list[tuple[int, str, str, Path]] = []
    for pair in confusion_detail.head(pair_limit).itertuples(index=False):
        ids = [int(value) for value in str(pair.example_ids).split(",") if value.strip()]
        for product_id in ids[:examples_per_pair]:
            if product_id not in prediction_by_id.index:
                raise ValueError(f"example ID {product_id} is absent from OOF predictions")
            prediction = prediction_by_id.loc[product_id]
            if (
                str(prediction["true_label"]) != str(pair.true_label)
                or str(prediction["predicted_label"]) != str(pair.predicted_label)
            ):
                raise ValueError(f"example ID {product_id} does not match its confusion pair")
            if product_id not in split_by_id.index:
                raise ValueError(f"example ID {product_id} is absent from the saved split")
            split = split_by_id.loc[product_id]
            if str(split["partition"]) != "development":
                raise ValueError(f"example ID {product_id} must belong to development")
            image_path = Path(str(split["path"]))
            if not image_path.is_absolute():
                image_path = Path(root) / image_path
            if not image_path.is_file():
                raise ValueError(f"example image does not exist for ID {product_id}: {image_path}")
            examples.append(
                (product_id, str(pair.true_label), str(pair.predicted_label), image_path)
            )
    if not examples:
        raise ValueError("confusion detail contains no example IDs")

    pair_count = min(pair_limit, len(confusion_detail))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        examples_per_pair,
        pair_count,
        figsize=(3.2 * pair_count, 3.5 * examples_per_pair),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    position = 0
    for pair_index, pair in enumerate(confusion_detail.head(pair_limit).itertuples(index=False)):
        ids = [int(value) for value in str(pair.example_ids).split(",") if value.strip()]
        for example_index, _ in enumerate(ids[:examples_per_pair]):
            product_id, true_label, predicted_label, image_path = examples[position]
            position += 1
            try:
                with Image.open(image_path) as raw_image:
                    image = ImageOps.exif_transpose(raw_image).convert("RGB")
                    axes[example_index, pair_index].imshow(image)
            except OSError as error:
                plt.close(figure)
                raise ValueError(
                    f"example image is unreadable for ID {product_id}: {image_path}"
                ) from error
            axes[example_index, pair_index].set_title(
                f"ID {product_id}\n{true_label} → {predicted_label}", fontsize=9
            )
            axes[example_index, pair_index].axis("off")
    figure.suptitle("Examples from the largest OOF confusion pairs", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

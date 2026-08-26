"""Leakage checks and fixed-label multiclass metrics for pooled OOF evidence."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from fashion.train.artifacts import canonical_sha256

SEASON_LABELS = ("Fall", "Spring", "Summer", "Winter")


class OOFValidationError(ValueError):
    """Raised when out-of-fold predictions are incomplete, duplicated, or unsafe."""


def _unique_labels(labels: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(str(label) for label in labels)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("labels must contain at least two unique values in class order")
    return ordered


def validate_oof(
    frame: pd.DataFrame,
    *,
    expected_ids: Collection[Any],
    labels: Sequence[str] = SEASON_LABELS,
    protected_ids: Collection[Any] = (),
    id_column: str = "id",
    true_column: str = "y_true",
    prediction_column: str = "y_pred",
    probability_prefix: str = "prob_",
) -> dict[str, Any]:
    """Validate exactly-once OOF coverage, labels, and probability consistency."""
    ordered_labels = _unique_labels(labels)
    probability_columns = tuple(f"{probability_prefix}{label}" for label in ordered_labels)
    required = {id_column, true_column, prediction_column, *probability_columns}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise OOFValidationError(f"OOF frame is missing columns: {missing_columns}")

    relevant = frame.loc[
        :, [id_column, true_column, prediction_column, *probability_columns]
    ].copy()
    if relevant.isna().any().any():
        raise OOFValidationError("OOF frame contains missing IDs, labels, or probabilities")

    ids = relevant[id_column].astype(str)
    duplicate_ids = sorted(ids.loc[ids.duplicated(keep=False)].unique())
    if duplicate_ids:
        raise OOFValidationError(f"OOF IDs must appear once; duplicates: {duplicate_ids[:10]}")

    actual_ids = set(ids)
    expected = {str(value) for value in expected_ids}
    protected = {str(value) for value in protected_ids}
    missing_ids = sorted(expected - actual_ids)
    extra_ids = sorted(actual_ids - expected)
    if missing_ids or extra_ids:
        raise OOFValidationError(
            "OOF ID coverage differs from the canonical development set; "
            f"missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )
    leaked_ids = sorted(actual_ids & protected)
    if leaked_ids:
        raise OOFValidationError(f"OOF frame contains protected IDs: {leaked_ids[:10]}")

    allowed = set(ordered_labels)
    for column in (true_column, prediction_column):
        unknown = sorted(set(relevant[column].astype(str)) - allowed)
        if unknown:
            raise OOFValidationError(f"{column} contains unknown labels: {unknown}")

    probabilities = relevant.loc[:, probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise OOFValidationError("OOF probabilities must all be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise OOFValidationError("OOF probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise OOFValidationError("each OOF probability row must sum to one")

    predicted = np.asarray(ordered_labels, dtype=object)[probabilities.argmax(axis=1)]
    recorded = relevant[prediction_column].astype(str).to_numpy()
    if not np.array_equal(predicted, recorded):
        mismatch_count = int(np.count_nonzero(predicted != recorded))
        raise OOFValidationError(
            f"recorded predictions disagree with probability argmax in {mismatch_count} rows"
        )

    return {
        "row_count": len(relevant),
        "expected_row_count": len(expected),
        "unique_id_count": len(actual_ids),
        "id_set_sha256": canonical_sha256(sorted(actual_ids)),
        "labels": list(ordered_labels),
        "protected_id_count": 0,
        "probability_sum_tolerance": 1e-5,
    }


def multiclass_metrics(
    y_true: Sequence[str] | np.ndarray,
    *,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[str] = SEASON_LABELS,
    y_pred: Sequence[str] | np.ndarray | None = None,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Compute fixed-label discrimination and probability-quality metrics."""
    ordered_labels = _unique_labels(labels)
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")
    true = np.asarray(y_true, dtype=object)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true must be a non-empty one-dimensional sequence")
    if probability_array.shape != (len(true), len(ordered_labels)):
        raise ValueError(
            "probabilities must have shape "
            f"({len(true)}, {len(ordered_labels)}), got {probability_array.shape}"
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must all be finite")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(probability_array.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("each probability row must sum to one")

    allowed = set(ordered_labels)
    unknown_true = sorted(set(true.astype(str)) - allowed)
    if unknown_true:
        raise ValueError(f"y_true contains unknown labels: {unknown_true}")
    if y_pred is None:
        predicted = np.asarray(ordered_labels, dtype=object)[probability_array.argmax(axis=1)]
    else:
        predicted = np.asarray(y_pred, dtype=object)
        if predicted.shape != true.shape:
            raise ValueError("y_pred must have the same shape as y_true")
        unknown_predicted = sorted(set(predicted.astype(str)) - allowed)
        if unknown_predicted:
            raise ValueError(f"y_pred contains unknown labels: {unknown_predicted}")

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_labels,
        average="weighted",
        zero_division=0,
    )

    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    true_indices = np.fromiter(
        (label_to_index[str(label)] for label in true),
        dtype=np.int64,
        count=len(true),
    )
    one_hot = np.eye(len(ordered_labels), dtype=np.float64)[true_indices]
    brier = float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1)))

    confidence = probability_array.max(axis=1)
    correct = predicted.astype(str) == true.astype(str)
    bin_indices = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    ece = 0.0
    calibration_bins: list[dict[str, Any]] = []
    for index in range(ece_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        fraction = count / len(true)
        ece += fraction * abs(accuracy - mean_confidence)
        calibration_bins.append(
            {
                "bin": index,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return {
        "n_samples": len(true),
        "labels": list(ordered_labels),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "nll": float(log_loss(true, probability_array, labels=ordered_labels)),
        "brier": brier,
        "ece": float(ece),
        "ece_bins": calibration_bins,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(ordered_labels)
        },
        "confusion_matrix": confusion_matrix(
            true,
            predicted,
            labels=ordered_labels,
        ).astype(int).tolist(),
    }

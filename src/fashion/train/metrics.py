"""Framework-neutral classification metrics for registered OOF evidence."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np


def confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Build a fixed-size true-row, predicted-column confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be matching one-dimensional arrays")
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("a true label is outside the fixed class range")
    if predictions.size and (predictions.min() < 0 or predictions.max() >= num_classes):
        raise ValueError("a prediction is outside the fixed class range")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray, class_names: Sequence[str]) -> dict[str, object]:
    """Calculate fixed-class metrics, including zero-support classes."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion matrix and class names disagree")
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    weighted_f1 = float(np.average(f1, weights=support)) if total else 0.0

    correct = int(true_positive.sum())
    numerator = correct * total - int(np.dot(support, predicted_count))
    support_term = total * total - int(np.dot(support, support))
    prediction_term = total * total - int(np.dot(predicted_count, predicted_count))
    denominator = sqrt(max(0, support_term * prediction_term))
    mcc = float(numerator / denominator) if denominator else 0.0

    per_class = [
        {
            "class_index": index,
            "class_name": name,
            "support": int(support[index]),
            "predicted_count": int(predicted_count[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(class_names)
    ]
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "support": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Calculate top-label expected calibration error with equal-width bins."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Calculate the frozen Task 3 classification metric bundle."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities must have one row per label")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("probability columns must match the fixed class order")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if probabilities.size and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, num_classes=len(class_names))
    metrics = metrics_from_confusion(matrix, class_names)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    metrics["nll"] = float(-np.log(clipped[np.arange(len(labels)), labels]).mean())
    one_hot = np.eye(len(class_names), dtype=np.float64)[labels]
    metrics["brier"] = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    metrics["ece_15"] = expected_calibration_error(labels, probabilities, bins=15)
    return metrics

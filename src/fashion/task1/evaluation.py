"""Fixed-class evaluation and prediction evidence for Task 1."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

TASK1_NUM_CLASSES = 124


def _validated_arrays(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != (len(labels), num_classes):
        raise ValueError("labels and probability matrix shapes do not align")
    if len(labels) == 0 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("true labels must be non-empty valid class indexes")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    return labels, scores


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int = TASK1_NUM_CLASSES,
) -> dict[str, float]:
    """Calculate fixed-class F1 and supporting top-k accuracy metrics."""
    y_true, probabilities = _validated_arrays(y_true, probabilities, num_classes)
    y_pred = probabilities.argmax(axis=1)
    top_k = min(5, num_classes)
    top_indexes = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    labels = np.arange(num_classes)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "top1_accuracy": float(np.mean(y_pred == y_true)),
        "top5_accuracy": float(np.mean((top_indexes == y_true[:, None]).any(axis=1))),
    }


def per_class_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Return precision, recall, F1, and support for all 124 classes."""
    if len(class_names) != TASK1_NUM_CLASSES:
        raise ValueError("Task 1 per-class evidence requires 124 class names")
    y_true, probabilities = _validated_arrays(y_true, probabilities, TASK1_NUM_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        probabilities.argmax(axis=1),
        labels=np.arange(TASK1_NUM_CLASSES),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_index": np.arange(TASK1_NUM_CLASSES),
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def build_prediction_frame(
    product_ids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Build row-level prediction evidence with all class probabilities."""
    if len(class_names) != TASK1_NUM_CLASSES:
        raise ValueError("Task 1 prediction evidence requires 124 class names")
    y_true, probabilities = _validated_arrays(y_true, probabilities, TASK1_NUM_CLASSES)
    product_ids = np.asarray(product_ids, dtype=np.int64)
    if len(product_ids) != len(y_true) or len(np.unique(product_ids)) != len(product_ids):
        raise ValueError("prediction product IDs must be unique and align with labels")
    y_pred = probabilities.argmax(axis=1)
    data = {
        "id": product_ids,
        "true_index": y_true,
        "predicted_index": y_pred,
        "true_label": [class_names[index] for index in y_true],
        "predicted_label": [class_names[index] for index in y_pred],
    }
    data.update(
        {
            f"prob_{index:03d}": probabilities[:, index]
            for index in range(TASK1_NUM_CLASSES)
        }
    )
    return pd.DataFrame(data)


def validate_oof_predictions(
    predictions: pd.DataFrame,
    expected_ids: Collection[int],
) -> None:
    """Require exactly one out-of-fold prediction for each expected ID."""
    if "id" not in predictions.columns:
        raise ValueError("OOF predictions must contain an id column")
    if predictions["id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate product IDs")
    actual = set(predictions["id"].astype(int))
    expected = {int(product_id) for product_id in expected_ids}
    if actual != expected:
        raise ValueError("OOF predictions do not match expected development IDs")


def aggregate_fold_metrics(fold_metrics: Sequence[Mapping[str, float]]) -> pd.DataFrame:
    """Summarise exactly five fold metric mappings using mean and sample std."""
    frame = pd.DataFrame(fold_metrics)
    if len(frame) != 5:
        raise ValueError("full Task 1 evidence requires exactly five folds")
    return pd.DataFrame({"mean": frame.mean(), "std": frame.std(ddof=1)})

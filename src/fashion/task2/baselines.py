"""Leakage-safe non-neural reference methods for Season classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics


def _valid_target_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    required = {"id", "partition", target, f"has_{target}_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold frame is missing columns: {missing}")
    if not frame["partition"].eq("development").all():
        raise ValueError("baselines may use development rows only")
    mask = frame[f"has_{target}_label"]
    if not pd.api.types.is_bool_dtype(mask):
        mask = mask.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    valid = frame.loc[mask].copy()
    if valid.empty:
        raise ValueError(f"fold frame has no valid {target} labels")
    if valid["id"].duplicated().any():
        raise ValueError("fold IDs must be unique")
    if valid[target].astype(str).str.strip().eq("").any():
        raise ValueError(f"valid {target} masks cannot accompany blank labels")
    return valid.reset_index(drop=True)


@dataclass(frozen=True)
class MajorityBaselineModel:
    """Training-fold label priors with majority-class point predictions."""

    labels: tuple[str, ...]
    class_counts: tuple[int, ...]
    class_probabilities: tuple[float, ...]
    majority_label: str
    training_product_count: int
    training_id_sha256: str

    def predict_proba(self, product_count: int) -> np.ndarray:
        """Repeat training-fold priors without reading any validation features."""
        if product_count < 0:
            raise ValueError("product_count must be non-negative")
        probabilities = np.asarray(self.class_probabilities, dtype=np.float64)
        return np.repeat(probabilities[np.newaxis, :], product_count, axis=0)

    def predict(self, product_count: int) -> np.ndarray:
        """Return the training-fold majority label for every requested product."""
        if product_count < 0:
            raise ValueError("product_count must be non-negative")
        return np.repeat(np.asarray([self.majority_label], dtype=object), product_count)


@dataclass(frozen=True)
class MajorityFoldResult:
    """One validation fold's B0 predictions and fixed-label metrics."""

    validation_fold: int
    model: MajorityBaselineModel
    oof: pd.DataFrame
    metrics: dict[str, Any]


def fit_training_fold_majority(
    training_frame: pd.DataFrame,
    *,
    labels: tuple[str, ...] = SEASON_LABELS,
    target: str = "season",
) -> MajorityBaselineModel:
    """Fit label priors from valid training rows only; resolve ties by class order."""
    ordered_labels = tuple(str(label) for label in labels)
    if len(ordered_labels) < 2 or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("labels must contain at least two unique values")
    training = _valid_target_rows(training_frame, target)
    unknown = sorted(set(training[target].astype(str)) - set(ordered_labels))
    if unknown:
        raise ValueError(f"training fold contains unknown {target} labels: {unknown}")
    counts = tuple(
        int(training[target].astype(str).eq(label).sum()) for label in ordered_labels
    )
    total = sum(counts)
    probabilities = tuple(count / total for count in counts)
    majority_index = int(np.argmax(np.asarray(counts)))
    return MajorityBaselineModel(
        labels=ordered_labels,
        class_counts=counts,
        class_probabilities=probabilities,
        majority_label=ordered_labels[majority_index],
        training_product_count=total,
        training_id_sha256=canonical_sha256(sorted(int(value) for value in training["id"])),
    )


def evaluate_majority_fold(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    validation_fold: int,
    labels: tuple[str, ...] = SEASON_LABELS,
    target: str = "season",
) -> MajorityFoldResult:
    """Fit B0 on one training complement and create product-level OOF evidence."""
    if validation_fold < 0:
        raise ValueError("validation_fold must be non-negative")
    model = fit_training_fold_majority(training_frame, labels=labels, target=target)
    validation = _valid_target_rows(validation_frame, target)
    training_ids = set(int(value) for value in training_frame["id"])
    validation_ids = set(int(value) for value in validation["id"])
    if training_ids & validation_ids:
        raise ValueError("training and validation IDs overlap")
    if "cv_fold" in validation:
        folds = pd.to_numeric(validation["cv_fold"], errors="raise").astype(int)
        if not folds.eq(validation_fold).all():
            raise ValueError("validation rows do not match the requested canonical fold")
    unknown = sorted(set(validation[target].astype(str)) - set(model.labels))
    if unknown:
        raise ValueError(f"validation fold contains unknown {target} labels: {unknown}")

    probabilities = model.predict_proba(len(validation))
    predictions = model.predict(len(validation))
    truth = validation[target].astype(str).to_numpy()
    oof = pd.DataFrame(
        {
            "id": validation["id"].astype(int).to_numpy(),
            "fold": validation_fold,
            "y_true": truth,
            "y_pred": predictions,
        }
    )
    for index, label in enumerate(model.labels):
        oof[f"prob_{label}"] = probabilities[:, index]
    metrics = multiclass_metrics(
        truth,
        probabilities=probabilities,
        labels=model.labels,
        y_pred=predictions,
    )
    return MajorityFoldResult(
        validation_fold=validation_fold,
        model=model,
        oof=oof,
        metrics=metrics,
    )

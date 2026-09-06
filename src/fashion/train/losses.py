"""Leakage-safe loss fitting for controlled Task 2 imbalance experiments."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from fashion.train.artifacts import canonical_sha256

EFFECTIVE_NUMBER_BETA = 0.9999
EFFECTIVE_NUMBER_LOSS_ID = "effective_number_beta_0.9999"


def _validated_beta(beta: float) -> float:
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be a real number")
    resolved = float(beta)
    if not math.isfinite(resolved) or not 0.0 <= resolved < 1.0:
        raise ValueError("beta must be finite and in [0, 1)")
    return resolved


def _validated_counts(class_counts: Sequence[int]) -> np.ndarray:
    counts = tuple(class_counts)
    if len(counts) < 2:
        raise ValueError("class_counts must contain at least two classes")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in counts):
        raise TypeError("class_counts must contain integers")
    if any(int(value) <= 0 for value in counts):
        raise ValueError("every class must have at least one training sample")
    return np.asarray(counts, dtype=np.float64)


def effective_number_class_weights(
    class_counts: Sequence[int],
    *,
    beta: float = EFFECTIVE_NUMBER_BETA,
) -> tuple[float, ...]:
    """Return Cui et al. inverse-effective-number weights with mean one."""
    counts = _validated_counts(class_counts)
    resolved_beta = _validated_beta(beta)
    if resolved_beta == 0.0:
        return tuple(1.0 for _ in counts)

    # -expm1(n * log(beta)) evaluates 1 - beta**n accurately near beta=1.
    effective_denominator = -np.expm1(counts * math.log(resolved_beta))
    raw_weights = (1.0 - resolved_beta) / effective_denominator
    weights = raw_weights * (len(raw_weights) / raw_weights.sum())
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("effective-number weights must be finite and positive")
    return tuple(float(value) for value in weights)


@dataclass(frozen=True)
class EffectiveNumberAudit:
    """Fold-fitted class counts, weights, and training identity written to history."""

    labels: tuple[str, ...]
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...]
    beta: float
    training_product_count: int
    training_id_sha256: str
    loss_id: str
    schema_version: str = "1.0.0"

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported effective-number audit schema")
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must contain at least two unique values")
        if any(not label for label in self.labels):
            raise ValueError("labels must be non-empty")
        _validated_beta(self.beta)
        _validated_counts(self.class_counts)
        if len(self.class_counts) != len(self.labels):
            raise ValueError("class_counts must align with labels")
        if len(self.class_weights) != len(self.labels):
            raise ValueError("class_weights must align with labels")
        weights = np.asarray(self.class_weights, dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("class_weights must be finite and positive")
        if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("class_weights must have mean one")
        if self.training_product_count != sum(self.class_counts):
            raise ValueError("training_product_count must equal the class-count total")
        if len(self.training_id_sha256) != 64:
            raise ValueError("training_id_sha256 must be a SHA-256 digest")
        expected_loss_id = f"effective_number_beta_{self.beta:g}"
        if self.loss_id != expected_loss_id:
            raise ValueError("loss_id must encode the fitted beta")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["class_counts"] = {
            label: count for label, count in zip(self.labels, self.class_counts, strict=True)
        }
        payload["class_weights"] = {
            label: weight for label, weight in zip(self.labels, self.class_weights, strict=True)
        }
        payload["labels"] = list(self.labels)
        return payload


def fit_effective_number_weights(
    targets: Sequence[str],
    labels: Sequence[str],
    *,
    training_ids: Sequence[int],
    beta: float = EFFECTIVE_NUMBER_BETA,
) -> EffectiveNumberAudit:
    """Fit weights from one training fold and record the exact training ID set."""
    ordered_labels = tuple(str(label) for label in labels)
    if len(ordered_labels) < 2 or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("labels must contain at least two unique values")
    if any(not label for label in ordered_labels):
        raise ValueError("labels must be non-empty")

    observed_targets = tuple(str(target) for target in targets)
    identifiers = tuple(training_ids)
    if len(observed_targets) != len(identifiers):
        raise ValueError("targets and training_ids must have the same length")
    if not identifiers:
        raise ValueError("training fold must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in identifiers):
        raise TypeError("training_ids must contain integers")
    resolved_ids = tuple(int(value) for value in identifiers)
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("training_ids must be unique")

    unknown = sorted(set(observed_targets) - set(ordered_labels))
    if unknown:
        raise ValueError(f"targets contain labels outside the canonical order: {unknown}")
    counts_by_label = Counter(observed_targets)
    missing = [label for label in ordered_labels if counts_by_label[label] == 0]
    if missing:
        raise ValueError(f"training fold is missing canonical classes: {missing}")
    class_counts = tuple(counts_by_label[label] for label in ordered_labels)
    resolved_beta = _validated_beta(beta)
    audit = EffectiveNumberAudit(
        labels=ordered_labels,
        class_counts=class_counts,
        class_weights=effective_number_class_weights(class_counts, beta=resolved_beta),
        beta=resolved_beta,
        training_product_count=len(resolved_ids),
        training_id_sha256=canonical_sha256(sorted(resolved_ids)),
        loss_id=f"effective_number_beta_{resolved_beta:g}",
    )
    audit.validate()
    return audit


def build_effective_number_cross_entropy(audit: EffectiveNumberAudit) -> nn.CrossEntropyLoss:
    """Build weighted softmax cross-entropy from a validated fold audit."""
    audit.validate()
    weights = torch.tensor(audit.class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights)

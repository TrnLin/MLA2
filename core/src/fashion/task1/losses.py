"""Explicit loss identities and fold-local class weighting for Task 1."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight


@dataclass(frozen=True)
class Task1LossConfig:
    loss_id: str
    weighting: Literal["none", "balanced"]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Task1LossWeights:
    tensor: torch.Tensor | None
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {"class_counts": list(self.class_counts), "class_weights": list(self.class_weights)}


TASK1_UNWEIGHTED_LOSS = Task1LossConfig("cross_entropy_unweighted_v1", "none")
TASK1_BALANCED_WEIGHTED_LOSS = Task1LossConfig(
    "cross_entropy_balanced_class_weighted_v1", "balanced"
)


def build_task1_loss_weights(
    training_rows: pd.DataFrame,
    label_to_index: dict[str, int],
    *,
    validation_fold: int,
    config: Task1LossConfig,
) -> Task1LossWeights:
    """Build weights from development rows in the current fold's training set only."""
    if training_rows.empty:
        raise ValueError("training rows must be non-empty")
    required = {"partition", "cv_fold", "articleType"}
    missing = required.difference(training_rows.columns)
    if missing:
        raise ValueError(f"training rows missing columns: {sorted(missing)}")
    if not (training_rows["partition"] == "development").all() or (
        training_rows["cv_fold"] == validation_fold
    ).any():
        raise ValueError("weights require development training rows")
    if not config.loss_id or config.weighting not in {"none", "balanced"}:
        raise ValueError("invalid loss configuration")

    counts = np.zeros(len(label_to_index), dtype=np.int64)
    for label, count in training_rows["articleType"].value_counts().items():
        label_key = str(label)
        if label_key not in label_to_index:
            raise ValueError(f"unknown articleType label: {label_key}")
        counts[label_to_index[label_key]] = int(count)
    present = counts > 0
    if not present.any():
        raise ValueError("training rows must contain at least one known class")

    weights = np.zeros(len(counts), dtype=np.float64)
    if config.weighting == "balanced":
        encoded = training_rows["articleType"].astype(str).map(label_to_index).to_numpy()
        present_indices = np.flatnonzero(present)
        weights[present] = compute_class_weight(
            class_weight="balanced",
            classes=present_indices,
            y=encoded,
        )
        tensor = torch.tensor(weights, dtype=torch.float32)
    else:
        tensor = None
    return Task1LossWeights(
        tensor=tensor,
        class_counts=tuple(int(value) for value in counts),
        class_weights=tuple(float(value) for value in weights),
    )

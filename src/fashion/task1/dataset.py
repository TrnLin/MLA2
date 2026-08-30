"""Validated PyTorch dataset and fold helpers for Task 1."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fashion.config import ROOT
from fashion.data.dataset import get_cv_split, get_samples


class Task1TorchDataset(torch.utils.data.Dataset):
    """Load validated development rows as image, label, and ID tensors."""

    def __init__(
        self,
        rows: pd.DataFrame,
        transform: Callable[[Path], np.ndarray],
        label_to_index: Mapping[str, int],
        *,
        root: str | Path = ROOT,
    ) -> None:
        required = {"id", "path", "partition", "articleType"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"Task 1 rows are missing columns: {sorted(missing)}")
        if rows.empty or not rows["partition"].eq("development").all():
            raise ValueError("Task 1 training rows must be non-empty development rows")
        if rows["id"].isna().any() or rows["id"].duplicated().any():
            raise ValueError("Task 1 row IDs must be present and unique")
        if rows["path"].astype(str).str.strip().eq("").any():
            raise ValueError("Task 1 image paths must not be blank")
        unknown = set(rows["articleType"].astype(str)) - set(label_to_index)
        if unknown:
            raise ValueError(f"unknown articleType labels: {sorted(unknown)}")
        self.rows = rows.reset_index(drop=True).copy()
        self.transform = transform
        self.label_to_index = dict(label_to_index)
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        array = self.transform(self.root / str(row["path"]))
        if array.shape != (3, 80, 60) or array.dtype != np.float32:
            raise ValueError("Task 1 transform must return float32 shape (3, 80, 60)")
        return {
            "image": torch.from_numpy(array),
            "label": torch.tensor(self.label_to_index[str(row["articleType"])], dtype=torch.long),
            "id": torch.tensor(int(row["id"]), dtype=torch.long),
        }


def get_task1_fold_rows(
    splits: pd.DataFrame,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return article-type-labelled rows for one precomputed CV fold."""
    training, validation = get_cv_split(splits, validation_fold)
    return (
        get_samples(training, target="articleType"),
        get_samples(validation, target="articleType"),
    )

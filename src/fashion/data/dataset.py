"""Manifest loading, safe filtering, and a framework-neutral dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import LABEL_MAPS_JSON, ROOT, SPLITS_CSV, TARGET_COLUMNS
from fashion.data.images import load_and_transform_image
from fashion.data.splits import validate_splits


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_manifest(path: str | Path = SPLITS_CSV) -> pd.DataFrame:
    """Load a generated manifest and validate it when it contains partitions."""
    frame = pd.read_csv(path, keep_default_na=False)
    for column in [f"has_{target}_label" for target in TARGET_COLUMNS] + [
        "product_name_repaired",
        "is_cross_role_duplicate",
        "has_conflicting_target_labels",
    ]:
        if column in frame:
            frame[column] = _coerce_boolean(frame[column])
    if "id" in frame and frame["id"].notna().all():
        frame["id"] = frame["id"].astype(int)
    if "partition" in frame:
        validate_splits(frame)
    return frame


def load_splits(path: str | Path = SPLITS_CSV) -> pd.DataFrame:
    """Load the canonical shared split with all leakage invariants enforced."""
    frame = load_manifest(path)
    if "partition" not in frame:
        raise ValueError("split manifest has no partition column")
    return frame


def load_label_maps(path: str | Path = LABEL_MAPS_JSON) -> dict[str, dict[str, object]]:
    """Load generated label maps."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_samples(
    manifest: pd.DataFrame,
    partition: str | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    """Filter a manifest by partition and target validity without creating a new split."""
    filtered = manifest.copy()
    if partition is not None:
        if "partition" not in filtered:
            raise ValueError("manifest has no partition column")
        filtered = filtered[filtered["partition"].eq(partition)]
    if target is not None:
        mask_column = f"has_{target}_label"
        if mask_column not in filtered:
            raise ValueError(f"manifest has no {mask_column} column")
        filtered = filtered[_coerce_boolean(filtered[mask_column])]
    return filtered.copy()


class FashionDataset:
    """Return transformed images and metadata without depending on a DL framework."""

    def __init__(
        self,
        frame: pd.DataFrame,
        root: str | Path = ROOT,
        image_size: int = 128,
        pad_color: tuple[int, int, int] = (255, 255, 255),
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        targets: Sequence[str] = TARGET_COLUMNS,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.image_size = image_size
        self.pad_color = pad_color
        self.mean = mean
        self.std = std
        self.targets = tuple(targets)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        relative_path = Path(str(row["path"]))
        sample: dict[str, Any] = {
            "id": int(row["id"]),
            "path": relative_path.as_posix(),
            "image": load_and_transform_image(
                self.root / relative_path,
                image_size=self.image_size,
                pad_color=self.pad_color,
                mean=self.mean,
                std=self.std,
            ),
        }
        for target in self.targets:
            if target in row.index:
                sample[target] = row[target]
            mask_column = f"has_{target}_label"
            if mask_column in row.index:
                sample[mask_column] = bool(row[mask_column])
        for column in ("duplicate_group", "partition"):
            if column in row.index:
                sample[column] = row[column]
        return sample

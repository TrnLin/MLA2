"""Manifest loading, safe filtering, and a framework-neutral dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    CV_FOLD_COUNT,
    LABEL_MAPS_JSON,
    ROOT,
    SPLITS_CSV,
    TARGET_COLUMNS,
    TEACHER_TRAIN_CSV,
)
from fashion.data.metadata import has_valid_label
from fashion.data.splits import validate_splits

PROTECTED_TARGET_PARTITIONS = ("holdout", "quarantine")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _redact_protected_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove protected target values while retaining structural split fields."""
    if "partition" not in frame:
        return frame
    result = frame.copy()
    protected = result["partition"].isin(PROTECTED_TARGET_PARTITIONS)
    for target in TARGET_COLUMNS:
        if target in result:
            result.loc[protected, target] = ""
        mask_column = f"has_{target}_label"
        if mask_column in result:
            result.loc[protected, mask_column] = False
    return result


def _normalise_manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce persisted manifest fields without deciding target access."""
    for column in [f"has_{target}_label" for target in TARGET_COLUMNS] + [
        "product_name_repaired",
        "is_cross_role_duplicate",
        "is_cross_role_exact_duplicate",
        "is_cross_role_near_duplicate",
        "has_conflicting_target_labels",
    ]:
        if column in frame:
            frame[column] = _coerce_boolean(frame[column])
    if "id" in frame and frame["id"].notna().all():
        frame["id"] = frame["id"].astype(int)
    if "cv_fold" in frame:
        frame["cv_fold"] = pd.to_numeric(
            frame["cv_fold"].replace("", pd.NA), errors="coerce"
        ).astype("Int64")
    return frame


def load_manifest(path: str | Path = SPLITS_CSV) -> pd.DataFrame:
    """Load a manifest, always redacting targets in protected partitions."""
    frame = _normalise_manifest_frame(pd.read_csv(path, keep_default_na=False))
    if "partition" in frame:
        validate_splits(frame)
    return _redact_protected_targets(frame)


def load_splits(path: str | Path = SPLITS_CSV) -> pd.DataFrame:
    """Load the canonical split with holdout and quarantine targets redacted."""
    frame = load_manifest(path)
    if "partition" not in frame:
        raise ValueError("split manifest has no partition column")
    return frame


def load_splits_for_final_evaluation(
    path: str | Path = SPLITS_CSV,
    *,
    evaluation_unlocked: bool = False,
    raw_teacher_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Join protected targets from local raw data only after an explicit unlock."""
    if not evaluation_unlocked:
        raise ValueError("protected targets stay sealed until final evaluation is unlocked")
    splits_path = Path(path)
    frame = _normalise_manifest_frame(pd.read_csv(splits_path, keep_default_na=False))
    validate_splits(frame)
    inferred_root = splits_path.resolve().parents[2]
    raw_path = (
        Path(raw_teacher_csv)
        if raw_teacher_csv
        else inferred_root / TEACHER_TRAIN_CSV.relative_to(ROOT)
    )
    raw = pd.read_csv(
        raw_path,
        usecols=["id", *TARGET_COLUMNS],
        keep_default_na=False,
    )
    if raw["id"].duplicated().any():
        raise ValueError("raw teacher IDs must be unique for final evaluation")
    protected = frame["partition"].isin(PROTECTED_TARGET_PARTITIONS)
    protected_ids = frame.loc[protected, ["id"]]
    joined = protected_ids.merge(raw, on="id", how="left", validate="one_to_one")
    if joined[list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("raw teacher targets are missing for protected split IDs")
    joined.set_index("id", inplace=True)
    for target in TARGET_COLUMNS:
        frame.loc[protected, target] = frame.loc[protected, "id"].map(joined[target])
        frame.loc[protected, f"has_{target}_label"] = frame.loc[protected, target].map(
            has_valid_label
        )
    return _normalise_manifest_frame(frame)


def load_label_maps(path: str | Path = LABEL_MAPS_JSON) -> dict[str, dict[str, object]]:
    """Load generated label maps."""
    with Path(path).open("r", encoding="utf-8") as handle:
        maps = json.load(handle)
    for target, mapping in maps.items():
        if mapping.get("source_scope") != "development":
            raise ValueError(f"{target} label map was not fitted on development")
    return maps


def get_cv_split(
    splits: pd.DataFrame,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return four training folds and one validation fold from development."""
    if validation_fold not in range(CV_FOLD_COUNT):
        raise ValueError(f"validation_fold must be in range({CV_FOLD_COUNT})")
    validate_splits(splits)
    development = splits[splits["partition"].eq("development")].copy()
    fold_values = pd.to_numeric(development["cv_fold"], errors="raise").astype(int)
    validation = development[fold_values.eq(validation_fold)].copy()
    training = development[fold_values.ne(validation_fold)].copy()
    if training.empty or validation.empty:
        raise ValueError(f"cv_fold {validation_fold} does not create train and validation rows")
    return training, validation


def iter_cv_folds(
    splits: pd.DataFrame,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield every precomputed fold without creating a second split."""
    for validation_fold in range(CV_FOLD_COUNT):
        training, validation = get_cv_split(splits, validation_fold)
        yield validation_fold, training, validation


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
        transform: Callable[[Path], Any],
        root: str | Path = ROOT,
        targets: Sequence[str] = TARGET_COLUMNS,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.targets = tuple(targets)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        relative_path = Path(str(row["path"]))
        sample: dict[str, Any] = {
            "id": int(row["id"]),
            "path": relative_path.as_posix(),
            "image": self.transform(self.root / relative_path),
        }
        for target in self.targets:
            if target in row.index:
                sample[target] = row[target]
            mask_column = f"has_{target}_label"
            if mask_column in row.index:
                sample[mask_column] = bool(row[mask_column])
        for column in (
            "variant",
            "variant_key",
            "sample_group",
            "independence_group",
            "duplicate_group",
            "product_family_group",
            "partition",
            "cv_fold",
        ):
            if column in row.index:
                sample[column] = row[column]
        if "per_product_weight" in row.index:
            sample["per_product_weight"] = float(row["per_product_weight"])
        return sample

"""Manifest loading, safe filtering, and a framework-neutral dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    IMAGE_SIZE,
    LABEL_MAPS_JSON,
    ROOT,
    SPLITS_CSV,
    TARGET_COLUMNS,
    TAXONOMY_JSON,
    TEACHER_TRAIN_CSV,
)
from fashion.data.images import ImageSize, load_and_transform_image
from fashion.data.metadata import has_valid_label
from fashion.data.splits import validate_splits
from fashion.data.taxonomy import apply_deployment_taxonomy

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
        for column in (target, f"{target}_deployed", f"{target}_supported"):
            if column in result:
                result.loc[protected, column] = ""
        for column in (
            f"has_{target}_label",
            f"has_{target}_deployed_label",
            f"has_{target}_supported_label",
        ):
            if column in result:
                result.loc[protected, column] = False
        for column in (f"{target}_deployment_status", f"{target}_evaluation_status"):
            if column in result:
                result.loc[protected, column] = "protected"
    return result


def _normalise_manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce persisted manifest fields without deciding target access."""
    for column in [f"has_{target}_label" for target in TARGET_COLUMNS] + [
        *[f"has_{target}_deployed_label" for target in TARGET_COLUMNS],
        *[f"has_{target}_supported_label" for target in TARGET_COLUMNS],
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
    taxonomy_json: str | Path | None = None,
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
    taxonomy_path = (
        Path(taxonomy_json) if taxonomy_json else splits_path.parent / TAXONOMY_JSON.name
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
    frame, rebuilt_policy = apply_deployment_taxonomy(frame)
    expected_policy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    if rebuilt_policy != expected_policy:
        raise ValueError("train-fitted taxonomy changed before final evaluation")
    return _normalise_manifest_frame(frame)


def load_label_maps(path: str | Path = LABEL_MAPS_JSON) -> dict[str, dict[str, object]]:
    """Load generated label maps."""
    with Path(path).open("r", encoding="utf-8") as handle:
        maps = json.load(handle)
    for target, mapping in maps.items():
        if mapping.get("source_partition") != "train":
            raise ValueError(f"{target} label map was not fitted on train")
    return maps


def get_samples(
    manifest: pd.DataFrame,
    partition: str | None = None,
    target: str | None = None,
    deployed: bool = True,
) -> pd.DataFrame:
    """Filter a manifest by partition and target validity without creating a new split."""
    filtered = manifest.copy()
    if partition is not None:
        if "partition" not in filtered:
            raise ValueError("manifest has no partition column")
        filtered = filtered[filtered["partition"].eq(partition)]
    if target is not None:
        deployed_mask = f"has_{target}_deployed_label"
        mask_column = (
            deployed_mask if deployed and deployed_mask in filtered else f"has_{target}_label"
        )
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
        image_size: ImageSize = IMAGE_SIZE,
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
        for column in (
            "variant",
            "variant_key",
            "sample_group",
            "independence_group",
            "duplicate_group",
            "product_family_group",
            "partition",
        ):
            if column in row.index:
                sample[column] = row[column]
        if "per_product_weight" in row.index:
            sample["per_product_weight"] = float(row["per_product_weight"])
        return sample

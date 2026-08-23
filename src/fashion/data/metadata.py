"""Metadata repair, label masks, and deterministic label maps."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def repair_product_name(segments: Sequence[Any]) -> str:
    """Join non-empty CSV spill segments in their original order."""
    cleaned: list[str] = []
    for segment in segments:
        if segment is None or pd.isna(segment):
            continue
        text = str(segment).strip()
        if text:
            cleaned.append(text)
    return ", ".join(cleaned)


def has_valid_label(value: Any) -> bool:
    """Return whether a target value contains a real, non-empty label."""
    if value is None or pd.isna(value):
        return False
    return str(value).strip() != ""


def build_label_maps(
    manifest: pd.DataFrame,
    targets: Sequence[str],
    source_partition: str = "development",
) -> dict[str, dict[str, object]]:
    """Build stable mappings from every nonblank development-observed label."""
    source = manifest
    if "partition" in manifest:
        source = manifest[manifest["partition"].eq(source_partition)]
    if source.empty:
        raise ValueError(f"label-map source partition {source_partition!r} is empty")
    maps: dict[str, dict[str, object]] = {}
    for target in targets:
        label_column = target
        mask_column = f"has_{target}_label"
        labels = sorted(
            source.loc[_as_bool(source[mask_column]), label_column].astype(str).unique()
        )
        label_to_index = {label: index for index, label in enumerate(labels)}
        maps[target] = {
            "source_scope": source_partition,
            "source_partition": source_partition,
            "label_column": label_column,
            "validity_column": mask_column,
            "unknown_policy": "report_without_expanding_during_development",
            "unknown_index": -1,
            "num_classes": len(labels),
            "classes": labels,
            "label_to_index": label_to_index,
            "index_to_label": {str(index): label for label, index in label_to_index.items()},
        }
    return maps


def write_label_maps_from_splits(
    splits: pd.DataFrame | str | Path,
    output_path: str | Path,
    targets: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Fit stable label maps on image-backed development rows."""
    frame = (
        pd.read_csv(splits, keep_default_na=False)
        if isinstance(splits, (str, Path))
        else splits.copy()
    )
    maps = build_label_maps(frame, targets, source_partition="development")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(maps, indent=2), encoding="utf-8")
    return maps


def encode_target_labels(
    frame: pd.DataFrame,
    target: str,
    mapping: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode known labels and report labels absent from the development mapping."""
    label_column = str(mapping.get("label_column", target))
    mask_column = str(mapping.get("validity_column", f"has_{target}_label"))
    label_to_index = {
        str(label): int(index) for label, index in dict(mapping["label_to_index"]).items()
    }
    raw_valid = _as_bool(frame[mask_column]).to_numpy()
    encoded = frame[label_column].astype(str).map(label_to_index)
    known = encoded.notna().to_numpy() & raw_valid
    values = encoded.fillna(int(mapping.get("unknown_index", -1))).astype(int).to_numpy()
    unknown = sorted(frame.loc[raw_valid & ~known, label_column].astype(str).unique())
    return values, known, unknown

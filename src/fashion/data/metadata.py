"""Metadata repair, label masks, and deterministic label maps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


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
) -> dict[str, dict[str, object]]:
    """Build stable alphabetical label/index mappings from valid manifest labels."""
    maps: dict[str, dict[str, object]] = {}
    for target in targets:
        mask_column = f"has_{target}_label"
        labels = sorted(
            manifest.loc[manifest[mask_column].astype(bool), target].astype(str).unique()
        )
        label_to_index = {label: index for index, label in enumerate(labels)}
        maps[target] = {
            "num_classes": len(labels),
            "classes": labels,
            "label_to_index": label_to_index,
            "index_to_label": {str(index): label for label, index in label_to_index.items()},
        }
    return maps

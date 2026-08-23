"""Describe raw target vocabularies observed on image-backed development rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from fashion.config import TARGET_COLUMNS


def _bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_development_taxonomy(
    frame: pd.DataFrame,
    group_column: str = "product_family_group",
    targets: Sequence[str] = TARGET_COLUMNS,
) -> dict[str, Any]:
    """Build a descriptive vocabulary without dropping low-support classes."""
    if "partition" not in frame:
        raise ValueError("taxonomy requires a partition column")
    if group_column not in frame:
        raise ValueError(f"missing family group column: {group_column}")
    development = frame[frame["partition"].eq("development")]
    if development.empty:
        raise ValueError("taxonomy requires development rows")

    policy: dict[str, Any] = {
        "schema_version": "4.0.0",
        "source_scope": "development",
        "source_partition": "development",
        "group_column": group_column,
        "class_policy": "preserve_every_nonblank_development_observed_label",
        "semantic_merges": {},
        "targets": {},
    }
    for target in targets:
        mask_column = f"has_{target}_label"
        valid = _bool_mask(development[mask_column])
        labelled = development.loc[valid, [target, group_column]].copy()
        labelled[target] = labelled[target].astype(str)
        product_counts = labelled[target].value_counts().sort_index()
        family_counts = (
            labelled.drop_duplicates([target, group_column])
            .groupby(target)[group_column]
            .nunique()
            .sort_index()
        )
        classes = sorted(product_counts.index.astype(str))
        policy["targets"][target] = {
            "source_scope": "development",
            "label_column": target,
            "validity_column": mask_column,
            "num_classes": len(classes),
            "classes": classes,
            "product_counts": {label: int(product_counts[label]) for label in classes},
            "family_counts": {label: int(family_counts[label]) for label in classes},
        }
    return policy


def validate_development_taxonomy(
    frame: pd.DataFrame,
    policy: dict[str, Any],
    targets: Sequence[str] = TARGET_COLUMNS,
) -> None:
    """Confirm that the saved vocabulary exactly matches development labels."""
    rebuilt = build_development_taxonomy(frame, targets=targets)
    if rebuilt != policy:
        raise ValueError("development taxonomy does not match the canonical split")

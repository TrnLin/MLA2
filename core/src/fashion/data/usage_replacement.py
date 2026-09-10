"""Replace reviewed external Usage rows in a new version of a frozen dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import ROOT
from fashion.data.families import normalize_product_name
from fashion.data.splits import validate_splits
from fashion.data.usage_extension import (
    TEXT_CLASSES,
    extend_usage_dataset,
    validate_usage_extension,
)


def replace_usage_dataset(
    previous: pd.DataFrame,
    retirements: pd.DataFrame,
    collection: pd.DataFrame,
    label_maps: dict[str, dict[str, Any]],
    family_links: pd.DataFrame,
    *,
    root: str | Path = ROOT,
    check_files: bool = True,
) -> pd.DataFrame:
    """Exchange external images, preserving class totals and every retained row.

    The caller writes a new dataset path and retains the retirement receipt and
    old image files. All new IDs are above the previous maximum, including IDs
    retired here. Source families retain their existing fold anchors.
    """
    validate_splits(previous)
    if not {"id", "reason"}.issubset(retirements):
        raise ValueError("retirements require exact IDs and review reasons")
    if retirements.empty or retirements.id.astype(str).duplicated().any():
        raise ValueError("retirements must contain unique reviewed IDs")
    if retirements.reason.isna().any() or retirements.reason.astype(str).str.strip().eq("").any():
        raise ValueError("every retired row needs a review reason")
    removed_ids = set(retirements.id.astype(str))
    if not removed_ids.issubset(set(previous.id.astype(str))):
        raise ValueError("retired IDs must exist in the previous dataset")
    removed = previous.loc[previous.id.astype(str).isin(removed_ids)]
    if removed.source_dataset.eq("teacher").any() or not removed.partition.eq("development").all():
        raise ValueError("only added external development images may be replaced")
    if not removed.usage.isin(TEXT_CLASSES).all():
        raise ValueError("replacement is limited to the four reviewed rare classes")
    if removed.usage.value_counts().to_dict() != collection.usage.value_counts().to_dict():
        raise ValueError("replace the same number of images within each Usage class")
    # Retired identities remain reserved: replacing an image must mean a new image.
    if set(collection.external_id) & (set(previous.external_id) - {""}):
        raise ValueError("a replacement reuses an earlier external identity")
    for new_column, old_column in (("sha256", "sha256"), ("source_file_sha256", "original_sha256")):
        if set(collection[new_column]) & (set(previous[old_column]) - {""}):
            raise ValueError("a replacement duplicates an earlier image")
    names = set(collection.product_name.map(normalize_product_name)) - {""}
    if names & (set(previous.product_name_key) - {""}):
        raise ValueError("a replacement reuses an earlier product name")
    retained = previous.loc[~previous.id.astype(str).isin(removed_ids)].copy()
    # A wholly retired family still has a historical fold. Do not silently
    # reassign that family when no retained row remains to carry its anchor.
    family_column = (
        "extension_family_group" if "extension_family_group" in previous else "source_group_id"
    )
    retired_groups = set(removed[family_column]) - {""}
    retained_groups = set(retained[family_column]) - {""}
    new_groups = set(
        family_links.loc[family_links.external_id.isin(collection.external_id), "external_group_id"]
    )
    if new_groups & (retired_groups - retained_groups):
        raise ValueError("replacement links to a wholly retired family with a frozen fold")
    combined = extend_usage_dataset(
        retained, collection, label_maps, family_links, root=root, check_files=check_files
    )
    new_mask = combined.external_id.isin(collection.external_id)
    first_new_id = max(1_000_000_000, int(previous.id.astype(int).max()) + 1)
    new_ids = {
        external_id: str(first_new_id + index)
        for index, external_id in enumerate(sorted(collection.external_id))
    }
    combined.loc[new_mask, "id"] = combined.loc[new_mask, "external_id"].map(new_ids)
    validate_usage_extension(combined, retained, collection, family_links)
    if len(combined) != len(previous):
        raise ValueError("replacement changed the dataset size")
    if set(combined.id.astype(str)) & removed_ids:
        raise ValueError("retired IDs were reused")
    if combined.usage.value_counts().to_dict() != previous.usage.value_counts().to_dict():
        raise ValueError("replacement changed the Usage counts")
    return combined

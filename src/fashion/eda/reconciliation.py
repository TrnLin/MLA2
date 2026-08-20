"""Tracked reconciliation from raw metadata to active modelling taxonomies."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fashion.eda.scope import bool_mask

STALE_HEADLINE_COUNTS = {"articleType": 125, "usage": 8}


def build_data_reconciliation(
    csv_summary: dict[str, Any],
    splits: pd.DataFrame,
    label_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reconcile raw rows, usable images, repairs, and active target classes."""
    raw = csv_summary["styles_train"]
    image_reconciliation = raw["image_reconciliation"]
    raw_rows = int(raw["total_rows"])
    image_backed_rows = int(image_reconciliation["image_backed_rows"])
    excluded_rows = int(image_reconciliation["metadata_rows_without_valid_image"])
    if raw_rows - image_backed_rows != excluded_rows:
        raise ValueError("raw-to-image reconciliation counts do not balance")
    if len(splits) != image_backed_rows:
        raise ValueError("splits row count disagrees with the image-backed audit")

    repaired_names = int(bool_mask(splits["product_name_repaired"]).sum())
    taxonomy: dict[str, Any] = {}
    for target in ("articleType", "usage"):
        audit_target = image_reconciliation["target_taxonomy_changes"][target]
        active_classes = [str(label) for label in label_maps[target]["classes"]]
        active_count = int(label_maps[target]["num_classes"])
        if active_count != len(active_classes):
            raise ValueError(f"{target} label map class count is inconsistent")
        if active_count != int(audit_target["image_backed_valid_classes"]):
            raise ValueError(f"{target} label map disagrees with image-backed audit")
        taxonomy[target] = {
            "stale_headline_classes": STALE_HEADLINE_COUNTS[target],
            "raw_valid_classes": int(audit_target["raw_valid_classes"]),
            "active_image_backed_classes": active_count,
            "removed_by_missing_images": audit_target["removed_classes"],
            "current_evidence_supersedes_headline": True,
        }
        if target == "usage":
            taxonomy[target].update(
                {
                    "active_classes": active_classes,
                    "literal_NA_is_valid": "NA" in active_classes,
                    "raw_blank_rows": int(raw["missing_counts"]["usage"]),
                }
            )

    return {
        "schema_version": "1.0.0",
        "raw_to_usable": {
            "raw_metadata_rows": raw_rows,
            "image_backed_rows": image_backed_rows,
            "excluded_missing_image_rows": excluded_rows,
            "missing_valid_image_ids": image_reconciliation["missing_valid_image_ids"],
            "product_names_repaired": repaired_names,
        },
        "target_taxonomy": taxonomy,
        "scope_note": (
            "This is structural reconciliation. Holdout and quarantine target distributions "
            "remain excluded from modelling EDA."
        ),
    }

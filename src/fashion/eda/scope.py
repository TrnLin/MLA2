"""Hard scope boundaries for modelling EDA and leakage evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fashion.config import TARGET_COLUMNS
from fashion.data.splits import find_exact_duplicate_label_conflicts, validate_splits

PROTECTED_TARGET_PARTITIONS = ("holdout", "quarantine")


def bool_mask(series: pd.Series) -> pd.Series:
    """Normalize common serialized boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def select_modelling_scope(
    splits: pd.DataFrame,
    image_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training labels and matching decoded training-side image evidence only."""
    required_split = {"id", "partition", "sha256", *TARGET_COLUMNS}
    required_audit = {"id", "role", "decode_ok", "sha256"}
    if missing := required_split.difference(splits.columns):
        raise ValueError(f"splits.csv is missing columns: {sorted(missing)}")
    if missing := required_audit.difference(image_audit.columns):
        raise ValueError(f"image audit is missing columns: {sorted(missing)}")

    train = splits[splits["partition"].eq("train")].copy()
    if train.empty:
        raise ValueError("splits.csv contains no training rows")
    if train["id"].isna().any() or train["id"].duplicated().any():
        raise ValueError("training IDs must be non-null and unique")

    decoded_training = image_audit[
        image_audit["role"].eq("train") & bool_mask(image_audit["decode_ok"])
    ].copy()
    decoded_training = decoded_training[decoded_training["id"].notna()]
    decoded_training["id"] = decoded_training["id"].astype(int)
    eligible_ids = set(train["id"].astype(int))
    images = decoded_training[decoded_training["id"].isin(eligible_ids)].copy()
    if images["id"].duplicated().any():
        raise ValueError("training image audit has duplicate IDs")
    if set(images["id"]) != eligible_ids:
        missing_ids = sorted(eligible_ids.difference(images["id"]))[:5]
        raise ValueError(f"training image audit coverage mismatch; missing IDs {missing_ids}")
    if images["role"].ne("train").any():
        raise ValueError("official prediction images entered modelling EDA")
    return train.sort_values("id"), images.sort_values("id")


def derive_duplicate_evidence(
    splits: pd.DataFrame,
    image_audit: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
) -> dict[str, Any]:
    """Cross-check exact duplicate audit evidence against quarantine membership."""
    validate_splits(splits)
    required_audit = {"id", "role", "decode_ok", "sha256"}
    required_groups = {
        "sha256",
        "is_cross_role",
        "train_count",
        "test_count",
        "total_count",
    }
    if missing := required_audit.difference(image_audit.columns):
        raise ValueError(f"image audit is missing duplicate columns: {sorted(missing)}")
    if missing := required_groups.difference(duplicate_groups.columns):
        raise ValueError(f"duplicate evidence is missing columns: {sorted(missing)}")

    decoded = image_audit[
        bool_mask(image_audit["decode_ok"])
        & image_audit["role"].isin(("train", "prediction"))
        & image_audit["sha256"].notna()
    ].copy()
    decoded["sha256"] = decoded["sha256"].astype(str)
    counts = decoded.groupby(["sha256", "role"]).size().unstack(fill_value=0)
    for role in ("train", "prediction"):
        if role not in counts:
            counts[role] = 0
    cross_role = counts[counts["train"].gt(0) & counts["prediction"].gt(0)]
    cross_hashes = set(cross_role.index.astype(str))

    groups = duplicate_groups.copy()
    groups["sha256"] = groups["sha256"].astype(str)
    if groups["sha256"].duplicated().any():
        raise ValueError("duplicate evidence repeats a SHA-256 hash")
    reported_hashes = set(groups.loc[bool_mask(groups["is_cross_role"]), "sha256"])
    if reported_hashes != cross_hashes:
        raise ValueError("cross-role hash set disagrees with decoded image audit")
    reported = groups.set_index("sha256")
    for sha256, row in cross_role.iterrows():
        expected = {
            "train_count": int(row["train"]),
            "test_count": int(row["prediction"]),
            "total_count": int(row["train"] + row["prediction"]),
        }
        if any(int(reported.loc[sha256, key]) != value for key, value in expected.items()):
            raise ValueError("cross-role sample counts disagree with image audit")

    expected_training = int(cross_role["train"].sum())
    expected_prediction = int(cross_role["prediction"].sum())
    observed_cross_role = splits[splits["sha256"].astype(str).isin(cross_hashes)]
    conflicts = find_exact_duplicate_label_conflicts(splits)
    conflict_hashes = set(conflicts)
    observed_conflicts = splits[splits["sha256"].astype(str).isin(conflict_hashes)]
    quarantine = splits[splits["partition"].eq("quarantine")]
    if len(observed_cross_role) != expected_training:
        raise ValueError("cross-role training count disagrees with audit")
    if observed_cross_role["partition"].ne("quarantine").any():
        raise ValueError("a cross-role exact duplicate exists outside quarantine")
    if observed_conflicts["partition"].ne("quarantine").any():
        raise ValueError("an exact duplicate with conflicting labels exists outside quarantine")
    expected_quarantine_ids = set(observed_cross_role["id"].astype(int)) | set(
        observed_conflicts["id"].astype(int)
    )
    if set(quarantine["id"].astype(int)) != expected_quarantine_ids:
        raise ValueError("quarantine membership disagrees with approved exact-duplicate reasons")

    prediction_ids = set(
        image_audit.loc[
            image_audit["role"].eq("prediction") & image_audit["id"].notna(), "id"
        ].astype(int)
    )
    labelled_ids = set(splits["id"].astype(int))
    return {
        "exact_duplicate_groups_source_audit": len(groups),
        "cross_role_exact_duplicate_groups": len(cross_hashes),
        "cross_role_training_samples": expected_training,
        "cross_role_prediction_samples": expected_prediction,
        "conflicting_label_exact_duplicate_groups": len(conflict_hashes),
        "conflicting_label_training_samples": len(observed_conflicts),
        "conflicting_label_groups_by_target": {
            target: sum(target in group_targets for group_targets in conflicts.values())
            for target in TARGET_COLUMNS
        },
        "quarantined_training_samples": len(quarantine),
        "cross_role_training_samples_outside_quarantine": 0,
        "conflicting_label_training_samples_outside_quarantine": 0,
        "exact_sha256_groups_crossing_partitions": int(
            (splits.groupby("sha256")["partition"].nunique() > 1).sum()
        ),
        "official_prediction_ids_in_labelled_splits": len(prediction_ids & labelled_ids),
    }


def scope_record(train: pd.DataFrame, images: pd.DataFrame) -> dict[str, Any]:
    """Describe the enforced EDA boundary without reading protected labels."""
    return {
        "modelling_partition": "train",
        "modelling_rows": len(train),
        "modelling_image_rows": len(images),
        "validation_usage": "normalized distribution and coverage diagnostics only",
        "protected_target_partitions": list(PROTECTED_TARGET_PARTITIONS),
        "prediction_images_in_modelling_evidence": 0,
    }

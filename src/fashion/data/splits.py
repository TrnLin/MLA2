"""Build and validate the project's sole leakage-safe split file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import (
    AUDIT_DIR,
    DEVELOPMENT_CLASS_SUMMARY_CSV,
    HOLDOUT_RATIO,
    RANDOM_SEED,
    SPLIT_SUMMARY_JSON,
    SPLITS_CSV,
    TARGET_COLUMNS,
    TRAIN_MANIFEST_CSV,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)

PARTITIONS = ("train", "val", "holdout", "quarantine")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def find_exact_duplicate_label_conflicts(
    frame: pd.DataFrame,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> dict[str, tuple[str, ...]]:
    """Return exact hashes with multiple distinct valid labels for any target."""
    required = {"sha256"}
    for target in targets:
        required.update({target, f"has_{target}_label"})
    if missing := required.difference(frame.columns):
        raise ValueError(f"cannot inspect duplicate labels; missing columns: {sorted(missing)}")

    duplicated = frame[frame.duplicated("sha256", keep=False)]
    conflicts: dict[str, tuple[str, ...]] = {}
    for sha256, group in duplicated.groupby("sha256", sort=True):
        conflicting_targets: list[str] = []
        for target in targets:
            valid = group.loc[_as_bool(group[f"has_{target}_label"]), target]
            if valid.astype(str).str.strip().nunique() > 1:
                conflicting_targets.append(target)
        if conflicting_targets:
            conflicts[str(sha256)] = tuple(conflicting_targets)
    return conflicts


def validate_splits(
    frame: pd.DataFrame,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> None:
    """Raise when split membership or exact-duplicate quarantine invariants fail."""
    required = {
        "id",
        "sha256",
        "duplicate_group",
        "partition",
        "is_cross_role_duplicate",
        "has_conflicting_target_labels",
        "conflicting_targets",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"split data is missing columns: {sorted(missing)}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("split IDs must be non-null and unique")
    if frame["partition"].isna().any():
        raise ValueError("every split row must have a partition")
    unknown = set(frame["partition"]) - set(PARTITIONS)
    if unknown:
        raise ValueError(f"unknown partitions: {sorted(unknown)}")
    if (frame.groupby("sha256")["partition"].nunique() > 1).any():
        raise ValueError("an exact SHA-256 duplicate group crosses partitions")
    if (frame.groupby("duplicate_group")["partition"].nunique() > 1).any():
        raise ValueError("a duplicate group crosses partitions")
    cross_role = _as_bool(frame["is_cross_role_duplicate"])
    if not frame.loc[cross_role, "partition"].eq("quarantine").all():
        raise ValueError("a cross-role exact duplicate exists outside quarantine")

    conflicts = find_exact_duplicate_label_conflicts(frame, targets)
    conflict_hashes = set(conflicts)
    expected_conflict = frame["sha256"].astype(str).isin(conflict_hashes)
    reported_conflict = _as_bool(frame["has_conflicting_target_labels"])
    if not reported_conflict.equals(expected_conflict):
        raise ValueError("conflicting-target flags disagree with exact duplicate labels")
    expected_targets = (
        frame["sha256"].astype(str).map(lambda digest: ",".join(conflicts.get(digest, ())))
    )
    if not frame["conflicting_targets"].astype(str).eq(expected_targets).all():
        raise ValueError("conflicting-target names disagree with exact duplicate labels")
    if not frame.loc[expected_conflict, "partition"].eq("quarantine").all():
        raise ValueError("an exact duplicate with conflicting labels exists outside quarantine")

    allowed_quarantine = cross_role | expected_conflict
    if frame.loc[~allowed_quarantine, "partition"].eq("quarantine").any():
        raise ValueError("quarantine contains a row without an approved quarantine reason")


def _allocate_groups(
    groups: pd.DataFrame,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[dict[str, str], list[str], list[str]]:
    allocation: dict[str, str] = {}
    quarantine = groups["is_cross_role"] | groups["has_label_conflict"]
    for group_id in groups.loc[quarantine, "duplicate_group"]:
        allocation[group_id] = "quarantine"

    available = groups.loc[~quarantine]
    random = np.random.RandomState(seed)
    singleton_classes: list[str] = []
    dual_classes: list[str] = []
    for article_type in sorted(available["articleType"].unique()):
        class_groups = available.loc[
            available["articleType"].eq(article_type), "duplicate_group"
        ].tolist()
        random.shuffle(class_groups)
        count = len(class_groups)
        if count == 1:
            singleton_classes.append(str(article_type))
            allocation[class_groups[0]] = "train"
            continue
        if count == 2:
            dual_classes.append(str(article_type))
            allocation[class_groups[0]] = "train"
            allocation[class_groups[1]] = "val"
            continue

        train_count = max(1, round(count * train_ratio))
        if count - train_count < 2:
            train_count = count - 2
        validation_count = max(1, round(count * validation_ratio))
        if train_count + validation_count >= count:
            validation_count = max(1, (count - train_count) // 2)
        boundaries = (train_count, train_count + validation_count)
        for group_id in class_groups[: boundaries[0]]:
            allocation[group_id] = "train"
        for group_id in class_groups[boundaries[0] : boundaries[1]]:
            allocation[group_id] = "val"
        for group_id in class_groups[boundaries[1] :]:
            allocation[group_id] = "holdout"
    return allocation, singleton_classes, dual_classes


def make_splits(
    train_manifest_csv: str | Path = TRAIN_MANIFEST_CSV,
    duplicate_groups_csv: str | Path = AUDIT_DIR / "exact_duplicate_groups.csv",
    output_csv: str | Path = SPLITS_CSV,
    summary_output: str | Path = SPLIT_SUMMARY_JSON,
    development_summary_output: str | Path = DEVELOPMENT_CLASS_SUMMARY_CSV,
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    holdout_ratio: float = HOLDOUT_RATIO,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Create one deterministic grouped split and no aliases."""
    if not np.isclose(train_ratio + validation_ratio + holdout_ratio, 1.0):
        raise ValueError("train, validation, and holdout ratios must total 1")
    output_csv = Path(output_csv)
    summary_output = Path(summary_output)
    development_summary_output = Path(development_summary_output)
    for path in (output_csv, summary_output, development_summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(train_manifest_csv, keep_default_na=False)
    duplicate_groups = pd.read_csv(duplicate_groups_csv)
    duplicate_map: dict[str, str] = {}
    cross_role_hashes: set[str] = set()
    if not duplicate_groups.empty:
        duplicate_groups["sha256"] = duplicate_groups["sha256"].astype(str)
        duplicate_map = duplicate_groups.set_index("sha256")["duplicate_group"].to_dict()
        cross_role_hashes = set(
            duplicate_groups.loc[_as_bool(duplicate_groups["is_cross_role"]), "sha256"]
        )

    manifest["sha256"] = manifest["sha256"].astype(str)
    manifest["duplicate_group"] = manifest["sha256"].map(duplicate_map)
    missing_group = manifest["duplicate_group"].isna()
    manifest.loc[missing_group, "duplicate_group"] = manifest.loc[missing_group, "sha256"].map(
        lambda digest: f"single_{digest[:12]}"
    )
    manifest["is_cross_role_duplicate"] = manifest["sha256"].isin(cross_role_hashes)
    conflicts = find_exact_duplicate_label_conflicts(manifest, targets)
    manifest["conflicting_targets"] = manifest["sha256"].map(
        lambda digest: ",".join(conflicts.get(str(digest), ()))
    )
    manifest["has_conflicting_target_labels"] = manifest["conflicting_targets"].ne("")

    grouped_rows: list[dict[str, Any]] = []
    for group_id, group in manifest.groupby("duplicate_group"):
        grouped_rows.append(
            {
                "duplicate_group": group_id,
                "sha256": group["sha256"].iloc[0],
                "size": len(group),
                "articleType": group["articleType"].iloc[0],
                "is_cross_role": bool(group["is_cross_role_duplicate"].any()),
                "has_label_conflict": bool(group["has_conflicting_target_labels"].any()),
            }
        )
    groups = pd.DataFrame(grouped_rows)
    allocation, singleton_classes, dual_classes = _allocate_groups(
        groups, seed, train_ratio, validation_ratio
    )
    manifest["partition"] = manifest["duplicate_group"].map(allocation)
    validate_splits(manifest)
    manifest.sort_values("id", inplace=True)
    manifest.to_csv(output_csv, index=False)

    partition_counts = manifest["partition"].value_counts()
    development_rows: list[dict[str, Any]] = []
    target_summary: dict[str, Any] = {}
    for target in targets:
        target_summary[target] = {}
        for partition in ("train", "val"):
            mask = _as_bool(manifest[f"has_{target}_label"])
            subset = manifest[manifest["partition"].eq(partition) & mask]
            counts = subset[target].value_counts()
            target_summary[target][partition] = {
                "valid_count": len(subset),
                "num_classes": len(counts),
            }
            for label, count in counts.items():
                development_rows.append(
                    {
                        "target": target,
                        "class": label,
                        "partition": partition,
                        "count": int(count),
                    }
                )
        target_summary[target]["protected_partitions"] = {
            "partitions": ["holdout", "quarantine"],
            "status": "target outcomes are not aggregated before final evaluation",
        }
    pd.DataFrame(development_rows).to_csv(development_summary_output, index=False)

    total = len(manifest)
    quarantine_count = int(partition_counts.get("quarantine", 0))
    cross_role_rows = int(manifest["is_cross_role_duplicate"].sum())
    conflict_rows = int(manifest["has_conflicting_target_labels"].sum())
    conflict_target_counts = {
        target: sum(target in conflicting_targets for conflicting_targets in conflicts.values())
        for target in targets
    }
    summary = {
        "schema_version": "2.0.0",
        "seed": seed,
        "split_file": "data/processed/splits.csv",
        "total_image_backed_rows": total,
        "partitions": {
            partition: {
                "count": int(partition_counts.get(partition, 0)),
                "percentage": float(partition_counts.get(partition, 0) / total * 100),
            }
            for partition in PARTITIONS
        },
        "rare_article_types": {
            "singleton_groups_train_only": singleton_classes,
            "dual_groups_train_validation": dual_classes,
        },
        "cross_role_exact_duplicates": {
            "hash_groups": len(cross_role_hashes),
            "quarantined_training_rows": cross_role_rows,
            "policy": (
                "all labelled byte-identical twins of official prediction images are quarantined"
            ),
        },
        "conflicting_label_exact_duplicates": {
            "hash_groups": len(conflicts),
            "quarantined_rows": conflict_rows,
            "groups_by_target": conflict_target_counts,
            "policy": (
                "all labelled rows in an exact-SHA group are quarantined when any target has "
                "more than one distinct valid label"
            ),
        },
        "quarantine": {
            "total_rows": quarantine_count,
            "allowed_reasons": [
                "cross-role exact duplicate",
                "conflicting target labels within an exact-SHA group",
            ],
        },
        "development_target_distributions": target_summary,
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest

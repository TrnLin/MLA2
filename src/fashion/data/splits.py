"""Build and validate the project's sole group-safe split and CV fold file."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import (
    AUDIT_DIR,
    CV_FOLD_COUNT,
    CV_FOLD_SUMMARY_JSON,
    DEVELOPMENT_CLASS_SUMMARY_CSV,
    DEVELOPMENT_RATIO,
    HOLDOUT_RATIO,
    PRODUCT_FAMILIES_CSV,
    RANDOM_SEED,
    SPLIT_SUMMARY_JSON,
    SPLITS_CSV,
    TARGET_COLUMNS,
    TAXONOMY_JSON,
)
from fashion.data.families import find_exact_duplicate_label_conflicts
from fashion.data.taxonomy import build_development_taxonomy

PARTITIONS = ("development", "holdout", "quarantine")
PROTECTED_PARTITIONS = ("holdout", "quarantine")
LEGACY_PARTITIONS = ("train", "val", "holdout", "quarantine")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def id_set_digest(values: pd.Series | list[int] | set[int]) -> str:
    """Hash sorted integer IDs, one per line, including the final newline."""
    ids = sorted(int(value) for value in values)
    payload = "".join(f"{item_id}\n" for item_id in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def cv_assignment_digest(frame: pd.DataFrame) -> str:
    """Hash sorted ``development ID,fold`` pairs, one per line."""
    development = frame[frame["partition"].eq("development")].copy()
    folds = _fold_values(development)
    if folds.isna().any():
        raise ValueError("every development row needs a fold before hashing")
    if not folds.mod(1).eq(0).all():
        raise ValueError("development folds must be integers before hashing")
    pairs = sorted(zip(development["id"].astype(int), folds.astype(int), strict=True))
    payload = "".join(f"{item_id},{fold}\n" for item_id, fold in pairs).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _fold_values(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["cv_fold"].replace("", pd.NA), errors="coerce")


def _drop_legacy_label_views(frame: pd.DataFrame) -> pd.DataFrame:
    stale = {
        column
        for column in frame.columns
        if "_deployed" in column
        or "_supported" in column
        or column.endswith("_deployment_status")
        or column.endswith("_evaluation_status")
    }
    return frame.drop(columns=sorted(stale), errors="ignore")


def _validate_zero_crossing(
    frame: pd.DataFrame,
    group_column: str,
    boundary_column: str,
    message: str,
) -> None:
    rows = frame[frame[group_column].astype(str).str.strip().ne("")]
    if (rows.groupby(group_column, dropna=False)[boundary_column].nunique() > 1).any():
        raise ValueError(message)


def validate_split_structure(frame: pd.DataFrame) -> None:
    """Validate membership and sealed structural fields without using target outcomes."""
    required = {
        "id",
        "sha256",
        "duplicate_group",
        "product_name_key",
        "product_family_group",
        "partition",
        "cv_fold",
        "is_cross_role_exact_duplicate",
        "is_cross_role_near_duplicate",
        "has_conflicting_target_labels",
        "conflicting_targets",
        "quarantine_reason",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"split data is missing columns: {sorted(missing)}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("split IDs must be non-null and unique")
    if frame["partition"].isna().any():
        raise ValueError("every split row must have a partition")
    if unknown := set(frame["partition"]) - set(PARTITIONS):
        raise ValueError(f"unknown partitions: {sorted(unknown)}")

    folds = _fold_values(frame)
    development = frame["partition"].eq("development")
    protected = frame["partition"].isin(PROTECTED_PARTITIONS)
    if folds.loc[development].isna().any():
        raise ValueError("every development row must have a cv_fold")
    if not folds.loc[development].dropna().mod(1).eq(0).all():
        raise ValueError("development cv_fold values must be integers")
    if not folds.loc[development].between(0, CV_FOLD_COUNT - 1).all():
        raise ValueError(f"development cv_fold values must be in range({CV_FOLD_COUNT})")
    if folds.loc[protected].notna().any():
        raise ValueError("holdout and quarantine rows cannot have a cv_fold")
    if frame.loc[development, "product_family_group"].nunique() >= CV_FOLD_COUNT:
        observed = set(folds.loc[development].astype(int))
        if observed != set(range(CV_FOLD_COUNT)):
            raise ValueError(f"development must contain all {CV_FOLD_COUNT} cv folds")

    active = frame[frame["partition"].ne("quarantine")]
    _validate_zero_crossing(
        active,
        "sha256",
        "partition",
        "an exact SHA-256 group crosses development and holdout",
    )
    _validate_zero_crossing(
        active,
        "duplicate_group",
        "partition",
        "an exact duplicate group crosses development and holdout",
    )
    _validate_zero_crossing(
        active,
        "product_family_group",
        "partition",
        "a product family crosses development and holdout",
    )
    _validate_zero_crossing(
        active,
        "product_name_key",
        "partition",
        "a normalized product-name block crosses development and holdout",
    )

    development_rows = frame.loc[development].copy()
    development_rows["_cv_fold"] = folds.loc[development].astype(int).to_numpy()
    for column, label in (
        ("sha256", "exact SHA-256"),
        ("duplicate_group", "exact duplicate"),
        ("product_family_group", "product family"),
        ("product_name_key", "normalized product-name"),
    ):
        _validate_zero_crossing(
            development_rows,
            column,
            "_cv_fold",
            f"a {label} group crosses cv folds",
        )

    cross_role = _as_bool(frame["is_cross_role_exact_duplicate"]) | _as_bool(
        frame["is_cross_role_near_duplicate"]
    )
    if not frame.loc[cross_role, "partition"].eq("quarantine").all():
        raise ValueError("a cross-role visual match exists outside quarantine")
    conflict_flags = _as_bool(frame["has_conflicting_target_labels"])
    conflict_names = frame["conflicting_targets"].astype(str).str.strip()
    if not conflict_names.ne("").equals(conflict_flags):
        raise ValueError("sealed conflict flags and target names disagree")
    if not frame.loc[conflict_flags, "partition"].eq("quarantine").all():
        raise ValueError("an exact duplicate with conflicting labels exists outside quarantine")
    has_reason = frame["quarantine_reason"].astype(str).str.strip().ne("")
    if not has_reason.equals(frame["partition"].eq("quarantine")):
        raise ValueError("quarantine membership and approved reasons disagree")


def _validate_protected_targets(
    frame: pd.DataFrame,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> None:
    protected = frame["partition"].isin(PROTECTED_PARTITIONS)
    for target in targets:
        if target not in frame or f"has_{target}_label" not in frame:
            raise ValueError(f"split data is missing raw target contract for {target}")
        if frame.loc[protected, target].astype(str).str.strip().ne("").any():
            raise ValueError(f"protected {target} values must be blank")
        if _as_bool(frame.loc[protected, f"has_{target}_label"]).any():
            raise ValueError(f"protected {target} masks must be false")


def validate_splits(frame: pd.DataFrame) -> None:
    """Validate the normal protected-safe runtime view."""
    validate_split_structure(frame)
    _validate_protected_targets(frame)


def validate_built_splits(
    frame: pd.DataFrame,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> None:
    """Build-time validation before protected target values are blanked."""
    validate_split_structure(frame)
    required = {column for target in targets for column in (target, f"has_{target}_label")}
    if missing := required.difference(frame.columns):
        raise ValueError(f"built split data is missing target columns: {sorted(missing)}")
    development = frame["partition"].eq("development")
    development_rows = frame.loc[development]
    conflicts = find_exact_duplicate_label_conflicts(development_rows, targets)
    expected_conflict = development_rows["sha256"].astype(str).isin(conflicts)
    actual_conflict = _as_bool(development_rows["has_conflicting_target_labels"])
    if not actual_conflict.equals(expected_conflict):
        raise ValueError("conflicting-target flags disagree with exact duplicate labels")
    expected_targets = (
        development_rows["sha256"]
        .astype(str)
        .map(lambda digest: ",".join(conflicts.get(digest, ())))
    )
    if not development_rows["conflicting_targets"].astype(str).eq(expected_targets).all():
        raise ValueError("conflicting-target names disagree with exact duplicate labels")


def _group_inputs(
    frame: pd.DataFrame,
    targets: tuple[str, ...],
) -> tuple[
    dict[str, int],
    dict[str, set[tuple[str, str]]],
    dict[str, dict[tuple[str, str], int]],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
]:
    group_sizes = frame.groupby("product_family_group").size().astype(int).to_dict()
    group_labels: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_label_counts: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    label_rows: dict[tuple[str, str], int] = defaultdict(int)
    for target in targets:
        mask = _as_bool(frame[f"has_{target}_label"])
        for group_id, label in frame.loc[mask, ["product_family_group", target]].itertuples(
            index=False, name=None
        ):
            key = (target, str(label))
            group_id = str(group_id)
            group_labels[group_id].add(key)
            group_label_counts[group_id][key] += 1
            label_rows[key] += 1
    label_groups: dict[tuple[str, str], int] = defaultdict(int)
    for labels in group_labels.values():
        for key in labels:
            label_groups[key] += 1
    return group_sizes, group_labels, group_label_counts, label_rows, label_groups


def _allocate_groups(
    frame: pd.DataFrame,
    bins: tuple[str, ...],
    ratios: dict[str, float],
    seed: int,
    targets: tuple[str, ...],
    preassigned: dict[str, str] | None = None,
) -> dict[str, str]:
    """Deterministically balance family groups, rows, and raw target tokens."""
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("allocation ratios must total 1")
    if set(bins) != set(ratios):
        raise ValueError("allocation bins and ratio keys differ")
    group_sizes, group_labels, group_label_counts, label_rows, label_groups = _group_inputs(
        frame, targets
    )
    group_ids = sorted(group_sizes)
    random = np.random.RandomState(seed)
    random_rank = {group: rank for rank, group in enumerate(random.permutation(group_ids))}
    total_rows = sum(group_sizes.values())
    desired_rows = {name: max(total_rows * ratios[name], 1.0) for name in bins}
    desired_labels = {
        key: {name: max(count * ratios[name], 1.0) for name in bins}
        for key, count in label_rows.items()
    }
    allocation: dict[str, str] = {}
    rows_by_bin = {name: 0 for name in bins}
    label_counts = {key: {name: 0 for name in bins} for key in label_rows}

    def assign(group: str, name: str) -> None:
        allocation[group] = name
        rows_by_bin[name] += group_sizes[group]
        for key in group_labels[group]:
            label_counts[key][name] += group_label_counts[group][key]

    for group, name in sorted((preassigned or {}).items()):
        if group not in group_sizes or name not in bins:
            raise ValueError("invalid preassigned group allocation")
        assign(group, name)

    remaining = [group for group in group_ids if group not in allocation]
    remaining.sort(
        key=lambda group: (
            min((label_groups[key] for key in group_labels[group]), default=10**9),
            -len(group_labels[group]),
            -group_sizes[group],
            random_rank[group],
            group,
        )
    )
    for group in remaining:
        scores: dict[str, tuple[float, float, str]] = {}
        for name in bins:
            new_rows = rows_by_bin[name] + group_sizes[group]
            row_load = new_rows / desired_rows[name]
            class_loads = [
                (label_counts[key][name] + group_label_counts[group][key])
                / desired_labels[key][name]
                for key in group_labels[group]
            ]
            class_load = float(np.mean(class_loads)) if class_loads else row_load
            overfill = max(0.0, new_rows - desired_rows[name]) / desired_rows[name]
            score = 0.60 * row_load + 0.35 * class_load + 0.05 * overfill
            scores[name] = (score, rows_by_bin[name] / desired_rows[name], name)
        assign(group, min(bins, key=lambda name: scores[name]))
    return allocation


def _assign_cv_folds(
    frame: pd.DataFrame,
    seed: int,
    targets: tuple[str, ...],
) -> pd.Series:
    development = frame[frame["partition"].eq("development")].copy()
    bins = tuple(str(index) for index in range(CV_FOLD_COUNT))
    ratios = {name: 1.0 / CV_FOLD_COUNT for name in bins}
    allocation = _allocate_groups(development, bins, ratios, seed, targets)
    result = pd.Series("", index=frame.index, dtype=object)
    result.loc[development.index] = development["product_family_group"].map(allocation).astype(int)
    return result


def migrate_legacy_split_frame(
    frame: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Preserve old membership while replacing train/val with development folds."""
    if set(frame["partition"]) - set(LEGACY_PARTITIONS):
        raise ValueError("legacy migration received unknown partition values")
    migrated = _drop_legacy_label_views(frame.copy())
    migrated.loc[migrated["partition"].isin({"train", "val"}), "partition"] = "development"
    migrated["cv_fold"] = _assign_cv_folds(migrated, seed, targets)
    validate_splits(migrated)
    return migrated


def _initial_partition_allocation(
    frame: pd.DataFrame,
    seed: int,
    targets: tuple[str, ...],
) -> pd.Series:
    active = frame[frame["quarantine_reason"].astype(str).str.strip().eq("")].copy()
    _, group_labels, _, _, label_groups = _group_inputs(active, targets)
    must_develop = {
        group
        for group, labels in group_labels.items()
        if any(label_groups[key] <= 1 for key in labels)
    }
    preassigned = {group: "development" for group in must_develop}
    allocation = _allocate_groups(
        active,
        ("development", "holdout"),
        {"development": DEVELOPMENT_RATIO, "holdout": HOLDOUT_RATIO},
        seed,
        targets,
        preassigned,
    )
    result = pd.Series("quarantine", index=frame.index, dtype=object)
    result.loc[active.index] = active["product_family_group"].map(allocation)
    return result


def _attach_structural_groups(
    manifest: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    families: pd.DataFrame,
) -> pd.DataFrame:
    if set(manifest["id"].astype(int)) != set(families["id"].astype(int)):
        raise ValueError("product-family evidence does not cover the labelled manifest")
    result = manifest.merge(families, on="id", how="left", validate="one_to_one")
    duplicate_map = (
        duplicate_groups.set_index("sha256")["duplicate_group"].to_dict()
        if not duplicate_groups.empty
        else {}
    )
    result["sha256"] = result["sha256"].astype(str)
    result["duplicate_group"] = result["sha256"].map(duplicate_map)
    missing = result["duplicate_group"].isna()
    result.loc[missing, "duplicate_group"] = result.loc[missing, "sha256"].map(
        lambda digest: f"single_{digest[:12]}"
    )
    result["is_cross_role_duplicate"] = _as_bool(
        result["is_cross_role_exact_duplicate"]
    ) | _as_bool(result["is_cross_role_near_duplicate"])
    result["quarantine_reason"] = result["pre_quarantine_reason"].astype(str)
    return _drop_legacy_label_views(result)


def _apply_existing_membership(
    manifest: pd.DataFrame,
    existing: pd.DataFrame,
    seed: int,
    targets: tuple[str, ...],
    *,
    refreeze_development_folds: bool = False,
) -> pd.DataFrame:
    if set(existing["partition"]).issubset(set(LEGACY_PARTITIONS)):
        existing = migrate_legacy_split_frame(existing, seed=seed, targets=targets)
    else:
        validate_splits(existing)
    if set(manifest["id"].astype(int)) != set(existing["id"].astype(int)):
        raise ValueError("canonical split IDs differ from the rebuilt image-backed manifest")
    sealed_columns = [
        "product_family_group",
        "family_group_basis",
        "has_conflicting_target_labels",
        "conflicting_targets",
        "pre_quarantine_reason",
        "quarantine_reason",
    ]
    protected_existing = existing.loc[
        existing["partition"].isin(PROTECTED_PARTITIONS), ["id", *sealed_columns]
    ].set_index("id")
    protected_ids = manifest["id"].isin(protected_existing.index)
    for column in sealed_columns:
        manifest.loc[protected_ids, column] = manifest.loc[protected_ids, "id"].map(
            protected_existing[column]
        )
    new_quarantine = set(
        manifest.loc[manifest["quarantine_reason"].astype(str).str.strip().ne(""), "id"].astype(int)
    )
    old_quarantine = set(existing.loc[existing["partition"].eq("quarantine"), "id"].astype(int))
    if new_quarantine != old_quarantine:
        raise ValueError(
            "quarantine membership changed; review before replacing the canonical split"
        )
    existing_family = existing.set_index("id")["product_family_group"].astype(str)
    rebuilt_family = manifest.set_index("id")["product_family_group"].astype(str)
    development_ids = set(existing.loc[existing["partition"].eq("development"), "id"].astype(int))
    changed_development_ids = sorted(
        item_id
        for item_id in development_ids
        if existing_family.loc[item_id] != rebuilt_family.loc[item_id]
    )
    if changed_development_ids:
        if not refreeze_development_folds:
            raise ValueError(
                "development family contract changed for "
                f"{len(changed_development_ids)} rows; explicitly refreeze CV folds before "
                "training with refreeze_development_folds=True"
            )

    membership = existing[["id", "partition", "cv_fold"]]
    result = manifest.drop(columns=["partition", "cv_fold"], errors="ignore").merge(
        membership,
        on="id",
        how="left",
        validate="one_to_one",
    )
    if changed_development_ids:
        result["cv_fold"] = _assign_cv_folds(result, seed, targets)
    return result


def write_protected_safe_splits(
    frame: pd.DataFrame,
    output_csv: str | Path,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Persist the sole split with protected target cells blank and masks false."""
    safe = _drop_legacy_label_views(frame.copy())
    protected = safe["partition"].isin(PROTECTED_PARTITIONS)
    for target in targets:
        safe.loc[protected, target] = ""
        safe.loc[protected, f"has_{target}_label"] = False
    development = safe["partition"].eq("development")
    fold_numbers = pd.to_numeric(safe["cv_fold"], errors="coerce")
    safe["cv_fold"] = pd.Series("", index=safe.index, dtype=object)
    safe.loc[development, "cv_fold"] = fold_numbers.loc[development].astype(int).astype(str)
    safe.sort_values("id", inplace=True)
    validate_splits(safe)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe.to_csv(output, index=False, lineterminator="\n")
    return safe


def _development_class_rows(
    manifest: pd.DataFrame,
    targets: tuple[str, ...],
) -> list[dict[str, Any]]:
    development = manifest[manifest["partition"].eq("development")].copy()
    folds = _fold_values(development).astype(int)
    development["_cv_fold"] = folds
    output: list[dict[str, Any]] = []
    for target in targets:
        valid = _as_bool(development[f"has_{target}_label"])
        labels = sorted(development.loc[valid, target].astype(str).unique())
        for label in labels:
            class_rows = development.loc[valid & development[target].astype(str).eq(label)]
            record: dict[str, Any] = {
                "target": target,
                "class": label,
                "source_scope": "development",
                "development_product_count": int(len(class_rows)),
                "development_family_count": int(class_rows["product_family_group"].nunique()),
                "folds_present": ",".join(
                    map(str, sorted(class_rows["_cv_fold"].astype(int).unique()))
                ),
                "fold_count": int(class_rows["_cv_fold"].nunique()),
            }
            untrainable = 0
            training_products: list[int] = []
            training_families: list[int] = []
            for fold in range(CV_FOLD_COUNT):
                validation_count = int(class_rows["_cv_fold"].eq(fold).sum())
                training = class_rows[class_rows["_cv_fold"].ne(fold)]
                training_count = int(len(training))
                family_count = int(training["product_family_group"].nunique())
                record[f"fold_{fold}_validation_products"] = validation_count
                record[f"fold_{fold}_training_products"] = training_count
                record[f"fold_{fold}_training_families"] = family_count
                training_products.append(training_count)
                training_families.append(family_count)
                untrainable += int(training_count == 0)
            record["minimum_fold_training_products"] = min(training_products)
            record["minimum_fold_training_families"] = min(training_families)
            record["untrainable_fold_count"] = untrainable
            record["rare_warning"] = (
                "untrainable_in_one_or_more_folds"
                if untrainable
                else "low_independent_family_support"
                if record["development_family_count"] < CV_FOLD_COUNT
                else ""
            )
            output.append(record)
    return output


def write_split_public_evidence(
    manifest: pd.DataFrame,
    taxonomy_policy: dict[str, Any],
    summary_output: str | Path,
    cv_summary_output: str | Path,
    development_summary_output: str | Path,
    *,
    seed: int = RANDOM_SEED,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> dict[str, Any]:
    """Write development-only fold and class evidence without protected outcomes."""
    validate_split_structure(manifest)
    counts = manifest["partition"].value_counts()
    total = len(manifest)
    active = manifest[manifest["partition"].ne("quarantine")]
    development = manifest[manifest["partition"].eq("development")].copy()
    development["_cv_fold"] = _fold_values(development).astype(int)
    fold_rows: dict[str, Any] = {}
    for fold in range(CV_FOLD_COUNT):
        validation = development[development["_cv_fold"].eq(fold)]
        training = development[development["_cv_fold"].ne(fold)]
        target_coverage: dict[str, Any] = {}
        for target in targets:
            train_valid = _as_bool(training[f"has_{target}_label"])
            val_valid = _as_bool(validation[f"has_{target}_label"])
            train_labels = set(training.loc[train_valid, target].astype(str))
            val_labels = set(validation.loc[val_valid, target].astype(str))
            target_coverage[target] = {
                "training_class_count": len(train_labels),
                "validation_class_count": len(val_labels),
                "validation_classes_absent_from_training": sorted(val_labels - train_labels),
            }
        fold_rows[str(fold)] = {
            "validation_products": int(len(validation)),
            "training_products": int(len(training)),
            "validation_families": int(validation["product_family_group"].nunique()),
            "training_families": int(training["product_family_group"].nunique()),
            "target_coverage": target_coverage,
        }
    largest_family = int(development.groupby("product_family_group").size().max())
    cv_summary = {
        "schema_version": "2.0.0",
        "source_scope": "development",
        "fold_count": CV_FOLD_COUNT,
        "seed": seed,
        "allocation_unit": "product_family_group",
        "ideal_validation_products": len(development) / CV_FOLD_COUNT,
        "maximum_allowed_size_deviation": largest_family,
        "folds": fold_rows,
        "rare_class_policy": "report_missing_training_support_without_dropping_rows",
        "cv_assignment_sha256": cv_assignment_digest(manifest),
    }
    Path(cv_summary_output).write_text(
        json.dumps(cv_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    class_rows = _development_class_rows(manifest, targets)
    pd.DataFrame(class_rows).sort_values(["target", "class"]).to_csv(
        development_summary_output, index=False, lineterminator="\n"
    )
    summary = {
        "schema_version": "9.0.0",
        "seed": seed,
        "split_file": "data/processed/splits.csv",
        "cv_fold_file": "data/processed/splits.csv",
        "persisted_protected_targets": "blank_with_false_masks",
        "total_image_backed_rows": total,
        "partitions": {
            partition: {
                "count": int(counts.get(partition, 0)),
                "percentage_of_image_backed_rows": float(counts.get(partition, 0) / total * 100),
            }
            for partition in PARTITIONS
        },
        "ideal_one_fold_round": {
            "training_products": len(development) * (CV_FOLD_COUNT - 1) / CV_FOLD_COUNT,
            "validation_products": len(development) / CV_FOLD_COUNT,
            "holdout_products": int(counts.get("holdout", 0)),
            "note": "group allocation makes fold counts close to, not exactly equal to, the ideal",
        },
        "id_set_sha256": {
            partition: id_set_digest(manifest.loc[manifest["partition"].eq(partition), "id"])
            for partition in PARTITIONS
        },
        "cv_assignment_sha256": cv_assignment_digest(manifest),
        "development_union_policy": "preserved legacy train plus validation IDs",
        "fold_policy": cv_summary,
        "group_safety": {
            "active_family_groups": int(active["product_family_group"].nunique()),
            "largest_development_family": largest_family,
            "partition_crossings": 0,
            "cv_fold_crossings": 0,
        },
        "taxonomy": taxonomy_policy,
        "protected_target_policy": {
            "partitions": list(PROTECTED_PARTITIONS),
            "public_target_aggregates": 0,
            "unlock_stage": "notebook_06_after_method_freeze",
        },
        "quarantine": {
            "total_rows": int(counts.get("quarantine", 0)),
            "allowed_reasons": sorted(
                manifest.loc[manifest["partition"].eq("quarantine"), "quarantine_reason"]
                .astype(str)
                .unique()
            ),
        },
    }
    Path(summary_output).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def refresh_fixed_split_public_artifacts(
    splits_csv: str | Path,
    summary_output: str | Path,
    cv_summary_output: str | Path,
    taxonomy_output: str | Path,
    development_summary_output: str | Path,
    *,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Migrate if needed, then refresh only development-safe public files."""
    splits_path = Path(splits_csv)
    manifest = pd.read_csv(splits_path, keep_default_na=False)
    if set(manifest["partition"]).issubset(set(LEGACY_PARTITIONS)):
        manifest = migrate_legacy_split_frame(manifest, seed=seed, targets=targets)
    else:
        validate_splits(manifest)
    taxonomy_policy = build_development_taxonomy(manifest, targets=targets)
    Path(taxonomy_output).write_text(
        json.dumps(taxonomy_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_split_public_evidence(
        manifest,
        taxonomy_policy,
        summary_output,
        cv_summary_output,
        development_summary_output,
        seed=seed,
        targets=targets,
    )
    return write_protected_safe_splits(manifest, splits_path, targets)


def make_splits(
    train_manifest_csv: str | Path | None = None,
    duplicate_groups_csv: str | Path = AUDIT_DIR / "exact_duplicate_groups.csv",
    product_families_csv: str | Path = PRODUCT_FAMILIES_CSV,
    output_csv: str | Path = SPLITS_CSV,
    summary_output: str | Path = SPLIT_SUMMARY_JSON,
    cv_summary_output: str | Path = CV_FOLD_SUMMARY_JSON,
    taxonomy_output: str | Path = TAXONOMY_JSON,
    development_summary_output: str | Path = DEVELOPMENT_CLASS_SUMMARY_CSV,
    seed: int = RANDOM_SEED,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    *,
    initialize_split: bool = False,
    refreeze_development_folds: bool = False,
) -> pd.DataFrame:
    """Build structural evidence while preserving canonical protected membership."""
    if train_manifest_csv is None:
        raise ValueError("the labelled manifest must be an explicit temporary build path")
    manifest = pd.read_csv(train_manifest_csv, keep_default_na=False)
    families = pd.read_csv(product_families_csv, keep_default_na=False)
    duplicate_groups = pd.read_csv(duplicate_groups_csv, keep_default_na=False)
    manifest = _attach_structural_groups(manifest, duplicate_groups, families)
    output_path = Path(output_csv)
    if output_path.is_file():
        existing = pd.read_csv(output_path, keep_default_na=False)
        manifest = _apply_existing_membership(
            manifest,
            existing,
            seed,
            targets,
            refreeze_development_folds=refreeze_development_folds,
        )
    elif initialize_split:
        manifest["partition"] = _initial_partition_allocation(manifest, seed, targets)
        manifest["cv_fold"] = _assign_cv_folds(manifest, seed, targets)
    else:
        raise FileNotFoundError(
            "canonical splits.csv is missing; routine preparation refuses to recreate protected "
            "membership. Tests or a reviewed initialization must pass initialize_split=True."
        )

    validate_built_splits(manifest, targets)
    manifest.sort_values("id", inplace=True)
    taxonomy_policy = build_development_taxonomy(manifest, targets=targets)
    for path in (
        output_path,
        Path(summary_output),
        Path(cv_summary_output),
        Path(taxonomy_output),
        Path(development_summary_output),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_split_public_evidence(
        manifest,
        taxonomy_policy,
        summary_output,
        cv_summary_output,
        development_summary_output,
        seed=seed,
        targets=targets,
    )
    Path(taxonomy_output).write_text(
        json.dumps(taxonomy_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_protected_safe_splits(manifest, output_path, targets)
    return manifest

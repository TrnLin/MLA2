"""Build and validate the project's sole product-family-safe split file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import (
    AUDIT_DIR,
    DEVELOPMENT_CLASS_SUMMARY_CSV,
    HOLDOUT_RATIO,
    MIN_SUPPORTED_GROUPS,
    PRODUCT_FAMILIES_CSV,
    RANDOM_SEED,
    SPLIT_SUMMARY_JSON,
    SPLITS_CSV,
    TARGET_COLUMNS,
    TAXONOMY_JSON,
    TRAIN_ONLY_EDA_EXAMPLE_IDS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from fashion.data.families import find_exact_duplicate_label_conflicts
from fashion.data.taxonomy import apply_deployment_taxonomy, validate_built_taxonomy_coverage

PARTITIONS = ("train", "val", "holdout", "quarantine")
MODEL_PARTITIONS = ("train", "val", "holdout")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def validate_split_structure(frame: pd.DataFrame) -> None:
    """Validate membership and sealed audit fields without reading target values."""
    required = {
        "id",
        "sha256",
        "duplicate_group",
        "product_name_key",
        "product_family_group",
        "partition",
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
    if (frame.groupby("sha256")["partition"].nunique() > 1).any():
        raise ValueError("an exact SHA-256 duplicate group crosses partitions")
    if (frame.groupby("duplicate_group")["partition"].nunique() > 1).any():
        raise ValueError("an exact duplicate group crosses partitions")
    if (frame.groupby("product_family_group")["partition"].nunique() > 1).any():
        raise ValueError("a product family group crosses partitions")

    active_names = frame[
        frame["partition"].ne("quarantine") & frame["product_name_key"].astype(str).ne("")
    ]
    if (active_names.groupby("product_name_key")["partition"].nunique() > 1).any():
        raise ValueError("an accepted normalized product-name block crosses model partitions")

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

    has_reason = frame["quarantine_reason"].astype(str).ne("")
    is_quarantine = frame["partition"].eq("quarantine")
    if not has_reason.equals(is_quarantine):
        raise ValueError("quarantine membership and approved reasons disagree")


def validate_splits(frame: pd.DataFrame) -> None:
    """Runtime-safe alias for structural split validation."""
    validate_split_structure(frame)


def validate_built_splits(
    frame: pd.DataFrame,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> None:
    """Build-time only: validate target-derived flags and taxonomy coverage."""
    validate_split_structure(frame)
    required: set[str] = set()
    for target in targets:
        required.update(
            {
                target,
                f"has_{target}_label",
                f"{target}_deployed",
                f"has_{target}_deployed_label",
                f"{target}_supported",
                f"has_{target}_supported_label",
            }
        )
    if missing := required.difference(frame.columns):
        raise ValueError(f"built split data is missing target columns: {sorted(missing)}")

    conflicts = find_exact_duplicate_label_conflicts(frame, targets)
    expected_conflict = frame["sha256"].astype(str).isin(conflicts)
    if not _as_bool(frame["has_conflicting_target_labels"]).equals(expected_conflict):
        raise ValueError("conflicting-target flags disagree with exact duplicate labels")
    expected_targets = (
        frame["sha256"].astype(str).map(lambda digest: ",".join(conflicts.get(digest, ())))
    )
    if not frame["conflicting_targets"].astype(str).eq(expected_targets).all():
        raise ValueError("conflicting-target names disagree with exact duplicate labels")
    validate_built_taxonomy_coverage(frame, targets)


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
    """Collect raw-label balance inputs before model partitions are sealed."""
    active = frame[frame["partition"].ne("quarantine")]
    group_sizes = active.groupby("product_family_group").size().astype(int).to_dict()
    group_labels: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_label_counts: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    label_rows: dict[tuple[str, str], int] = defaultdict(int)
    for target in targets:
        mask = _as_bool(active[f"has_{target}_label"])
        for row in active.loc[mask, ["product_family_group", target]].itertuples(index=False):
            group_id = str(row[0])
            key = (target, str(row[1]))
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
    seed: int,
    ratios: dict[str, float],
    targets: tuple[str, ...],
    minimum_groups: int,
    required_train_ids: tuple[int, ...],
) -> dict[str, str]:
    """Allocate family groups once with raw-label stratification before sealing."""
    (
        group_sizes,
        group_labels,
        group_label_counts,
        label_rows,
        label_groups,
    ) = _group_inputs(frame, targets)
    group_ids = sorted(group_sizes)
    random = np.random.RandomState(seed)
    random_rank = {group: rank for rank, group in enumerate(random.permutation(group_ids))}
    desired_rows = {
        partition: sum(group_sizes.values()) * ratio for partition, ratio in ratios.items()
    }
    desired_labels = {
        key: {partition: count * ratio for partition, ratio in ratios.items()}
        for key, count in label_rows.items()
    }
    allocation: dict[str, str] = {}
    rows_by_partition = {partition: 0 for partition in ratios}
    label_presence = {key: {partition: 0 for partition in ratios} for key in label_groups}
    label_counts = {key: {partition: 0 for partition in ratios} for key in label_rows}

    def assign(group: str, partition: str) -> None:
        previous = allocation.get(group)
        if previous == partition:
            return
        if previous is not None:
            rows_by_partition[previous] -= group_sizes[group]
            for key in group_labels[group]:
                label_presence[key][previous] -= 1
                label_counts[key][previous] -= group_label_counts[group][key]
        allocation[group] = partition
        rows_by_partition[partition] += group_sizes[group]
        for key in group_labels[group]:
            label_presence[key][partition] += 1
            label_counts[key][partition] += group_label_counts[group][key]

    requested_ids = set(map(int, required_train_ids))
    requested_rows = frame[frame["id"].astype(int).isin(requested_ids)]
    if requested_rows["partition"].eq("quarantine").any():
        raise ValueError("a required train-only evidence ID is quarantined")
    required_train_groups = set(
        requested_rows.loc[
            requested_rows["partition"].ne("quarantine"), "product_family_group"
        ].astype(str)
    )
    for group in sorted(required_train_groups):
        assign(group, "train")

    # This is pre-seal stratification only. Taxonomy is fitted later from train.
    for key in sorted(label_groups, key=lambda item: (label_groups[item], item)):
        coverage_partitions = (
            ("holdout", "val", "train") if label_groups[key] >= minimum_groups else ("train",)
        )
        for partition in coverage_partitions:
            if label_presence[key][partition] > 0:
                continue
            candidates = [
                group
                for group in group_ids
                if group not in allocation and key in group_labels[group]
            ]
            if candidates:

                def coverage_score(group: str) -> tuple[int, int, int]:
                    newly_covered = sum(
                        label_presence[label][partition] == 0 for label in group_labels[group]
                    )
                    return -newly_covered, group_sizes[group], random_rank[group]

                assign(min(candidates, key=coverage_score), partition)
                continue

            movable = []
            for group, source in allocation.items():
                if key not in group_labels[group] or source == partition:
                    continue
                if group in required_train_groups:
                    continue
                if all(label_presence[label][source] > 1 for label in group_labels[group]):
                    movable.append(group)
            if not movable:
                raise ValueError(f"cannot independently cover required class {key}")
            assign(
                min(
                    movable,
                    key=lambda group: (
                        abs(
                            (rows_by_partition[partition] + group_sizes[group])
                            - desired_rows[partition]
                        ),
                        group_sizes[group],
                        random_rank[group],
                    ),
                ),
                partition,
            )

    remaining = [group for group in group_ids if group not in allocation]
    remaining.sort(
        key=lambda group: (
            min((label_groups[key] for key in group_labels[group]), default=10**9),
            -group_sizes[group],
            random_rank[group],
        )
    )
    for group in remaining:
        scores: dict[str, float] = {}
        for partition in ratios:
            new_rows = rows_by_partition[partition] + group_sizes[group]
            row_load = new_rows / desired_rows[partition]
            class_loads = []
            for key in group_labels[group]:
                desired = max(desired_labels[key][partition], 1.0)
                new_count = label_counts[key][partition] + group_label_counts[group][key]
                class_loads.append(new_count / desired)
            class_load = float(np.mean(class_loads)) if class_loads else row_load
            overfill = max(0.0, new_rows - desired_rows[partition]) / desired_rows[partition]
            scores[partition] = 0.60 * row_load + 0.35 * class_load + 0.05 * overfill
        chosen = min(
            scores,
            key=lambda partition: (
                scores[partition],
                rows_by_partition[partition] / desired_rows[partition],
                partition,
            ),
        )
        assign(group, chosen)
    return allocation


def _taxonomy_sensitivity(
    frame: pd.DataFrame,
    targets: tuple[str, ...],
) -> dict[str, Any]:
    active = frame[frame["partition"].eq("train")]
    result: dict[str, Any] = {
        "fit_scope": "train_only",
        "fit_partition": "train",
        "support_unit": "distinct_product_family_group",
        "targets": {},
    }
    for target in targets:
        mask = _as_bool(active[f"has_{target}_label"])
        support = (
            active.loc[mask, ["product_family_group", target]]
            .drop_duplicates()
            .groupby(target)["product_family_group"]
            .nunique()
        )
        result["targets"][target] = {
            str(threshold): {
                "kept_classes": int(support.ge(threshold).sum()),
                "excluded_classes": int(support.lt(threshold).sum()),
            }
            for threshold in (2, 3, 4, 5)
        }
    return result


def write_protected_safe_splits(
    frame: pd.DataFrame,
    output_csv: str | Path,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Persist the sole split with protected target cells blank and masks false."""
    safe = frame.copy()
    protected = safe["partition"].isin({"holdout", "quarantine"})
    for target in targets:
        for column in (target, f"{target}_deployed", f"{target}_supported"):
            if column in safe:
                safe.loc[protected, column] = ""
        for column in (
            f"has_{target}_label",
            f"has_{target}_deployed_label",
            f"has_{target}_supported_label",
        ):
            if column in safe:
                safe.loc[protected, column] = False
        for column in (f"{target}_deployment_status", f"{target}_evaluation_status"):
            if column in safe:
                safe.loc[protected, column] = "protected"
    safe.sort_values("id", inplace=True)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe.to_csv(output, index=False)
    return safe


def write_split_public_evidence(
    manifest: pd.DataFrame,
    taxonomy_policy: dict[str, Any],
    summary_output: str | Path,
    development_summary_output: str | Path,
    *,
    seed: int = RANDOM_SEED,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    minimum_groups: int = MIN_SUPPORTED_GROUPS,
    required_train_ids: tuple[int, ...] = TRAIN_ONLY_EDA_EXAMPLE_IDS,
) -> dict[str, Any]:
    """Write split evidence using structure plus train/validation targets only."""
    development_rows: list[dict[str, Any]] = []
    target_summary: dict[str, Any] = {}
    for target in targets:
        target_summary[target] = {}
        expected_classes = taxonomy_policy["targets"][target]["supported_classes"]
        for partition in ("train", "val"):
            rows = manifest[manifest["partition"].eq(partition)]
            mask = _as_bool(rows[f"has_{target}_supported_label"])
            counts = rows.loc[mask, f"{target}_supported"].value_counts()
            absent = sorted(set(expected_classes).difference(counts.index.astype(str)))
            target_summary[target][partition] = {
                "valid_count": int(mask.sum()),
                "num_classes": len(counts),
                "missing_supported_classes": absent,
            }
            for label, count in counts.items():
                development_rows.append(
                    {
                        "target": target,
                        "class": label,
                        "partition": partition,
                        "count": int(count),
                        "scope": "supported_primary_metric_slice",
                    }
                )
        target_summary[target]["protected_partitions"] = {
            "partitions": ["holdout", "quarantine"],
            "status": "target outcomes are not aggregated before final evaluation",
        }
    pd.DataFrame(development_rows).to_csv(development_summary_output, index=False)

    partition_counts = manifest["partition"].value_counts()
    total = len(manifest)
    active_frame = manifest[manifest["partition"].ne("quarantine")]
    conflict_rows = _as_bool(manifest["has_conflicting_target_labels"])
    summary = {
        "schema_version": "7.0.0",
        "seed": seed,
        "split_file": "data/processed/splits.csv",
        "persisted_protected_targets": "redacted",
        "total_image_backed_rows": total,
        "partitions": {
            partition: {
                "count": int(partition_counts.get(partition, 0)),
                "percentage": float(partition_counts.get(partition, 0) / total * 100),
            }
            for partition in PARTITIONS
        },
        "product_family_policy": {
            "active_family_groups": int(active_frame["product_family_group"].nunique()),
            "largest_active_family_group": int(
                active_frame.groupby("product_family_group").size().max()
            ),
            "family_groups_crossing_model_partitions": int(
                (active_frame.groupby("product_family_group")["partition"].nunique() > 1).sum()
            ),
            "normalized_name_keys_crossing_model_partitions": int(
                (
                    active_frame.loc[active_frame["product_name_key"].ne("")]
                    .groupby("product_name_key")["partition"]
                    .nunique()
                    > 1
                ).sum()
            ),
            "exact_sha_groups_crossing_partitions": int(
                (manifest.groupby("sha256")["partition"].nunique() > 1).sum()
            ),
            "approved_for_model_comparison": True,
            "predeclared_train_only_evidence": {
                "requested_ids": list(map(int, required_train_ids)),
                "applied_ids": sorted(
                    set(map(int, required_train_ids)).intersection(manifest["id"].astype(int))
                ),
                "selection_basis": (
                    "IDs fixed before allocation for required train-only visual evidence; "
                    "no validation or holdout outcome selects examples"
                ),
            },
        },
        "deployment_taxonomy": taxonomy_policy,
        "taxonomy_sensitivity": _taxonomy_sensitivity(manifest, targets),
        "taxonomy_fit": {
            "partition": "train",
            "minimum_independent_family_groups": minimum_groups,
            "sensitivity_scope": "train_only",
            "holdout_target_coverage_audited": False,
        },
        "quarantine": {
            "total_rows": int(partition_counts.get("quarantine", 0)),
            "cross_role_exact_rows": int(_as_bool(manifest["is_cross_role_exact_duplicate"]).sum()),
            "cross_role_near_rows_conservatively_quarantined": int(
                _as_bool(manifest["is_cross_role_near_duplicate"]).sum()
            ),
            "conflicting_exact_sha_groups": int(
                manifest.loc[conflict_rows, "sha256"].astype(str).nunique()
            ),
            "conflicting_exact_sha_rows": int(conflict_rows.sum()),
            "allowed_reasons": [
                "cross_role_exact_duplicate",
                "cross_role_near_duplicate_conservative_quarantine",
                "conflicting_labels_exact_sha",
            ],
        },
        "development_target_distributions": target_summary,
    }
    Path(summary_output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def refresh_fixed_split_public_artifacts(
    splits_csv: str | Path,
    summary_output: str | Path,
    taxonomy_output: str | Path,
    development_summary_output: str | Path,
    *,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    minimum_groups: int = MIN_SUPPORTED_GROUPS,
) -> pd.DataFrame:
    """Refit train-only mappings on fixed partitions and rewrite only safe public files."""
    manifest = pd.read_csv(splits_csv, keep_default_na=False)
    validate_split_structure(manifest)
    manifest, taxonomy_policy = apply_deployment_taxonomy(
        manifest,
        group_column="product_family_group",
        targets=targets,
        minimum_groups=minimum_groups,
    )
    Path(taxonomy_output).write_text(
        json.dumps(taxonomy_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_split_public_evidence(
        manifest,
        taxonomy_policy,
        summary_output,
        development_summary_output,
        targets=targets,
        minimum_groups=minimum_groups,
    )
    return write_protected_safe_splits(manifest, splits_csv, targets)


def make_splits(
    train_manifest_csv: str | Path | None = None,
    duplicate_groups_csv: str | Path = AUDIT_DIR / "exact_duplicate_groups.csv",
    product_families_csv: str | Path = PRODUCT_FAMILIES_CSV,
    output_csv: str | Path = SPLITS_CSV,
    summary_output: str | Path = SPLIT_SUMMARY_JSON,
    taxonomy_output: str | Path = TAXONOMY_JSON,
    development_summary_output: str | Path = DEVELOPMENT_CLASS_SUMMARY_CSV,
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    holdout_ratio: float = HOLDOUT_RATIO,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    minimum_groups: int = MIN_SUPPORTED_GROUPS,
    required_train_ids: tuple[int, ...] = TRAIN_ONLY_EDA_EXAMPLE_IDS,
) -> pd.DataFrame:
    """Allocate groups, then fit and freeze the deployment taxonomy from train."""
    if not np.isclose(train_ratio + validation_ratio + holdout_ratio, 1.0):
        raise ValueError("train, validation, and holdout ratios must total 1")
    if train_manifest_csv is None:
        raise ValueError("the labelled manifest must be an explicit temporary build path")
    manifest = pd.read_csv(train_manifest_csv, keep_default_na=False)
    families = pd.read_csv(product_families_csv, keep_default_na=False)
    if set(manifest["id"].astype(int)) != set(families["id"].astype(int)):
        raise ValueError("product-family evidence does not cover the labelled manifest")
    manifest = manifest.merge(families, on="id", how="left", validate="one_to_one")

    duplicate_groups = pd.read_csv(duplicate_groups_csv, keep_default_na=False)
    duplicate_map = (
        duplicate_groups.set_index("sha256")["duplicate_group"].to_dict()
        if not duplicate_groups.empty
        else {}
    )
    manifest["sha256"] = manifest["sha256"].astype(str)
    manifest["duplicate_group"] = manifest["sha256"].map(duplicate_map)
    missing_group = manifest["duplicate_group"].isna()
    manifest.loc[missing_group, "duplicate_group"] = manifest.loc[missing_group, "sha256"].map(
        lambda digest: f"single_{digest[:12]}"
    )
    manifest["is_cross_role_duplicate"] = _as_bool(
        manifest["is_cross_role_exact_duplicate"]
    ) | _as_bool(manifest["is_cross_role_near_duplicate"])
    manifest["quarantine_reason"] = manifest["pre_quarantine_reason"].astype(str)
    manifest["partition"] = ""
    manifest.loc[manifest["quarantine_reason"].ne(""), "partition"] = "quarantine"

    ratios = {"train": train_ratio, "val": validation_ratio, "holdout": holdout_ratio}
    allocation = _allocate_groups(
        manifest,
        seed,
        ratios,
        targets,
        minimum_groups,
        required_train_ids,
    )
    active = manifest["partition"].ne("quarantine")
    manifest.loc[active, "partition"] = manifest.loc[active, "product_family_group"].map(allocation)
    manifest, taxonomy_policy = apply_deployment_taxonomy(
        manifest,
        group_column="product_family_group",
        targets=targets,
        minimum_groups=minimum_groups,
    )
    validate_built_splits(manifest, targets)
    manifest.sort_values("id", inplace=True)

    output_csv = Path(output_csv)
    summary_output = Path(summary_output)
    taxonomy_output = Path(taxonomy_output)
    development_summary_output = Path(development_summary_output)
    for path in (output_csv, summary_output, taxonomy_output, development_summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_protected_safe_splits(manifest, output_csv, targets)
    taxonomy_output.write_text(
        json.dumps(taxonomy_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_split_public_evidence(
        manifest,
        taxonomy_policy,
        summary_output,
        development_summary_output,
        seed=seed,
        targets=targets,
        minimum_groups=minimum_groups,
        required_train_ids=required_train_ids,
    )
    return manifest

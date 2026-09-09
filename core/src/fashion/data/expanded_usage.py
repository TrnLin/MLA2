"""Combine teacher data and admitted rare-class images for the same Usage task."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import CV_FOLD_COUNT, RANDOM_SEED, ROOT, TARGET_COLUMNS
from fashion.data.external_usage import validate_external_manifest
from fashion.data.families import normalize_product_name
from fashion.data.splits import validate_splits

DATASET_NAME = "teacher_plus_rare_usage_20260906"
EXTERNAL_ID_START = 1_000_000_000


def assign_added_folds(frame: pd.DataFrame, *, seed: int = RANDOM_SEED) -> pd.Series:
    """Balance added label counts across five folds without splitting a product family."""
    assignments: dict[str, int] = {}
    totals = [0] * CV_FOLD_COUNT
    families = frame.groupby("external_group_id", sort=True).agg(
        source_label=("source_label", "first"), size=("external_id", "size")
    )
    if (frame.groupby("external_group_id").source_label.nunique() > 1).any():
        raise ValueError("an added product family has conflicting Usage labels")

    def tie(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    for label, groups in families.groupby("source_label", sort=True):
        counts = [0] * CV_FOLD_COUNT
        ordered = sorted(
            groups.index, key=lambda group: (-int(groups.loc[group, "size"]), tie(group))
        )
        for group in ordered:
            fold = min(
                range(CV_FOLD_COUNT),
                key=lambda value: (counts[value], totals[value], tie(f"{label}:{value}")),
            )
            size = int(groups.loc[group, "size"])
            assignments[group] = fold
            counts[fold] += size
            totals[fold] += size
    return frame.external_group_id.map(assignments).astype(int)


def combine_teacher_and_rare_usage(
    teacher: pd.DataFrame,
    admitted: pd.DataFrame,
    label_maps: dict[str, dict[str, Any]],
    *,
    root: str | Path = ROOT,
    check_files: bool = True,
) -> pd.DataFrame:
    """Append direct Usage supervision, preserving every original teacher field.

    Source occasions/styles map explicitly to the same-named teacher Usage class.
    Source-only intake partitions are replaced by this combined dataset's CV folds.
    Other targets stay blank on added rows, with false validity masks.
    """
    validate_splits(teacher)
    validate_external_manifest(admitted, root=root, check_files=check_files)
    usage_map = label_maps["usage"]
    if usage_map["label_column"] != "usage":
        raise ValueError("the combined dataset needs the existing Usage label map")
    if unknown := set(admitted.source_label) - set(usage_map["classes"]):
        raise ValueError(f"source labels are outside teacher Usage classes: {sorted(unknown)}")
    if "source_dataset" in teacher:
        raise ValueError("the base must be the original teacher dataset, not an expanded dataset")
    added = admitted.sort_values("external_id").reset_index(drop=True).copy()
    ids = list(range(EXTERNAL_ID_START, EXTERNAL_ID_START + len(added)))
    if set(ids).intersection(teacher.id.astype(int)):
        raise ValueError("reserved external IDs collide with teacher IDs")
    if set(added.sha256).intersection(teacher.sha256):
        raise ValueError("an admitted image exactly duplicates a teacher image")
    added_names = added.source_title.map(normalize_product_name)
    known_names = set(teacher.product_name_key.astype(str)) - {""}
    if set(added_names).intersection(known_names):
        raise ValueError(
            "an added product name matches a teacher group; resolve its identity first"
        )

    # Work in persisted scalar strings so the original teacher columns survive unchanged.
    base = teacher.copy()
    base["source_dataset"] = "teacher"
    base["source_id"] = base.id.astype(str)
    base["external_id"] = ""
    base["source_label"] = ""
    base["label_mapping_basis"] = "original_teacher_label"
    base["source_url"] = ""
    base["image_url"] = ""
    base["label_evidence"] = ""
    base["label_strength"] = ""
    base["source_group_id"] = ""
    base["rights_basis"] = "teacher_supplied"
    base["original_sha256"] = base.sha256
    records = []
    folds = assign_added_folds(added)
    for index, row in added.iterrows():
        record = {column: "" for column in base.columns}
        record.update(
            id=str(ids[index]),
            usage=str(row.source_label),
            productDisplayName=str(row.source_title),
            product_name_repaired=False,
            path=str(row.path),
            width=60,
            height=80,
            aspect_ratio=0.75,
            mode="RGB",
            format="PNG",
            file_size_bytes=(Path(root) / str(row.path)).stat().st_size if check_files else 0,
            sha256=str(row.sha256),
            product_name_key=str(added_names.iloc[index]),
            product_family_group="external:" + str(row.external_group_id),
            family_group_basis="source_identity_and_image_audit",
            duplicate_group="external:" + str(row.decoded_sha256),
            is_cross_role_exact_duplicate=False,
            is_cross_role_near_duplicate=False,
            is_cross_role_duplicate=False,
            has_conflicting_target_labels=False,
            partition="development",
            cv_fold=str(int(folds.iloc[index])),
            source_dataset=str(row.source),
            source_id=str(row.source_id),
            external_id=str(row.external_id),
            source_label=str(row.source_label),
            label_mapping_basis="exact_source_occasion_or_style_to_same_named_usage",
            source_url=str(row.source_url),
            image_url=str(row.image_url),
            label_evidence=str(row.label_evidence),
            label_strength=str(row.label_strength),
            source_group_id=str(row.external_group_id),
            rights_basis=str(row.rights_basis),
            original_sha256=str(row.original_sha256),
        )
        for target in TARGET_COLUMNS:
            record[f"has_{target}_label"] = target == "usage"
        records.append(record)
    combined = pd.concat([base, pd.DataFrame(records, columns=base.columns)], ignore_index=True)
    validate_splits(combined)
    validate_combined_usage(combined, teacher, admitted)
    return combined


def validate_combined_usage(
    combined: pd.DataFrame, teacher: pd.DataFrame, admitted: pd.DataFrame
) -> None:
    """Check direct target mapping, intact teacher rows and complete source membership."""
    validate_splits(combined)
    original = combined.loc[combined.source_dataset.eq("teacher"), teacher.columns]
    original = original.sort_values("id", key=lambda x: x.astype(int)).reset_index(drop=True)
    expected = teacher.sort_values("id", key=lambda x: x.astype(int)).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, expected, check_dtype=False)
    external = combined.loc[combined.source_dataset.ne("teacher")]
    if len(external) != len(admitted) or set(external.external_id) != set(admitted.external_id):
        raise ValueError("combined dataset must contain every admitted image exactly once")
    if not external.partition.eq("development").all():
        raise ValueError("added images belong to development; teacher holdout remains fixed")
    expected_usage = admitted.set_index("external_id").source_label
    if not external.usage.eq(external.external_id.map(expected_usage)).all():
        raise ValueError("added images must train the same Usage target with the explicit mapping")
    for target in TARGET_COLUMNS:
        flags = external[f"has_{target}_label"].astype(str).str.lower().isin({"true", "1"})
        if not flags.eq(target == "usage").all():
            raise ValueError(f"incorrect validity mask for added {target} labels")
        if target != "usage" and external[target].astype(str).str.strip().ne("").any():
            raise ValueError(f"no {target} labels were established for added images")
    if (external.groupby("source_group_id").cv_fold.nunique() > 1).any():
        raise ValueError("an added source family crosses combined CV folds")

"""Conservative product-family groups for split and quarantine decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    AUDIT_DIR,
    PREDICTION_MANIFEST_CSV,
    PRODUCT_FAMILIES_CSV,
    PRODUCT_NAME_REVIEW_CSV,
    PRODUCT_NAME_TRIAGE_JSON,
    TARGET_COLUMNS,
)
from fashion.data.hashing import write_deterministic_csv
from fashion.data.reviews import validate_review_ledger


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalize_product_name(value: Any) -> str:
    """Case-fold and collapse whitespace without deleting semantic tokens."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def find_exact_duplicate_label_conflicts(
    frame: pd.DataFrame,
    targets: Sequence[str] = TARGET_COLUMNS,
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
        fields = []
        for target in targets:
            valid = group.loc[_as_bool(group[f"has_{target}_label"]), target]
            if valid.astype(str).str.strip().nunique() > 1:
                fields.append(target)
        if fields:
            conflicts[str(sha256)] = tuple(fields)
    return conflicts


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            lower, upper = sorted((first_root, second_root))
            self.parent[upper] = lower


def _group_id(prefix: str, ids: Sequence[int]) -> str:
    payload = ",".join(map(str, sorted(ids))).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _union_equal_values(
    frame: pd.DataFrame,
    values: pd.Series,
    union: _UnionFind,
    eligible: pd.Series | None = None,
) -> None:
    first: dict[str, int] = {}
    for index, value in values.items():
        if eligible is not None and not bool(eligible.loc[index]):
            continue
        text = str(value)
        if not text:
            continue
        if text in first:
            union.union(int(index), first[text])
        else:
            first[text] = int(index)


def _review_counts(path: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not supplied", "reviewed_pairs": 0}
    review = pd.read_csv(path, keep_default_na=False)
    required = {
        "review_id",
        "family_key",
        "group_rows",
        "id_1",
        "id_2",
        "decision",
        "notes",
    }
    if missing := required.difference(review.columns):
        raise ValueError(f"product-name review is missing columns: {sorted(missing)}")
    audit = validate_review_ledger(review, path)
    if review["review_id"].duplicated().any():
        raise ValueError("product-name review IDs must be unique")
    if not set(review["decision"]).issubset({"same_or_variant", "different", "uncertain"}):
        raise ValueError("product-name review contains an unknown decision")
    lookup = manifest.set_index("id")["product_name_key"].astype(str).to_dict()
    # This triage was frozen after the earlier exact/collision quarantine and before
    # the new reviewed cross-role near-duplicate decisions were applied.
    review_scope = manifest[
        ~_as_bool(manifest["is_cross_role_exact_duplicate"])
        & ~_as_bool(manifest["has_conflicting_target_labels"])
    ]
    group_sizes = review_scope.groupby("product_name_key").size().to_dict()
    for row in review.itertuples(index=False):
        first_key = lookup.get(int(row.id_1))
        second_key = lookup.get(int(row.id_2))
        if not first_key or first_key != second_key or first_key != str(row.family_key):
            raise ValueError(f"product-name review pair {row.review_id} has a stale key")
        if int(row.group_rows) != int(group_sizes[first_key]):
            raise ValueError(f"product-name review pair {row.review_id} has a stale group size")
    bands = pd.cut(
        review["group_rows"].astype(int),
        bins=[1, 2, 5, 10, float("inf")],
        labels=["2", "3-5", "6-10", "11+"],
    )
    band_counts = bands.value_counts().to_dict()
    if any(int(band_counts.get(band, 0)) != 30 for band in ("2", "3-5", "6-10", "11+")):
        raise ValueError("product-name review must contain 30 pairs in each size band")
    counts = review["decision"].value_counts()
    return {
        **audit,
        "provisional_calls": len(review),
        "provisional_same_or_variant": int(counts.get("same_or_variant", 0)),
        "provisional_different": int(counts.get("different", 0)),
        "provisional_uncertain": int(counts.get("uncertain", 0)),
        "sampling": (
            "30 pre-policy crossing keys in each size band: 2, 3-5, 6-10, and "
            "11+ rows; exact/collision quarantine excluded"
        ),
        "interpretation": (
            "The name key is a conservative blocking signal, not proof of one SKU. False merges "
            "reduce effective sample size but cannot leak a name block across active partitions. "
            "The provisional calls are not human-validated evidence until team sign-off."
        ),
    }


def _load_pre_policy_triage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not supplied"}
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "source_split_sha256",
        "source_split_rows",
        "normalization",
        "scope",
        "crossing_groups",
        "rows_in_crossing_groups",
        "distinct_sha256_in_crossing_groups",
        "interpretation",
        "limitation",
    }
    if missing := required.difference(record):
        raise ValueError(f"pre-policy product-name triage is missing fields: {sorted(missing)}")
    if len(str(record["source_split_sha256"])) != 64:
        raise ValueError("pre-policy product-name triage has an invalid split digest")
    return record


def build_product_families(
    train_manifest_csv: str | Path | None = None,
    prediction_manifest_csv: str | Path = PREDICTION_MANIFEST_CSV,
    candidates_csv: str | Path = AUDIT_DIR / "near_duplicate_candidates.csv.gz",
    output_csv: str | Path = PRODUCT_FAMILIES_CSV,
    summary_output: str | Path = AUDIT_DIR / "product_family_summary.json",
    product_name_review_csv: str | Path = PRODUCT_NAME_REVIEW_CSV,
    product_name_triage_json: str | Path = PRODUCT_NAME_TRIAGE_JSON,
    targets: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Build family blocks after conservatively removing cross-role visual matches."""
    if train_manifest_csv is None:
        raise ValueError("the labelled manifest must be an explicit temporary build path")
    manifest = pd.read_csv(train_manifest_csv, keep_default_na=False).reset_index(drop=True)
    prediction = pd.read_csv(prediction_manifest_csv, keep_default_na=False)
    candidates = pd.read_csv(candidates_csv, keep_default_na=False)
    candidates["accepted_near_duplicate"] = _as_bool(candidates["accepted_near_duplicate"])

    labelled_inventory = manifest[["id", "sha256"]].copy()
    labelled_inventory["role"] = "labelled"
    prediction_inventory = prediction[["id", "sha256"]].copy()
    prediction_inventory["role"] = "prediction"
    inventory = pd.concat([labelled_inventory, prediction_inventory], ignore_index=True)
    inventory.reset_index(drop=True, inplace=True)
    inventory_index = {(str(row.role), int(row.id)): index for index, row in inventory.iterrows()}
    visual_union = _UnionFind(len(inventory))
    _union_equal_values(inventory, inventory["sha256"].astype(str), visual_union)
    accepted = candidates[candidates["accepted_near_duplicate"]]
    for row in accepted.itertuples(index=False):
        visual_union.union(
            inventory_index[(str(row.role_1), int(row.id_1))],
            inventory_index[(str(row.role_2), int(row.id_2))],
        )
    inventory["visual_root"] = [visual_union.find(index) for index in inventory.index]
    roles_by_root = inventory.groupby("visual_root")["role"].agg(lambda values: set(values))
    cross_roots = {
        int(root) for root, roles in roles_by_root.items() if roles == {"labelled", "prediction"}
    }
    cross_visual_ids = set(
        inventory.loc[
            inventory["visual_root"].isin(cross_roots) & inventory["role"].eq("labelled"),
            "id",
        ].astype(int)
    )
    role_counts_by_sha = inventory.groupby(["sha256", "role"]).size().unstack(fill_value=0)
    for role in ("labelled", "prediction"):
        if role not in role_counts_by_sha:
            role_counts_by_sha[role] = 0
    cross_exact_hashes = set(
        role_counts_by_sha[
            role_counts_by_sha["labelled"].gt(0) & role_counts_by_sha["prediction"].gt(0)
        ].index.astype(str)
    )

    conflicts = find_exact_duplicate_label_conflicts(manifest, targets)
    conflict_hashes = set(conflicts)
    manifest["product_name_key"] = manifest["productDisplayName"].map(normalize_product_name)
    manifest["is_cross_role_exact_duplicate"] = (
        manifest["sha256"].astype(str).isin(cross_exact_hashes)
    )
    manifest["is_cross_role_near_duplicate"] = (
        manifest["id"].astype(int).isin(cross_visual_ids)
        & ~manifest["is_cross_role_exact_duplicate"]
    )
    manifest["has_conflicting_target_labels"] = manifest["sha256"].astype(str).isin(conflict_hashes)
    manifest["conflicting_targets"] = (
        manifest["sha256"].astype(str).map(lambda digest: ",".join(conflicts.get(digest, ())))
    )
    reasons: list[str] = []
    for row in manifest.itertuples(index=False):
        row_reasons = []
        if row.is_cross_role_exact_duplicate:
            row_reasons.append("cross_role_exact_duplicate")
        elif row.is_cross_role_near_duplicate:
            row_reasons.append("cross_role_near_duplicate_pending_or_confirmed")
        if row.has_conflicting_target_labels:
            row_reasons.append("conflicting_labels_exact_sha")
        reasons.append(";".join(row_reasons))
    manifest["pre_quarantine_reason"] = reasons
    active = manifest["pre_quarantine_reason"].eq("")

    family_union = _UnionFind(len(manifest))
    _union_equal_values(manifest, manifest["sha256"].astype(str), family_union, active)
    _union_equal_values(manifest, manifest["product_name_key"], family_union, active)
    labelled_index = {int(item_id): int(index) for index, item_id in manifest["id"].items()}
    internal_edges: set[tuple[int, int]] = set()
    for row in accepted.itertuples(index=False):
        if row.role_1 != "labelled" or row.role_2 != "labelled":
            continue
        first = labelled_index[int(row.id_1)]
        second = labelled_index[int(row.id_2)]
        if active.loc[first] and active.loc[second]:
            family_union.union(first, second)
            internal_edges.add(tuple(sorted((first, second))))

    roots = [family_union.find(index) if active.loc[index] else index for index in manifest.index]
    root_members: dict[int, list[int]] = {}
    for index, root in enumerate(roots):
        root_members.setdefault(root, []).append(index)
    group_ids: dict[int, str] = {}
    bases: dict[int, str] = {}
    for root, members in root_members.items():
        ids = manifest.loc[members, "id"].astype(int).tolist()
        if not active.loc[members].all():
            group_ids[root] = _group_id("quarantine", ids)
            bases[root] = "quarantine"
            continue
        basis = []
        subset = manifest.loc[members]
        if subset["sha256"].duplicated().any():
            basis.append("exact_sha256")
        if subset.loc[subset["product_name_key"].ne(""), "product_name_key"].duplicated().any():
            basis.append("normalized_product_name")
        member_set = set(members)
        if any(first in member_set and second in member_set for first, second in internal_edges):
            basis.append("accepted_near_duplicate")
        group_ids[root] = _group_id("family", ids)
        bases[root] = "+".join(basis) if basis else "singleton"
    manifest["product_family_group"] = [group_ids[root] for root in roots]
    manifest["family_group_basis"] = [bases[root] for root in roots]

    output_columns = [
        "id",
        "product_name_key",
        "product_family_group",
        "family_group_basis",
        "is_cross_role_exact_duplicate",
        "is_cross_role_near_duplicate",
        "has_conflicting_target_labels",
        "conflicting_targets",
        "pre_quarantine_reason",
    ]
    result = manifest[output_columns].copy()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(
        result,
        output_csv,
        index=False,
    )

    active_rows = result[result["pre_quarantine_reason"].eq("")]
    name_groups = (
        active_rows[active_rows["product_name_key"].ne("")]
        .groupby("product_name_key")
        .agg(rows=("id", "size"), families=("product_family_group", "nunique"))
    )
    summary = {
        "schema_version": "3.0.0",
        "labelled_rows": len(result),
        "pre_quarantined_rows": int(result["pre_quarantine_reason"].ne("").sum()),
        "cross_role_exact_rows": int(result["is_cross_role_exact_duplicate"].sum()),
        "cross_role_near_rows_pending_or_confirmed": int(
            result["is_cross_role_near_duplicate"].sum()
        ),
        "conflicting_exact_sha_rows": int(result["has_conflicting_target_labels"].sum()),
        "active_family_groups": int(active_rows["product_family_group"].nunique()),
        "active_multirow_family_groups": int(
            active_rows.groupby("product_family_group").size().gt(1).sum()
        ),
        "largest_active_family_group": int(
            active_rows.groupby("product_family_group").size().max()
        ),
        "active_normalized_name_keys": len(name_groups),
        "active_name_keys_crossing_family_groups": int(name_groups["families"].gt(1).sum()),
        "product_name_review": _review_counts(Path(product_name_review_csv), manifest),
        "pre_policy_crossing_triage": _load_pre_policy_triage(Path(product_name_triage_json)),
        "policy": (
            "First quarantine exact and all pending-or-confirmed cross-role visual components, "
            "plus conflicting exact hashes. Then block each remaining normalized name, exact "
            "hash, and accepted visual component for train/validation/holdout allocation."
        ),
    }
    summary_output = Path(summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result

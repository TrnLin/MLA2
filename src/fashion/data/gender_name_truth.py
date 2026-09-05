"""Build an explicit name-based gender label variant over the canonical split."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from fashion.config import ROOT
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.splits import cv_assignment_digest, validate_splits

VARIANT_ID = "gender_name_truth_v1"
RULE_VERSION = "single_explicit_gender_cue_v1"
VARIANT_RELATIVE_PATH = Path("data/processed/variants") / VARIANT_ID
PATTERNS = {
    "Boys": r"\bboy(?:s|'s)?\b",
    "Girls": r"\bgirl(?:s|'s)?\b",
    "Men": r"\bmen(?:s|'s)?\b",
    "Women": r"\bwomen(?:s|'s)?\b",
    "Unisex": r"\bunisex\b",
}


def product_name_gender_cues(names: pd.Series) -> pd.DataFrame:
    """Use the frozen inspection rule; neither existing labels nor predictions decide ties."""
    normalized = names.fillna("").astype(str).str.lower().str.replace("’", "'", regex=False)
    hits = pd.DataFrame(
        {
            label: normalized.str.contains(pattern, regex=True)
            for label, pattern in PATTERNS.items()
        },
        index=names.index,
    )
    counts = hits.sum(axis=1)
    return pd.DataFrame(
        {
            "name_gender": hits.idxmax(axis=1).where(counts.eq(1), ""),
            "matched_cues": hits.apply(lambda row: "|".join(row.index[row]), axis=1),
            "cue_count": counts,
        },
        index=names.index,
    )


def make_name_truth_labels(splits: pd.DataFrame) -> pd.DataFrame:
    """Return one label row per eligible development image, without split assignments."""
    validate_splits(splits)
    eligible = splits.partition.eq("development") & splits.has_gender_label
    source = splits.loc[eligible].sort_values("id").reset_index(drop=True)
    if source.empty:
        raise ValueError("The canonical split has no labeled development gender rows")
    if unknown := set(source.gender) - set(PATTERNS):
        raise ValueError(f"Unexpected source gender classes: {sorted(unknown)}")
    cues = product_name_gender_cues(source.productDisplayName)
    labels = pd.concat(
        [source[["id", "gender"]].rename(columns={"gender": "original_gender"}), cues], axis=1
    )
    single = labels.cue_count.eq(1)
    labels["gender"] = labels.name_gender.where(single, labels.original_gender)
    labels["label_source"] = "original_no_cue"
    labels.loc[labels.cue_count.gt(1), "label_source"] = "original_multiple_cues"
    labels.loc[single, "label_source"] = "product_name"
    labels["changed"] = labels.gender.ne(labels.original_gender)
    return labels


def apply_name_truth_labels(splits: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Apply only the exact rule-derived labels; preserve every other canonical cell."""
    expected = make_name_truth_labels(splits)
    if labels.id.isna().any() or labels.id.duplicated().any():
        raise ValueError("Variant label IDs must be non-null and unique")
    try:
        pd.testing.assert_frame_equal(
            labels.sort_values("id").reset_index(drop=True),
            expected,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Variant labels do not match the canonical names and frozen rule"
        ) from error
    result = splits.copy(deep=True)
    eligible = result.id.isin(expected.id)
    mapping = expected.set_index("id").gender
    result.loc[eligible, "gender"] = result.loc[eligible, "id"].map(mapping)
    pd.testing.assert_frame_equal(result.drop(columns="gender"), splits.drop(columns="gender"))
    pd.testing.assert_frame_equal(result.loc[~eligible], splits.loc[~eligible])
    validate_splits(result)
    return result


def _csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def build_gender_name_truth_variant(root: str | Path = ROOT) -> dict[str, object]:
    """Create a reproducible label-only dataset; refuse to overwrite a different artifact."""
    root = Path(root)
    source_path = root / "data/processed/splits.csv"
    source_hash = compute_sha256(source_path)
    splits = load_splits(source_path)
    labels = make_name_truth_labels(splits)
    variant = apply_name_truth_labels(splits, labels)
    audit = labels.merge(
        splits[["id", "productDisplayName", "cv_fold", "product_family_group", "sha256", "path"]],
        on="id",
        validate="one_to_one",
    )
    changes = audit[audit.changed].copy()
    unclear = audit[audit.cue_count.ne(1)].copy()
    conflicts = audit.groupby("sha256").gender.nunique()
    conflicts = audit[audit.sha256.isin(conflicts[conflicts.gt(1)].index)].copy()
    conflicts = conflicts.sort_values(["sha256", "id"])
    folds = []
    for fold in range(5):
        rows = splits[splits.partition.eq("development") & splits.cv_fold.eq(fold)]
        changed = changes[changes.cv_fold.eq(fold)]
        folds.append(
            {
                "fold": fold,
                "validation_rows": len(rows),
                "changed_validation_labels": len(changed),
                "changed_training_labels": len(changes) - len(changed),
            }
        )
    counts = (
        pd.DataFrame(
            {
                "original": labels.original_gender.value_counts(),
                "name_truth": labels.gender.value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .sort_index()
        .rename_axis("gender")
        .reset_index()
    )
    summary = {
        "variant_id": VARIANT_ID,
        "rule_version": RULE_VERSION,
        "source_of_truth": "product name when exactly one explicit gender class cue is present",
        "unclear_name_policy": (
            "keep original gender; list separately; do not infer from images or models"
        ),
        "scope": "all labeled development rows, including validation rows in every canonical fold",
        "canonical_split_path": "data/processed/splits.csv",
        "canonical_split_sha256": source_hash,
        "canonical_cv_assignment_sha256": cv_assignment_digest(splits),
        "dataset_rows": len(variant),
        "development_label_rows": len(labels),
        "changed_labels": len(changes),
        "changed_product_families": changes.product_family_group.nunique(),
        "single_cue_rows": int(labels.cue_count.eq(1).sum()),
        "no_cue_rows": int(labels.cue_count.eq(0).sum()),
        "multiple_cue_rows": int(labels.cue_count.gt(1).sum()),
        "protected_rows_unchanged": int(splits.partition.ne("development").sum()),
        "image_rows_added_removed_or_moved": 0,
        "images_copied": False,
        "training_performed": False,
        "independent_blind_test": False,
        "interpretation": (
            "name-truth label sensitivity experiment proposed after development error inspection"
        ),
        "same_image_conflicting_variant_label_groups": conflicts.sha256.nunique(),
        "same_image_conflicting_variant_label_rows": len(conflicts),
        "duplicate_conflict_policy": (
            "retain per-name labels and canonical membership; report without dropping or regrouping"
        ),
        "patterns": PATTERNS,
        "folds": folds,
        "class_counts": counts.to_dict(orient="records"),
        "label_transitions": changes.groupby(["original_gender", "gender"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict(orient="records"),
    }
    payloads = {
        "labels.csv": _csv(labels),
        "changes.csv": _csv(changes),
        "unclear_names.csv": _csv(unclear),
        "same_image_label_conflicts.csv": _csv(conflicts),
        "class_counts.csv": _csv(counts),
    }
    summary["files"] = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in payloads.items()
    }
    payloads["summary.json"] = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    destination = root / VARIANT_RELATIVE_PATH
    if compute_sha256(source_path) != source_hash:
        raise ValueError("Canonical split changed during variant creation")
    if destination.exists():
        if all(
            (destination / name).is_file()
            and (destination / name).read_bytes() == text.encode("utf-8")
            for name, text in payloads.items()
        ):
            return summary
        raise FileExistsError(f"Different variant files already exist at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{VARIANT_ID}-", dir=destination.parent) as staging:
        stage = Path(staging) / VARIANT_ID
        stage.mkdir()
        for name, content in payloads.items():
            (stage / name).write_text(content, encoding="utf-8")
        stage.rename(destination)
    return summary


def load_gender_name_truth_variant(root: str | Path = ROOT) -> pd.DataFrame:
    """Read the canonical split first, verify this variant, then replace only gender."""
    root = Path(root)
    directory = root / VARIANT_RELATIVE_PATH
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    source = root / "data/processed/splits.csv"
    if summary.get("variant_id") != VARIANT_ID or summary.get("rule_version") != RULE_VERSION:
        raise ValueError("Unexpected name-truth dataset variant or rule")
    if summary.get("canonical_split_sha256") != compute_sha256(source):
        raise ValueError("Canonical split changed since the name-truth variant was created")
    for name in (
        "labels.csv",
        "changes.csv",
        "unclear_names.csv",
        "same_image_label_conflicts.csv",
        "class_counts.csv",
    ):
        if summary.get("files", {}).get(name) != compute_sha256(directory / name):
            raise ValueError(f"Name-truth artifact hash mismatch: {name}")
    splits = load_splits(source)
    labels = pd.read_csv(directory / "labels.csv", keep_default_na=False)
    result = apply_name_truth_labels(splits, labels)
    result.attrs["gender_label_variant"] = {
        "variant_id": VARIANT_ID,
        "rule_version": RULE_VERSION,
        "labels_sha256": summary["files"]["labels.csv"],
        "canonical_split_sha256": summary["canonical_split_sha256"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = build_gender_name_truth_variant(args.root)
    print(f"Created/verified {args.root / VARIANT_RELATIVE_PATH}")
    print(f"Changed {summary['changed_labels']} labels; all canonical splits preserved.")


if __name__ == "__main__":
    main()
